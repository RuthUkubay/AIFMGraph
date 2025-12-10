// aifm/test/test_graph_prefetch.cpp
extern "C" {
#include <runtime/runtime.h>
}

#include "device.hpp"
#include "manager.hpp"
#include "deref_scope.hpp"
#include "graph_adj.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <queue>
#include <random>
#include <vector>

using namespace far_memory;
using std::cout;
using std::endl;

constexpr uint64_t kCacheSize    = (128ULL << 20);
constexpr uint64_t kFarMemSize   = (4ULL  << 30);
constexpr uint32_t kNumGCThreads = 12;

/* Placement policies */
struct AllRemote : RemotingPolicy {
  uint16_t inline_capacity(Vid, uint32_t) const override { return 0; }
};

struct Local8 : RemotingPolicy {
  uint16_t inline_capacity(Vid, uint32_t deg) const override {
    return deg <= 8 ? deg : 0;
  }
};

/* Row of results */
struct Row {
  const char* policy;          // "All-remote" or "Local-8"
  const char* variant;         // "baseline" or "tailhint(th=16)"
  uint32_t    edge_work_iters; // synthetic per-edge work
  uint64_t    inline_any;
  uint64_t    remote_any;
  uint64_t    remote_bytes;
  double      bfs_us;
};

/* Banded generator: neighbors of u fall mostly in [u, u+W) */
static std::vector<std::pair<Vid,Vid>>
gen_banded_edges(uint64_t N, uint64_t E, uint32_t W, uint64_t seed=42) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<uint64_t> Uu(0, N - 1);
  std::uniform_int_distribution<uint32_t> Uw(0, W - 1);

  std::vector<std::pair<Vid,Vid>> edges;
  edges.reserve(E);
  for (uint64_t i = 0; i < E; ++i) {
    Vid u = static_cast<Vid>(Uu(rng));
    Vid v = static_cast<Vid>((u + Uw(rng)) % N);
    if (v == u) v = (u + 1) % N;
    edges.emplace_back(u, v);
  }
  return edges;
}

/* Tiny integer-mix loop to simulate per-edge compute. */
static inline void do_edge_work(volatile uint32_t &sink, Vid v, uint32_t edge_work) {
  uint32_t x = static_cast<uint32_t>(v) ^ 0x9e3779b9u;
  for (uint32_t i = 0; i < edge_work; ++i) {
    x = x * 1664525u + 1013904223u + i;
  }
  sink ^= x;  // keep it “live”
}

/*
 * BFS with optional "tail hint" and tunable per-edge compute.
 *
 * - edge_work: how much synthetic work to do per edge.
 * - use_tail_hint: if true, we pre-hint the remote tail for high-degree vertices.
 * - tail_threshold: only hint if remote tail length >= this.
 */
static double bfs_time_us_with_tail_hint(GraphAdj &G, Vid src,
                                         uint32_t iters,
                                         uint32_t edge_work,
                                         bool use_tail_hint,
                                         uint32_t tail_threshold) {
  using clk = std::chrono::high_resolution_clock;
  const uint64_t n = G.num_vertices();
  if (src >= n) return 0.0;

  auto run_once = [&](){
    std::vector<int> dist(n, -1);
    std::queue<Vid> q;
    volatile uint32_t sink = 0;      // accumulates work

    dist[src] = 0;
    q.push(src);

    while (!q.empty()) {
      Vid u = q.front();
      q.pop();

      // Optional tail hint: look at the header first, then hint the tail
      // *before* we traverse neighbors. All derefs are scoped and bounded.
      if (use_tail_hint && tail_threshold > 0) {
        uint32_t tail_len = 0;
        {
          DerefScope hs;
          auto h = G.header_info(u, hs);
          tail_len = (h.degree > h.inline_len) ? (h.degree - h.inline_len) : 0;
        }
        if (tail_len >= tail_threshold) {
          // This just maps the tail once via its own scope and returns.
          G.hint_tail_present(u);
        }
      }

      // Now do the actual neighbor traversal.
      {
        DerefScope s;
        G.for_each_neighbor(u, s, [&](Vid v){
          // Simulate compute on every edge so comparison is fair.
          if (edge_work) {
            do_edge_work(sink, v, edge_work);
          }
          if (dist[v] == -1) {
            dist[v] = dist[u] + 1;
            q.push(v);
          }
        });
      }
    }
  };

  // Warm run
  run_once();

  auto t0 = clk::now();
  for (uint32_t i = 0; i < iters; ++i)
    run_once();
  auto t1 = clk::now();

  return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
         / static_cast<double>(iters);
}

/* Build a graph, gather layout stats, run BFS with/without tail hint. */
static Row run_case_with_work(FarMemManager* mgr,
                              const std::vector<std::pair<Vid,Vid>>& edges,
                              const char* policy_name,
                              const RemotingPolicy& pol,
                              uint32_t edge_work,
                              bool use_tail_hint,
                              uint32_t tail_threshold) {
  Row r{};
  r.policy          = policy_name;
  r.variant         = use_tail_hint ? "tailhint(th=16)" : "baseline";
  r.edge_work_iters = edge_work;

  // Infer N from edges' max ID
  uint64_t N = 0;
  for (auto &e : edges) {
    N = std::max<uint64_t>(N, std::max<uint64_t>(e.first, e.second));
  }
  N += 1;

  GraphAdj G(mgr, N, pol);
  G.build_from_edges(edges);

  // Placement stats
  uint64_t inline_any=0, remote_any=0, remote_bytes=0;
  for (uint64_t u = 0; u < N; ++u) {
    DerefScope s;
    auto h = G.header_info(u, s);
    if (h.inline_len > 0) inline_any++;
    if (h.degree > h.inline_len) {
      remote_any++;
      remote_bytes += (h.degree - h.inline_len) * sizeof(Vid);
    }
  }
  r.inline_any   = inline_any;
  r.remote_any   = remote_any;
  r.remote_bytes = remote_bytes;

  // BFS timing (5 iterations)
  r.bfs_us = bfs_time_us_with_tail_hint(G, /*src=*/0,
                                        /*iters=*/5,
                                        /*edge_work=*/edge_work,
                                        /*use_tail_hint=*/use_tail_hint,
                                        /*tail_threshold=*/tail_threshold);
  return r;
}

/* Main test entry point for the AIFM runtime. */
static void _main(void*) {
  std::unique_ptr<FarMemManager> manager(
      FarMemManagerFactory::build(kCacheSize, kNumGCThreads,
                                  new FakeDevice(kFarMemSize)));

  // Workload
  const uint64_t N = 200000;
  const uint64_t E = 1200000;
  const uint32_t W = 64;   // mild spatial locality
  auto edges = gen_banded_edges(N, E, W);

  AllRemote all_remote;
  Local8    local_8;

  cout << "Graph: |V|=" << N << " |E|=" << E
       << " (banded window=" << W << ")\n";
  cout << "Policy,Variant,EdgeWorkIters,Vertices w/ local neighbors,"
          "Vertices w/ remote,Remote bytes,BFS per-iter (µs),"
          "Speedup vs baseline (same EdgeWork)\n";

  const uint32_t work_levels[] = {0, 64, 256};  // no work, light, heavier
  const uint32_t tail_threshold = 16;           // only hint for "big" tails

  for (uint32_t work : work_levels) {
    // --- All-remote baseline and tail-hint ---
    Row base = run_case_with_work(manager.get(), edges,
                                  "All-remote", all_remote,
                                  /*edge_work=*/work,
                                  /*use_tail_hint=*/false,
                                  tail_threshold);
    cout << base.policy << "," << base.variant << "," << base.edge_work_iters << ","
         << base.inline_any << "," << base.remote_any << ","
         << base.remote_bytes << "," << base.bfs_us << ",—\n";

    Row hint = run_case_with_work(manager.get(), edges,
                                  "All-remote", all_remote,
                                  /*edge_work=*/work,
                                  /*use_tail_hint=*/true,
                                  tail_threshold);
    {
      double speedup = (base.bfs_us > 0)
                       ? (1.0 - (hint.bfs_us / base.bfs_us)) * 100.0
                       : 0.0;
      cout << hint.policy << "," << hint.variant << "," << hint.edge_work_iters << ","
           << hint.inline_any << "," << hint.remote_any << ","
           << hint.remote_bytes << "," << hint.bfs_us << ","
           << (speedup >= 0 ? "+" : "") << speedup << "%\n";
    }

    // --- Local-8 baseline and tail-hint ---
    Row baseL = run_case_with_work(manager.get(), edges,
                                   "Local-8", local_8,
                                   /*edge_work=*/work,
                                   /*use_tail_hint=*/false,
                                   tail_threshold);
    cout << baseL.policy << "," << baseL.variant << "," << baseL.edge_work_iters << ","
         << baseL.inline_any << "," << baseL.remote_any << ","
         << baseL.remote_bytes << "," << baseL.bfs_us << ",—\n";

    Row hintL = run_case_with_work(manager.get(), edges,
                                   "Local-8", local_8,
                                   /*edge_work=*/work,
                                   /*use_tail_hint=*/true,
                                   tail_threshold);
    {
      double speedup = (baseL.bfs_us > 0)
                       ? (1.0 - (hintL.bfs_us / baseL.bfs_us)) * 100.0
                       : 0.0;
      cout << hintL.policy << "," << hintL.variant << "," << hintL.edge_work_iters << ","
           << hintL.inline_any << "," << hintL.remote_any << ","
           << hintL.remote_bytes << "," << hintL.bfs_us << ","
           << (speedup >= 0 ? "+" : "") << speedup << "%\n";
    }
  }

  cout << "Done.\n";
}

/* Standard AIFM runtime glue. */
int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " [cfg_file]\n";
    return -EINVAL;
  }
  int ret = runtime_init(argv[1], _main, nullptr);
  if (ret) {
    std::cerr << "failed to start runtime\n";
    return ret;
  }
  return 0;
}
