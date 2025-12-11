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

struct AllRemote : RemotingPolicy {
  uint16_t inline_capacity(Vid, uint32_t) const override { return 0; }
};
struct Local8 : RemotingPolicy {
  uint16_t inline_capacity(Vid, uint32_t deg) const override {
    return deg <= 8 ? deg : 0;
  }
};

struct Row {
  const char* policy;
  const char* variant;
  uint32_t    edge_work_iters;
  uint64_t    inline_any;
  uint64_t    remote_any;
  uint64_t    remote_bytes;
  double      bfs_us;
};

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

static inline void do_edge_work(volatile uint32_t &sink, Vid v, uint32_t edge_work) {
  uint32_t x = static_cast<uint32_t>(v) ^ 0x9e3779b9u;
  for (uint32_t i = 0; i < edge_work; ++i)
    x = x * 1664525u + 1013904223u + i;
  sink ^= x;
}

// A safe BFS that optionally hints large tails before traversal.
static double bfs_time_us(GraphAdj &G, Vid src,
                          uint32_t iters,
                          uint32_t edge_work,
                          bool use_tail_hint,
                          uint32_t tail_threshold) {
  using clk = std::chrono::high_resolution_clock;
  const uint64_t n = G.num_vertices();
  if (src >= n) return 0.0;

  auto run_once = [&]() {
    std::vector<int> dist(n, -1);
    std::queue<Vid> q;
    volatile uint32_t sink = 0;
    dist[src] = 0;
    q.push(src);

    while (!q.empty()) {
      Vid u = q.front();
      q.pop();

      if (use_tail_hint && tail_threshold > 0) {
        DerefScope hs;
        auto h = G.header_info(u, hs);
        const uint32_t tail_len = (h.degree > h.inline_len) ?
                                  (h.degree - h.inline_len) : 0;
        if (tail_len >= tail_threshold)
          G.hint_tail_present(u);
      }

      DerefScope s;
      G.for_each_neighbor(u, s, [&](Vid v) {
        if (edge_work) do_edge_work(sink, v, edge_work);
        if (dist[v] == -1) { dist[v] = dist[u] + 1; q.push(v); }
      });
    }
  };

  run_once(); // warm
  auto t0 = clk::now();
  for (uint32_t i = 0; i < iters; ++i) run_once();
  auto t1 = clk::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
         / static_cast<double>(iters);
}

static Row run_case(FarMemManager* mgr,
                    const std::vector<std::pair<Vid,Vid>>& edges,
                    const char* policy_name,
                    const RemotingPolicy& pol,
                    uint32_t edge_work,
                    bool use_tail_hint,
                    uint32_t tail_threshold) {
  Row r{};
  r.policy = policy_name;
  r.variant = use_tail_hint ? "tailhint(th=16)" : "baseline";
  r.edge_work_iters = edge_work;

  // Rebuild graph fresh in its own scope each run
  uint64_t N = 0;
  for (auto &e : edges)
    N = std::max<uint64_t>(N, std::max<uint64_t>(e.first, e.second));
  N += 1;

  {
    auto G = std::make_unique<GraphAdj>(mgr, N, pol);
    G->build_from_edges(edges);

    uint64_t inline_any=0, remote_any=0, remote_bytes=0;
    for (uint64_t u = 0; u < N; ++u) {
      DerefScope s;
      auto h = G->header_info(u, s);
      if (h.inline_len > 0) inline_any++;
      if (h.degree > h.inline_len) {
        remote_any++;
        remote_bytes += (h.degree - h.inline_len) * sizeof(Vid);
      }
    }
    r.inline_any   = inline_any;
    r.remote_any   = remote_any;
    r.remote_bytes = remote_bytes;

    r.bfs_us = bfs_time_us(*G, 0, 5, edge_work, use_tail_hint, tail_threshold);
  }

  return r;
}

static void _main(void*) {
  std::unique_ptr<FarMemManager> manager(
      FarMemManagerFactory::build(kCacheSize, kNumGCThreads,
                                  new FakeDevice(kFarMemSize)));

  const uint64_t N = 200000, E = 1200000;
  const uint32_t W = 64;
  auto edges = gen_banded_edges(N, E, W);

  AllRemote all_remote;
  Local8 local_8;

  cout << "Graph: |V|=" << N << " |E|=" << E
       << " (banded window=" << W << ")\n";
  cout << "Policy,Variant,EdgeWorkIters,Vertices w/ local neighbors,"
          "Vertices w/ remote,Remote bytes,BFS per-iter (µs),Speedup vs baseline\n";

  const uint32_t work_levels[] = {0, 64, 256};
  const uint32_t tail_th = 16;

  for (uint32_t work : work_levels) {
    // --- All-remote ---
    Row base = run_case(manager.get(), edges, "All-remote", all_remote, work, false, tail_th);
    cout << base.policy << "," << base.variant << "," << base.edge_work_iters << ","
         << base.inline_any << "," << base.remote_any << ","
         << base.remote_bytes << "," << base.bfs_us << ",—\n";

    Row hint = run_case(manager.get(), edges, "All-remote", all_remote, work, true, tail_th);
    double sp1 = (base.bfs_us > 0) ? (1.0 - (hint.bfs_us / base.bfs_us)) * 100.0 : 0.0;
    cout << hint.policy << "," << hint.variant << "," << hint.edge_work_iters << ","
         << hint.inline_any << "," << hint.remote_any << ","
         << hint.remote_bytes << "," << hint.bfs_us << ","
         << (sp1 >= 0 ? "+" : "") << sp1 << "%\n";

    // --- Local-8 ---
    Row baseL = run_case(manager.get(), edges, "Local-8", local_8, work, false, tail_th);
    cout << baseL.policy << "," << baseL.variant << "," << baseL.edge_work_iters << ","
         << baseL.inline_any << "," << baseL.remote_any << ","
         << baseL.remote_bytes << "," << baseL.bfs_us << ",—\n";

    Row hintL = run_case(manager.get(), edges, "Local-8", local_8, work, true, tail_th);
    double sp2 = (baseL.bfs_us > 0) ? (1.0 - (hintL.bfs_us / baseL.bfs_us)) * 100.0 : 0.0;
    cout << hintL.policy << "," << hintL.variant << "," << hintL.edge_work_iters << ","
         << hintL.inline_any << "," << hintL.remote_any << ","
         << hintL.remote_bytes << "," << hintL.bfs_us << ","
         << (sp2 >= 0 ? "+" : "") << sp2 << "%\n";
  }

  cout << "Done.\n";
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " [cfg_file]\n";
    return -EINVAL;
  }
  return runtime_init(argv[1], _main, nullptr);
}
