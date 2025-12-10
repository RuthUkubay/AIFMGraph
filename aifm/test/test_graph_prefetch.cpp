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
#include <thread>
#include <vector>

using namespace far_memory;
using std::cout;
using std::endl;

constexpr uint64_t kCacheSize    = (128ULL << 20);
constexpr uint64_t kFarMemSize   = (4ULL  << 30);
constexpr uint32_t kNumGCThreads = 12;

/* ------------------- Placement policies ------------------- */
struct AllRemote : RemotingPolicy {
  uint16_t inline_capacity(Vid, uint32_t) const override { return 0; }
};
struct Local8 : RemotingPolicy {
  uint16_t inline_capacity(Vid, uint32_t deg) const override {
    return deg <= 8 ? deg : 0;
  }
};

/* ------------------- Banded edges (spatial locality) ------------------- */
static std::vector<std::pair<Vid,Vid>>
gen_banded_edges(uint64_t N, uint64_t E, uint32_t window, uint64_t seed=42) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<uint64_t> Uv(0, N - 1);
  std::uniform_int_distribution<uint32_t> Uw(0, window - 1);

  std::vector<std::pair<Vid,Vid>> edges;
  edges.reserve(E);
  for (uint64_t i = 0; i < E; ++i) {
    Vid u = static_cast<Vid>(Uv(rng));
    Vid v = static_cast<Vid>((u + Uw(rng)) % N);
    if (v == u) v = (u + 1) % N;
    edges.emplace_back(u, v);
  }
  return edges;
}

/* ------------------- BFS variants ------------------- */

// Plain BFS
static std::vector<int> bfs_baseline(GraphAdj &G, Vid src) {
  const uint64_t n = G.num_vertices();
  std::vector<int> dist(n, -1);
  if (src >= n) return dist;

  std::queue<Vid> q;
  dist[src] = 0; q.push(src);

  while (!q.empty()) {
    Vid u = q.front(); q.pop();
    DerefScope s;
    auto nv = G.neighbors(u, s);
    for (uint32_t i = 0; i < nv.inline_len; ++i) {
      Vid v = nv.inline_ptr[i];
      if (dist[v] == -1) { dist[v] = dist[u] + 1; q.push(v); }
    }
    for (uint32_t i = 0; i < nv.tail_len; ++i) {
      Vid v = nv.tail_ptr[i];
      if (dist[v] == -1) { dist[v] = dist[u] + 1; q.push(v); }
    }
  }
  return dist;
}

// Synchronous-warm BFS:
//  (a) sort current work by vertex id → sequential header touches
//  (b) warm headers (deref) for the batch
//  (c) optionally warm first K entries of each tail
static std::vector<int> bfs_sync_warm(GraphAdj &G, Vid src,
                                      uint32_t tail_peek_k) {
  const uint64_t n = G.num_vertices();
  std::vector<int> dist(n, -1);
  if (src >= n) return dist;

  std::vector<Vid> curr, next;
  curr.reserve(1024); next.reserve(1024);
  dist[src] = 0; curr.push_back(src);

  while (!curr.empty()) {
    std::sort(curr.begin(), curr.end());

    // warm headers for this batch
    {
      std::vector<uint64_t> batch(curr.begin(), curr.end());
      G.warm_headers_sorted_span(batch.data(), (uint32_t)batch.size());
    }

    // warm tiny prefix of tails
    if (tail_peek_k > 0) {
      for (Vid u : curr) G.warm_tail_prefix(u, tail_peek_k);
    }

    // expand
    for (Vid u : curr) {
      DerefScope s;
      auto nv = G.neighbors(u, s);
      for (uint32_t i = 0; i < nv.inline_len; ++i) {
        Vid v = nv.inline_ptr[i];
        if (dist[v] == -1) { dist[v] = dist[u] + 1; next.push_back(v); }
      }
      for (uint32_t i = 0; i < nv.tail_len; ++i) {
        Vid v = nv.tail_ptr[i];
        if (dist[v] == -1) { dist[v] = dist[u] + 1; next.push_back(v); }
      }
    }
    curr.swap(next);
    next.clear();
  }
  return dist;
}

/* ------------------- Harness ------------------- */
struct RunCfg {
  const char* name;
  const RemotingPolicy& pol;
  uint32_t tail_peek;   // 0 = no tail warm
};

static void time_and_report(GraphAdj& G, Vid src,
                            uint32_t iters,
                            uint32_t tail_peek,
                            double &out_us,
                            bool use_sync_warm) {
  using clk = std::chrono::high_resolution_clock;

  // Warm once
  (void)(use_sync_warm ? bfs_sync_warm(G, src, tail_peek)
                       : bfs_baseline(G, src));

  auto t0 = clk::now();
  for (uint32_t i = 0; i < iters; ++i) {
    if (use_sync_warm) {
      (void)bfs_sync_warm(G, src, tail_peek);
    } else {
      (void)bfs_baseline(G, src);
    }
  }
  auto t1 = clk::now();
  out_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
           / static_cast<double>(iters);
}

static void do_work(FarMemManager* mgr) {
  const uint64_t N = 200000;
  const uint64_t E = 1200000;
  const uint32_t W = 64;     // band window
  const uint32_t iters = 5;
  auto edges = gen_banded_edges(N, E, W);

  AllRemote all_remote;
  Local8    local_8;

  // Runs: baseline vs synchronous warm (peek K tail entries)
  RunCfg runs[] = {
    {"All-remote / baseline",            all_remote, 0},
    {"All-remote / sync-warm + tail(4)", all_remote, 4},

    {"Local-8 / baseline",               local_8,    0},
    {"Local-8 / sync-warm + tail(4)",    local_8,    4},
  };

  cout << "Graph: |V|=" << N << " |E|=" << E
       << " (banded window=" << W << ")\n";

  for (auto &cfg : runs) {
    GraphAdj G(mgr, N, cfg.pol);
    G.build_from_edges(edges);

    // layout stats
    uint64_t remote_bytes = 0, inline_any = 0, remote_any = 0;
    for (uint64_t u = 0; u < N; ++u) {
      DerefScope s;
      auto h = G.header_info(u, s);
      if (h.inline_len > 0) inline_any++;
      if (h.degree > h.inline_len) {
        remote_any++;
        remote_bytes += (h.degree - h.inline_len) * sizeof(Vid);
      }
    }

    double base_us = 0.0, warm_us = 0.0;

    time_and_report(G, /*src=*/0, iters,
                    /*tail_peek=*/cfg.tail_peek, base_us,
                    /*use_sync_warm=*/false);

    // small pause to let GC settle before next timed path on the same G
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    time_and_report(G, /*src=*/0, iters,
                    /*tail_peek=*/cfg.tail_peek, warm_us,
                    /*use_sync_warm=*/true);

    cout << cfg.name
         << " | inline_any=" << inline_any
         << " remote_any=" << remote_any
         << " remote_bytes=" << remote_bytes
         << " | BFS baseline_us=" << base_us
         << " | sync_warm_us=" << warm_us
         << " | delta=" << (base_us - warm_us)
         << " (" << (100.0 * (base_us - warm_us) / base_us) << "% faster)"
         << "\n";

    // extra pause before destroying G to be extra safe on devices
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  cout << "Done.\n";
}

static void _main(void*) {
  std::unique_ptr<FarMemManager> manager(
      FarMemManagerFactory::build(kCacheSize, kNumGCThreads, new FakeDevice(kFarMemSize)));
  do_work(manager.get());
}

int main(int argc, char* argv[]) {
  if (argc < 2) { std::cerr << "usage: " << argv[0] << " [cfg_file]\n"; return -EINVAL; }
  int ret = runtime_init(argv[1], _main, nullptr);
  if (ret) { std::cerr << "failed to start runtime\n"; return ret; }
  return 0;
}
