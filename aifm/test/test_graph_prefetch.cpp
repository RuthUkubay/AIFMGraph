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

/* ------------------- Placement policies ------------------- */
struct AllRemote : RemotingPolicy {
  uint16_t inline_capacity(Vid, uint32_t) const override { return 0; }
};
struct Local8 : RemotingPolicy {
  uint16_t inline_capacity(Vid, uint32_t deg) const override {
    return deg <= 8 ? deg : 0;
  }
};

/* ------------------- A graph with spatially local tails ------------------- */
// "Banded" generator: neighbors of u mostly sit in [u, u+W), making tails
// contiguous-ish so prefetch/warm can help.
static std::vector<std::pair<Vid,Vid>>
gen_banded_edges(uint64_t N, uint64_t E, uint32_t window, uint64_t seed=42) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<uint64_t> Uv(0, N - 1);
  std::uniform_int_distribution<uint32_t> Uw(0, window - 1);

  std::vector<std::pair<Vid,Vid>> edges;
  edges.reserve(E);
  for (uint64_t i = 0; i < E; ++i) {
    Vid u = static_cast<Vid>(Uv(rng));
    Vid v = static_cast<Vid>((u + Uw(rng)) % N); // keep neighbors near u
    if (v == u) v = (u + 1) % N;
    edges.emplace_back(u, v);
  }
  return edges;
}

/* ------------------- RAII prefetch guard (scoped) ------------------- */
struct PrefetchGuard {
  GraphAdj &G;
  bool armed;
  explicit PrefetchGuard(GraphAdj &g, uint32_t distance)
      : G(g), armed(distance > 0) {
    if (armed) G.enable_header_static_prefetch(distance);
  }
  ~PrefetchGuard() {
    if (armed) G.disable_header_prefetch();
  }
};

/* ------------------- BFS variants ------------------- */

// Baseline BFS: plain queue, no prefetch.
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

// Header-prefetch BFS: process vertices in ascending ID per "level"
// AND enable static prefetch on headers.
static std::vector<int> bfs_header_prefetch(GraphAdj &G, Vid src, uint32_t pf_dist) {
  PrefetchGuard guard(G, pf_dist);  // scoped: on enter/exit

  const uint64_t n = G.num_vertices();
  std::vector<int> dist(n, -1);
  if (src >= n) return dist;

  std::vector<Vid> curr, next;
  curr.reserve(1024); next.reserve(1024);
  dist[src] = 0; curr.push_back(src);

  while (!curr.empty()) {
    // Make header access nearly sequential → helps static prefetch.
    std::sort(curr.begin(), curr.end());

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

// Header + tiny tail warmup: like above, but “peek” K tail entries first.
static std::vector<int> bfs_header_and_tail_warm(GraphAdj &G, Vid src,
                                                 uint32_t pf_dist,
                                                 uint32_t peek_k) {
  PrefetchGuard guard(G, pf_dist);  // scoped: on enter/exit

  const uint64_t n = G.num_vertices();
  std::vector<int> dist(n, -1);
  if (src >= n) return dist;

  std::vector<Vid> curr, next;
  curr.reserve(1024); next.reserve(1024);
  dist[src] = 0; curr.push_back(src);

  while (!curr.empty()) {
    std::sort(curr.begin(), curr.end());

    for (Vid u : curr) {
      DerefScope s;
      auto nv = G.neighbors(u, s);

      // Warm a few tail entries (bounded).
      uint32_t warm = std::min(peek_k, nv.tail_len);
      for (uint32_t i = 0; i < warm; ++i) {
        volatile Vid tmp = nv.tail_ptr[i];
        (void)tmp;
      }

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
  uint32_t pf_distance; // 0 = no prefetch
  uint32_t tail_peek;   // 0 = no tail warm
};

static void time_and_report(GraphAdj& G, Vid src,
                            uint32_t iters,
                            uint32_t pf_distance,
                            uint32_t tail_peek,
                            double &out_us) {
  using clk = std::chrono::high_resolution_clock;

  // Warm
  (void)(pf_distance == 0 && tail_peek == 0
         ? bfs_baseline(G, src)
         : (tail_peek == 0 ? bfs_header_prefetch(G, src, pf_distance)
                           : bfs_header_and_tail_warm(G, src, pf_distance, tail_peek)));

  auto t0 = clk::now();
  for (uint32_t i = 0; i < iters; ++i) {
    if (pf_distance == 0 && tail_peek == 0) {
      (void)bfs_baseline(G, src);
    } else if (tail_peek == 0) {
      (void)bfs_header_prefetch(G, src, pf_distance);
    } else {
      (void)bfs_header_and_tail_warm(G, src, pf_distance, tail_peek);
    }
  }
  auto t1 = clk::now();
  out_us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
           / static_cast<double>(iters);
}

static void do_work(FarMemManager* mgr) {
  // Make a workload where prefetch can help:
  // - banded tails → spatial locality
  // - level-sorted BFS → header stride = 1
  const uint64_t N = 200000;
  const uint64_t E = 1200000;
  const uint32_t W = 64;     // band window (neighbors near u)
  const uint32_t iters = 5;
  auto edges = gen_banded_edges(N, E, W);

  AllRemote all_remote;
  Local8    local_8;

  RunCfg runs[] = {
    {"All-remote / baseline",               all_remote, 0,   0},
    {"All-remote / header-pf(d=64)",        all_remote, 64,  0},
    {"All-remote / header-pf+tail-peek(4)", all_remote, 64,  4},

    {"Local-8 / baseline",                  local_8,    0,   0},
    {"Local-8 / header-pf(d=64)",           local_8,    64,  0},
    {"Local-8 / header-pf+tail-peek(4)",    local_8,    64,  4},
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

    double us = 0.0;
    time_and_report(G, /*src=*/0, iters, cfg.pf_distance, cfg.tail_peek, us);

    cout << cfg.name
         << " | inline_any=" << inline_any
         << " remote_any=" << remote_any
         << " remote_bytes=" << remote_bytes
         << " | BFS per-iter (µs)=" << us << "\n";
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
