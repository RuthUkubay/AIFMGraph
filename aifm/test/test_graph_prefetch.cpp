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

/* Policies */
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
  const char* variant;   // e.g., "baseline", "header(d=64)", "tailwarm(th=16)"
  uint64_t inline_any;
  uint64_t remote_any;
  uint64_t remote_bytes;
  double   bfs_us;
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

/* BFS baseline using for_each_neighbor (safe for all layouts) */
static double bfs_baseline_time_us(GraphAdj &G, Vid src, uint32_t iters) {
  using clk = std::chrono::high_resolution_clock;
  const uint64_t n = G.num_vertices();
  if (src >= n) return 0.0;

  auto run_once = [&](){
    std::vector<int> dist(n, -1);
    std::queue<Vid> q;
    dist[src] = 0; q.push(src);

    while (!q.empty()) {
      Vid u = q.front(); q.pop();
      DerefScope s;
      G.for_each_neighbor(u, s, [&](Vid v){
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

/* BFS with tiny tail warm (peek K entries) right before expansion */
static double bfs_tailpeek_time_us(GraphAdj &G, Vid src, uint32_t peek_k, uint32_t iters) {
  using clk = std::chrono::high_resolution_clock;
  const uint64_t n = G.num_vertices();
  if (src >= n) return 0.0;

  auto run_once = [&](){
    std::vector<int> dist(n, -1);
    std::queue<Vid> q;
    dist[src] = 0; q.push(src);

    while (!q.empty()) {
      Vid u = q.front(); q.pop();
      DerefScope s;

      // **Only now**, because we will expand u, warm a tiny prefix of tail.
      // This maps the tail and touches K entries at most (bounded cost).
      G.warm_tail_prefix(u, peek_k, s);

      // Then do the regular neighbor iteration.
      G.for_each_neighbor(u, s, [&](Vid v){
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

// Gated tail warm-up: if a vertex has a large enough remote tail,
// call hint_tail_present(u) first (to start the fetch), then iterate neighbors.
// Tail threshold keeps it cheap and focused on vertices where it matters.
static double bfs_time_us_tailwarm(GraphAdj &G, Vid src,
                                   uint32_t iters,
                                   uint32_t tail_threshold) {
  using clk = std::chrono::high_resolution_clock;
  const uint64_t n = G.num_vertices();
  if (src >= n) return 0.0;

  auto run_once = [&](){
    std::vector<int> dist(n, -1);
    std::queue<Vid> q;
    dist[src] = 0; q.push(src);

    while (!q.empty()) {
      Vid u = q.front(); q.pop();
      DerefScope s;

      // Check tail size from header; if big enough, warm it.
      auto h = G.header_info(u, s);
      const uint32_t tail_len = (h.degree > h.inline_len) ? (h.degree - h.inline_len) : 0;
      if (tail_len >= tail_threshold) {
        // This only hints the tail; it doesn't read entries or allocate extra buffers.
        G.hint_tail_present(u);
      }

      // Now do the normal walk (inline first, tail second).
      G.for_each_neighbor(u, s, [&](Vid v){
        if (dist[v] == -1) { dist[v] = dist[u] + 1; q.push(v); }
      });
    }
  };

  // warm
  run_once();
  auto t0 = clk::now();
  for (uint32_t i = 0; i < iters; ++i) run_once();
  auto t1 = clk::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
         / static_cast<double>(iters);
}

/* BFS that sorts each frontier by vertex ID before expanding (improves locality). */
static double bfs_sorted_frontier_time_us(GraphAdj &G, Vid src, uint32_t iters) {
  using clk = std::chrono::high_resolution_clock;
  const uint64_t n = G.num_vertices();
  if (src >= n) return 0.0;

  auto run_once = [&](){
    std::vector<int> dist(n, -1);
    std::vector<Vid> curr, next;
    curr.reserve(1024); next.reserve(1024);
    dist[src] = 0; curr.push_back(src);

    while (!curr.empty()) {
      // Key idea: make header/tail touches nearly sequential
      std::sort(curr.begin(), curr.end());

      for (Vid u : curr) {
        DerefScope s;
        G.for_each_neighbor(u, s, [&](Vid v){
          if (dist[v] == -1) { dist[v] = dist[u] + 1; next.push_back(v); }
        });
      }
      curr.swap(next);
      next.clear();
    }
  };

  run_once(); // warm
  auto t0 = clk::now();
  for (uint32_t i = 0; i < iters; ++i) run_once();
  auto t1 = clk::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
         / static_cast<double>(iters);
}

// Adaptive frontier-sort: only sort a level when it's big AND tail-heavy
static double bfs_frontier_sort_adaptive_time_us(GraphAdj &G, Vid src,
                                                 uint32_t sort_min_frontier,   // e.g., 4096
                                                 uint32_t tail_threshold,      // e.g., 64
                                                 uint32_t iters) {
  using clk = std::chrono::high_resolution_clock;
  const uint64_t n = G.num_vertices();
  if (src >= n) return 0.0;

  auto run_once = [&](){
    std::vector<int> dist(n, -1);
    std::vector<Vid> curr, next;
    curr.reserve(1024);
    next.reserve(1024);

    dist[src] = 0;
    curr.push_back(src);

    while (!curr.empty()) {
      // Compute average tail length on this level
      uint64_t tail_sum = 0;
      {
        DerefScope s;
        for (Vid u : curr) {
          auto h = G.header_info(u, s);
          const uint32_t tail_len = (h.degree > h.inline_len) ? (h.degree - h.inline_len) : 0;
          tail_sum += tail_len;
        }
      }
      const double avg_tail = curr.empty() ? 0.0 : (double)tail_sum / (double)curr.size();

      // Sort only if BOTH conditions are met
      if (curr.size() >= sort_min_frontier && avg_tail >= (double)tail_threshold) {
        std::sort(curr.begin(), curr.end());
      }

      // Expand this level
      for (Vid u : curr) {
        DerefScope s;
        G.for_each_neighbor(u, s, [&](Vid v){
          if (dist[v] == -1) { dist[v] = dist[u] + 1; next.push_back(v); }
        });
      }

      curr.swap(next);
      next.clear();
    }
  };

  // warm
  run_once();
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
                    uint32_t peek_k /*0=baseline*/) {
  Row r{};
  r.policy  = policy_name;
  r.variant = (peek_k == 0) ? "baseline" : "tailpeek(2)";

  // Build graph in a tight scope
  {
    // Compute N from edges’ max ID
    uint64_t N = 0;
    for (auto &e : edges) {
      N = std::max<uint64_t>(N, std::max<uint64_t>(e.first, e.second));
    }
    N += 1;

    GraphAdj G(mgr, N, pol);
    G.build_from_edges(edges);

    // layout stats
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

    // time BFS
    r.bfs_us = (peek_k == 0)
      ? bfs_baseline_time_us(G, /*src=*/0, /*iters=*/5)
      : bfs_tailpeek_time_us(G, /*src=*/0, /*peek_k=*/peek_k, /*iters=*/5);
  }

  return r;
}
static Row run_case_tailwarm(FarMemManager* mgr,
                             const std::vector<std::pair<Vid,Vid>>& edges,
                             const char* policy_name,
                             const RemotingPolicy& pol,
                             uint32_t tail_threshold /* e.g., 16 */) {
  Row r{};
  r.policy   = policy_name;
  r.variant = tail_threshold ? "tailwarm(th=16)" : "baseline";

  {
    GraphAdj G(mgr, /*N=*/0, pol); // dummy scope – ensures no premature alloc
  }

  {
    // Build with the actual vertex count inferred from edges
    GraphAdj G(mgr, /*N=*/edges.empty()?0:(
      [&](){ Vid maxv=0; for (auto &e : edges) { if (e.first>maxv) maxv=e.first; if (e.second>maxv) maxv=e.second; }
             return (uint64_t)maxv + 1; }()),
      pol);

    G.build_from_edges(edges);

    const uint64_t N = G.num_vertices();
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

    r.bfs_us = bfs_time_us_tailwarm(G, /*src=*/0, /*iters=*/5, tail_threshold);
    // No static prefetcher to disable; no speculative reads; safe and simple.
  }
  return r;
}

// ---- Tiny-tail peek sweep (K in {0,1,2,4,8}) with speedup vs baseline ----
static Row run_case_sorted_frontier(FarMemManager* mgr,
                                    const std::vector<std::pair<Vid,Vid>>& edges,
                                    const char* policy_name,
                                    const RemotingPolicy& pol) {
  Row r{};
  r.policy  = policy_name;
  r.variant = "frontier-sort";

  // infer N
  uint64_t N = 0;
  for (auto &e : edges) N = std::max<uint64_t>(N, std::max<uint64_t>(e.first, e.second));
  N += 1;

  GraphAdj G(mgr, N, pol);
  G.build_from_edges(edges);

  // stats (same as baseline)
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

  r.bfs_us = bfs_sorted_frontier_time_us(G, /*src=*/0, /*iters=*/5);
  return r;
}



{
  const uint32_t kSortMinFrontier = 4096; // only sort when frontier is big
  const uint32_t kTailThreshold   = 64;   // and tails are heavy on average

  // All-remote (adaptive)
  {
    // rebuild the graph once for timing (same as run_case baseline pattern)
    uint64_t Nmax = 0;
    for (auto &e : edges) Nmax = std::max<uint64_t>(Nmax, std::max<uint64_t>(e.first, e.second));
    GraphAdj G(manager.get(), Nmax + 1, all_remote);
    G.build_from_edges(edges);

    double us = bfs_frontier_sort_adaptive_time_us(G, /*src=*/0,
                                                   kSortMinFrontier, kTailThreshold,
                                                   /*iters=*/5);
    // stats just for context
    uint64_t inline_any=0, remote_any=0, remote_bytes=0;
    for (uint64_t u = 0; u < G.num_vertices(); ++u) {
      DerefScope s;
      auto h = G.header_info(u, s);
      if (h.inline_len > 0) inline_any++;
      if (h.degree > h.inline_len) {
        remote_any++;
        remote_bytes += (h.degree - h.inline_len) * sizeof(Vid);
      }
    }
    cout << "All-remote,frontier-sort(adaptive f>=4096 tail>=64),"
         << inline_any << "," << remote_any << "," << remote_bytes << ","
         << us << "\n";
  }

  // Local-8 (adaptive)
  {
    uint64_t Nmax = 0;
    for (auto &e : edges) Nmax = std::max<uint64_t>(Nmax, std::max<uint64_t>(e.first, e.second));
    GraphAdj G(manager.get(), Nmax + 1, local_8);
    G.build_from_edges(edges);

    double us = bfs_frontier_sort_adaptive_time_us(G, /*src=*/0,
                                                   kSortMinFrontier, kTailThreshold,
                                                   /*iters=*/5);
    uint64_t inline_any=0, remote_any=0, remote_bytes=0;
    for (uint64_t u = 0; u < G.num_vertices(); ++u) {
      DerefScope s;
      auto h = G.header_info(u, s);
      if (h.inline_len > 0) inline_any++;
      if (h.degree > h.inline_len) {
        remote_any++;
        remote_bytes += (h.degree - h.inline_len) * sizeof(Vid);
      }
    }
    cout << "Local-8,frontier-sort(adaptive f>=4096 tail>=64),"
         << inline_any << "," << remote_any << "," << remote_bytes << ","
         << us << "\n";
  }
}




int main(int argc, char* argv[]) {
  if (argc < 2) { std::cerr << "usage: " << argv[0] << " [cfg_file]\n"; return -EINVAL; }
  int ret = runtime_init(argv[1], _main, nullptr);
  if (ret) { std::cerr << "failed to start runtime\n"; return ret; }
  return 0;
}
