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
static inline void bin_frontier(std::vector<Vid>& curr,
                                std::vector<Vid>& tmp,
                                uint32_t block_bits /* e.g., 10 for 1024 */) {
  const uint32_t B = 1u << block_bits;
  if (curr.empty()) return;

  // Find bin range
  Vid minv = curr[0], maxv = curr[0];
  for (Vid v : curr) { if (v < minv) minv = v; if (v > maxv) maxv = v; }
  uint32_t first_bin = minv >> block_bits;
  uint32_t last_bin  = maxv >> block_bits;
  uint32_t nbins = last_bin - first_bin + 1;

  // Count
  std::vector<uint32_t> cnt(nbins, 0);
  for (Vid v : curr) cnt[(v >> block_bits) - first_bin]++;

  // Prefix
  std::vector<uint32_t> off(nbins, 0);
  for (uint32_t i = 1; i < nbins; ++i) off[i] = off[i-1] + cnt[i-1];

  // Scatter into tmp by bin order
  tmp.resize(curr.size());
  std::vector<uint32_t> cur = off; // copy
  for (Vid v : curr) {
    uint32_t b = (v >> block_bits) - first_bin;
    tmp[cur[b]++] = v;
  }
  curr.swap(tmp);
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
      if (curr.size() >= sort_min_frontier && avg_tail >= tail_threshold) {
        // New: only bin if the ID span is wide enough (≥ 8 bins)
        Vid minv = curr[0], maxv = curr[0];
        for (Vid v : curr) { if (v < minv) minv = v; if (v > maxv) maxv = v; }
        const uint32_t block_bits = 10;  // 1024-ID bin
        const uint32_t first_bin  = minv >> block_bits;
        const uint32_t last_bin   = maxv >> block_bits;
        const uint32_t nbins      = last_bin - first_bin + 1;

        if (nbins >= 8) { // span gate
            static thread_local std::vector<Vid> tmp_bin;
            bin_frontier(curr, tmp_bin, block_bits);
            // --- NEW: tiny, bounded header run-ahead prefetch per bin ---
            const uint32_t pf_batch = 256;   // small & safe
            uint32_t batches_done = 0;       // cap total work per level
            Vid last_bin_id = ~Vid(0);

            for (size_t i = 0; i < curr.size() && batches_done < 16; ) {
            const Vid v        = curr[i];
            const Vid bin_id   = v >> block_bits;

            // find the contiguous run of this bin
            size_t j = i + 1;
            while (j < curr.size() && (curr[j] >> block_bits) == bin_id) ++j;

            // one prefetch at the start of each bin run
            if (bin_id != last_bin_id) {
                const uint64_t start = static_cast<uint64_t>(curr[i]);
                G.prefetch_headers_span(start, pf_batch);
                last_bin_id = bin_id;
                ++batches_done;
            }

            i = j;
            }
    // --- end NEW ---

        }
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
static void _main(void*) {
  std::unique_ptr<FarMemManager> manager(
      FarMemManagerFactory::build(kCacheSize, kNumGCThreads, new FakeDevice(kFarMemSize)));

  // Workload
  const uint64_t N = 200000;
  const uint64_t E = 1200000;
  const uint32_t W = 64;   // mild spatial locality
  auto edges = gen_banded_edges(N, E, W);

  AllRemote all_remote;
  Local8    local_8;

  cout << "Graph: |V|=" << N << " |E|=" << E << " (banded window=" << W << ")\n";
  cout << "Policy,Variant,Vertices w/ local neighbors,Vertices w/ remote,Remote bytes,BFS per-iter (µs)\n";

  // ---- Baselines (no sorting) ----
  {
    Row base = run_case(manager.get(), edges, "All-remote", all_remote, /*peek_k=*/0);
    cout << base.policy << "," << base.variant << "," << base.inline_any << ","
         << base.remote_any << "," << base.remote_bytes << "," << base.bfs_us << "\n";
  }
  {
    Row base = run_case(manager.get(), edges, "Local-8", local_8, /*peek_k=*/0);
    cout << base.policy << "," << base.variant << "," << base.inline_any << ","
         << base.remote_any << "," << base.remote_bytes << "," << base.bfs_us << "\n";
  }

  // ---- Adaptive frontier sort (only when frontier large & tails heavy) ----
    const uint32_t kSortMinFrontier = 20000; // was 4096
    const uint32_t kTailThreshold   = 128;   // was 64
    // ADD THIS ONE-LINER so we can verify the run-time knobs:
   std::cout << "frontier-bin(adaptive f>=" << kSortMinFrontier
            << " tail>=" << kTailThreshold << ")\n";

  // All-remote (adaptive)
  {
    // infer N from edges
    uint64_t Nmax = 0;
    for (auto &e : edges)
      Nmax = std::max<uint64_t>(Nmax, std::max<uint64_t>(e.first, e.second));
    GraphAdj G(manager.get(), Nmax + 1, all_remote);
    G.build_from_edges(edges);

    // stats (for table)
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

    double us = bfs_frontier_sort_adaptive_time_us(G, /*src=*/0,
                                                   kSortMinFrontier, kTailThreshold,
                                                   /*iters=*/5);

    cout << "All-remote,frontier-bin(adaptive f>=" << kSortMinFrontier
     << " tail>=" << kTailThreshold << "),"
     << inline_any << "," << remote_any << "," << remote_bytes << ","
     << us << "\n";
  }

  // Local-8 (adaptive)
  {
    uint64_t Nmax = 0;
    for (auto &e : edges)
      Nmax = std::max<uint64_t>(Nmax, std::max<uint64_t>(e.first, e.second));
    GraphAdj G(manager.get(), Nmax + 1, local_8);
    G.build_from_edges(edges);

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

    double us = bfs_frontier_sort_adaptive_time_us(G, /*src=*/0,
                                                   kSortMinFrontier, kTailThreshold,
                                                   /*iters=*/5);

    cout << "Local-8,frontier-bin(adaptive f>=" << kSortMinFrontier
     << " tail>=" << kTailThreshold << "),"
     << inline_any << "," << remote_any << "," << remote_bytes << ","
     << us << "\n";
  }

  cout << "Done.\n";
}

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