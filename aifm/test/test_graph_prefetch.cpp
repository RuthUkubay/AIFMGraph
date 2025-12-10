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

/* Random edges (stable and light). */
static std::vector<std::pair<Vid,Vid>>
gen_random_edges(uint64_t N, uint64_t E, uint64_t seed=42) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<uint64_t> U(0, N - 1);
  std::vector<std::pair<Vid,Vid>> edges;
  edges.reserve(E);
  for (uint64_t i = 0; i < E; ++i) {
    Vid u = static_cast<Vid>(U(rng));
    Vid v = static_cast<Vid>(U(rng));
    if (u == v) v = (u + 1 < N) ? u + 1 : 0;
    edges.emplace_back(u, v);
  }
  return edges;
}

/* ----- BFS variants ----- */

// 1) Baseline BFS (no warm-up)
static double bfs_baseline_time(GraphAdj &G, Vid src, uint32_t iters) {
  using clk = std::chrono::high_resolution_clock;
  const uint64_t n = G.num_vertices();
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

// 2) Look-ahead header warm: before expanding each u, touch the headers
// for the next L vertices in this BFS level (if any). This is synchronous,
// safe, and requires no allocator hints.
static double bfs_header_lookahead_time(GraphAdj &G, Vid src,
                                        uint32_t lookahead_L,
                                        uint32_t iters) {
  using clk = std::chrono::high_resolution_clock;
  const uint64_t n = G.num_vertices();

  auto run_once = [&](){
    std::vector<int> dist(n, -1);
    std::vector<Vid> curr, next;
    curr.reserve(1024); next.reserve(1024);

    dist[src] = 0; curr.push_back(src);

    while (!curr.empty()) {
      // Optional: keep headers somewhat sequential to help cache locality.
      std::sort(curr.begin(), curr.end());

      for (size_t idx = 0; idx < curr.size(); ++idx) {
        Vid u = curr[idx];

        // --- tiny, safe "prefetch": header touches for the next L IDs in this level
        if (lookahead_L) {
          DerefScope sh;
          size_t end = std::min(curr.size(), idx + 1 + static_cast<size_t>(lookahead_L));
          for (size_t j = idx + 1; j < end; ++j) {
            (void)G.header_info(curr[j], sh); // read-only header map
          }
        }

        // expand neighbors
        DerefScope se;
        G.for_each_neighbor(u, se, [&](Vid v){
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

/* ----- One run row ----- */
struct Row {
  const char* policy;
  const char* variant;     // "baseline" or "hdr_lookahead(L)"
  uint64_t inline_any;
  uint64_t remote_any;
  uint64_t remote_bytes;
  double   bfs_us;
};

static Row build_and_run(FarMemManager* mgr,
                         const std::vector<std::pair<Vid,Vid>>& edges,
                         const char* policy_name,
                         const RemotingPolicy& pol,
                         uint32_t lookaheadL) {
  Row r{};
  r.policy  = policy_name;
  r.variant = (lookaheadL == 0) ? "baseline" : "hdr_lookahead";

  // Build once per row
  const uint64_t N = [&](){
    Vid maxv=0; for (auto &e : edges) { if (e.first > maxv) maxv=e.first;
                                        if (e.second> maxv) maxv=e.second; }
    return static_cast<uint64_t>(maxv) + 1;
  }();

  GraphAdj G(mgr, N, pol);
  G.build_from_edges(edges);

  // stats
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

  // time
  r.bfs_us = (lookaheadL == 0)
               ? bfs_baseline_time(G, /*src=*/0, /*iters=*/5)
               : bfs_header_lookahead_time(G, /*src=*/0, lookaheadL, /*iters=*/5);

  return r;
}

/* ----- Driver ----- */
static void _main(void*) {
  std::unique_ptr<FarMemManager> manager(
      FarMemManagerFactory::build(kCacheSize, kNumGCThreads, new FakeDevice(kFarMemSize)));

  // Back to stable scale that never crashed for you.
  const uint64_t N = 200000;
  const uint64_t E = 1200000;
  auto edges = gen_random_edges(N, E);

  AllRemote all_remote;
  Local8    local_8;

  cout << "Graph: |V|=" << N << " |E|=" << E << "\n";
  cout << "Policy,Variant,Vertices w/ local neighbors,Vertices w/ remote,Remote bytes,BFS per-iter (µs)\n";

  // All-remote: baseline vs small look-ahead (L=32)
  {
    Row a = build_and_run(manager.get(), edges, "All-remote", all_remote, 0);
    cout << a.policy << "," << a.variant << "," << a.inline_any << "," << a.remote_any
         << "," << a.remote_bytes << "," << a.bfs_us << "\n";

    Row b = build_and_run(manager.get(), edges, "All-remote", all_remote, 32);
    cout << b.policy << "," << b.variant << "," << b.inline_any << "," << b.remote_any
         << "," << b.remote_bytes << "," << b.bfs_us << "\n";
  }

  // Local-8: baseline vs small look-ahead (L=32)
  {
    Row a = build_and_run(manager.get(), edges, "Local-8", local_8, 0);
    cout << a.policy << "," << a.variant << "," << a.inline_any << "," << a.remote_any
         << "," << a.remote_bytes << "," << a.bfs_us << "\n";

    Row b = build_and_run(manager.get(), edges, "Local-8", local_8, 32);
    cout << b.policy << "," << b.variant << "," << b.inline_any << "," << b.remote_any
         << "," << b.remote_bytes << "," << b.bfs_us << "\n";
  }

  cout << "Done.\n";
}

int main(int argc, char* argv[]) {
  if (argc < 2) { std::cerr << "usage: " << argv[0] << " [cfg_file]\n"; return -EINVAL; }
  int ret = runtime_init(argv[1], _main, nullptr);
  if (ret) { std::cerr << "failed to start runtime\n"; return ret; }
  return 0;
}
