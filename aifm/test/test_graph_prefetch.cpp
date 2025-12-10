// test/test_graph_prefetch.cpp
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

/*** Placement policies ***/
struct AllRemote : RemotingPolicy {
  uint16_t inline_capacity(Vid, uint32_t) const override { return 0; }
};
struct Local8 : RemotingPolicy {
  uint16_t inline_capacity(Vid, uint32_t deg) const override {
    return deg <= 8 ? deg : 0;
  }
};

/*** Banded generator: neighbors of u mostly in [u, u+W) ***/
static std::vector<std::pair<Vid,Vid>>
gen_banded_edges(uint64_t N, uint64_t E, uint32_t window, uint64_t seed=42) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<uint64_t> Uu(0, N - 1);
  std::uniform_int_distribution<uint32_t> Uw(0, window - 1);

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

/*** BFS variants using the safe iterator ***/

// Baseline BFS: plain queue (no prefetch)
static std::vector<int> bfs_baseline(GraphAdj &G, Vid src) {
  const uint64_t n = G.num_vertices();
  std::vector<int> dist(n, -1);
  if (src >= n) return dist;

  std::queue<Vid> q;
  dist[src] = 0; q.push(src);

  while (!q.empty()) {
    Vid u = q.front(); q.pop();
    DerefScope s;
    G.for_each_neighbor(u, s, [&](Vid v){
      if (dist[v] == -1) { dist[v] = dist[u] + 1; q.push(v); }
    });
  }
  return dist;
}

// Header-prefetch BFS: process vertices in ascending ID per “wave”
static std::vector<int> bfs_header_prefetch(GraphAdj &G, Vid src, uint32_t pf_dist) {
  G.enable_header_static_prefetch(pf_dist);

  const uint64_t n = G.num_vertices();
  std::vector<int> dist(n, -1);
  if (src >= n) return dist;

  std::vector<Vid> curr, next;
  curr.reserve(4096);
  next.reserve(4096);
  dist[src] = 0; curr.push_back(src);

  while (!curr.empty()) {
    std::sort(curr.begin(), curr.end()); // header stride becomes 1
    for (Vid u : curr) {
      DerefScope s;
      G.for_each_neighbor(u, s, [&](Vid v){
        if (dist[v] == -1) { dist[v] = dist[u] + 1; next.push_back(v); }
      });
    }
    curr.swap(next); next.clear();
  }
  return dist;
}

/*** Timing harness ***/
struct RunCfg {
  const char* name;
  const RemotingPolicy& pol;
  uint32_t pf_distance; // 0 = baseline
};

static double time_bfs(GraphAdj& G, Vid src, uint32_t iters, uint32_t pf_distance) {
  using clk = std::chrono::high_resolution_clock;

  // Warm once
  (void)(pf_distance == 0 ? bfs_baseline(G, src)
                          : bfs_header_prefetch(G, src, pf_distance));

  auto t0 = clk::now();
  for (uint32_t i = 0; i < iters; ++i) {
    if (pf_distance == 0) {
      (void)bfs_baseline(G, src);
    } else {
      (void)bfs_header_prefetch(G, src, pf_distance);
    }
  }
  auto t1 = clk::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
         / static_cast<double>(iters);
}

static void do_work(FarMemManager* mgr) {
  // Larger E + small window used to crash before → now safe via chunking.
  const uint64_t N = 200000;
  const uint64_t E = 3200000; // 3.2M edges
  const uint32_t W = 16;      // strong locality
  const uint32_t iters = 5;

  auto edges = gen_banded_edges(N, E, W);

  AllRemote all_remote;
  Local8    local_8;

  std::vector<RunCfg> runs = {
    {"All-remote / no-prefetch", all_remote, 0},
    {"All-remote / header-pf(d=64)", all_remote, 64},
    {"Local-8 / no-prefetch", local_8, 0},
    {"Local-8 / header-pf(d=64)", local_8, 64},
  };

  cout << "Graph: |V|=" << N << " |E|=" << E
       << " (banded window=" << W << ")\n";

  // Header row
  cout << "Policy,Vertices w/ local neighbors,Vertices w/ remote,Remote bytes,"
          "BFS per-iter (µs),Speedup vs All-remote(no-pf)\n";

  double baseline_us = 0.0;

  for (size_t i = 0; i < runs.size(); ++i) {
    const auto& cfg = runs[i];
    GraphAdj G(mgr, N, cfg.pol);
    G.build_from_edges(edges);

    // layout stats
    uint64_t remote_bytes = 0, inline_any = 0, remote_any = 0;
    for (uint64_t u = 0; u < N; ++u) {
      DerefScope s;
      auto h = G.header_info(u, s);
      if (h.inline_len > 0) inline_any++;
      if (h.degree > h.inline_len) {
        // Sum all tail chunk sizes
        const auto &vh = *reinterpret_cast<const VertexHdr*>(
            G.header_info(u, s), // not accessible here — compute again:
            nullptr);
        // Simpler: compute by algebra (degree - inline_len)*4:
        remote_any++;
        remote_bytes += (h.degree - h.inline_len) * sizeof(Vid);
      }
    }

    double us = time_bfs(G, /*src=*/0, iters, cfg.pf_distance);
    if (i == 0) baseline_us = us;

    double speedup = (baseline_us - us) / baseline_us * 100.0;

    cout << cfg.name << ","
         << inline_any << ","
         << remote_any << ","
         << remote_bytes << ","
         << us << ",";
    if (i == 0) cout << "—";
    else        cout << (speedup >= 0 ? "+" : "") << speedup << "%";
    cout << "\n";

    // Clean up any static prefetch hint before G is destroyed
    G.disable_header_prefetch();
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
