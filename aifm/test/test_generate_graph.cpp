// aifm/test/test_generate_graph.cpp
extern "C" {
#include <runtime/runtime.h>
}

#include "graph_adj.hpp"
#include "device.hpp"   // FakeDevice
#include "manager.hpp"
#include "deref_scope.hpp"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <queue>
#include <random>
#include <utility>
#include <vector>

using namespace far_memory;
using std::cout;
using std::endl;

constexpr uint64_t kCacheSize    = (128ULL << 20); // 128 MB
constexpr uint64_t kFarMemSize   = (4ULL  << 30);  // 4 GB
constexpr uint32_t kNumGCThreads = 12;

struct GenParams {
  uint64_t num_vertices;
  uint64_t num_edges;
  uint64_t seed;
};

static std::vector<std::pair<Vid,Vid>> gen_random_edges(const GenParams& p) {
  std::mt19937_64 rng(p.seed);
  std::uniform_int_distribution<uint64_t> U(0, p.num_vertices - 1);

  std::vector<std::pair<Vid,Vid>> edges;
  edges.reserve(p.num_edges);
  for (uint64_t i = 0; i < p.num_edges; ++i) {
    Vid u = static_cast<Vid>(U(rng));
    Vid v = static_cast<Vid>(U(rng));
    if (u == v) v = (u + 1 < p.num_vertices) ? u + 1 : 0; // avoid self-loop
    edges.emplace_back(u, v);
  }
  return edges;
}

static void sanity_check(GraphAdj& G, const std::vector<std::pair<Vid,Vid>>& edges) {
  const uint64_t N = G.num_vertices();

  // total degree check
  uint64_t sum_deg = 0;
  for (uint64_t u = 0; u < N; ++u) {
    DerefScope s;
    sum_deg += G.degree(u, s);
  }
  if (sum_deg != edges.size()) {
    cout << "Sanity failed: sum_deg=" << sum_deg
         << " edges=" << edges.size() << endl;
  }

  // touch a few neighbor entries (inline and tail) for first few vertices
  for (uint64_t u = 0; u < std::min<uint64_t>(N, 5); ++u) {
    DerefScope s;
    auto view = G.neighbors(u, s);

    for (uint32_t i = 0; i < std::min<uint32_t>(view.inline_len, 3u); ++i) {
      volatile Vid v = view.inline_ptr[i]; (void)v;
    }
    for (uint32_t i = 0; i < std::min<uint32_t>(view.tail_len, 3u); ++i) {
      volatile Vid v = view.tail_ptr[i]; (void)v;
    }
  }
}

// Minimal BFS that understands inline + tail layout
static std::vector<int> bfs(GraphAdj& G, Vid src) {
  const uint64_t n = G.num_vertices();
  std::vector<int> dist(n, -1);
  if (src >= n) return dist;

  std::queue<Vid> q;
  dist[src] = 0; q.push(src);

  while (!q.empty()) {
    Vid u = q.front(); q.pop();
    DerefScope scope;
    auto view = G.neighbors(u, scope);

    for (uint32_t i = 0; i < view.inline_len; ++i) {
      Vid v = view.inline_ptr[i];
      if (dist[v] == -1) { dist[v] = dist[u] + 1; q.push(v); }
    }
    for (uint32_t i = 0; i < view.tail_len; ++i) {
      Vid v = view.tail_ptr[i];
      if (dist[v] == -1) { dist[v] = dist[u] + 1; q.push(v); }
    }
  }
  return dist;
}

// Simple “all-remote” policy for this test
struct AllRemotePolicy : RemotingPolicy {
  uint16_t inline_capacity(Vid, uint32_t) const override { return 0; }
};

static void do_work(FarMemManager* manager, const GenParams& gen) {
  cout << "Running " << __FILE__ << "..." << endl;

  auto edges = gen_random_edges(gen);

  // NOTE: GraphAdj now needs a policy
  AllRemotePolicy pol;
  GraphAdj G(manager, gen.num_vertices, pol);
  G.build_from_edges(edges);

  sanity_check(G, edges);

  // quick BFS timing (optional)
  using clk = std::chrono::high_resolution_clock;
  auto warm = bfs(G, 0); (void)warm;

  auto t0 = clk::now();
  auto dist = bfs(G, 0);
  auto t1 = clk::now();
  auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
  cout << "BFS one run: " << us << " us" << endl;

  cout << "Graph built: |V|=" << gen.num_vertices
       << " |E|=" << edges.size() << endl;
  cout << "Passed" << endl;
}

static void _main(void* /*arg*/) {
  std::unique_ptr<FarMemManager> manager(
      FarMemManagerFactory::build(kCacheSize, kNumGCThreads, new FakeDevice(kFarMemSize)));

  GenParams gen {
    .num_vertices = 2000,
    .num_edges    = 100000,
    .seed         = 42
  };

  do_work(manager.get(), gen);
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "usage: [cfg_file]" << std::endl;
    return -EINVAL;
  }
  int ret = runtime_init(argv[1], _main, nullptr);
  if (ret) {
    std::cerr << "failed to start runtime" << std::endl;
    return ret;
  }
  return 0;
}
