// aifm/test/test_graph_compute_tcp.cpp
extern "C" {
#include <runtime/runtime.h>
}

#include "device.hpp"
#include "helpers.hpp"
#include "manager.hpp"
#include "graph_adj.hpp"
#include "deref_scope.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace far_memory;
using std::cout;
using std::endl;

constexpr static uint64_t kCacheSize      = (128ULL << 20);
constexpr static uint64_t kFarMemSize     = (4ULL  << 30);
constexpr static uint32_t kNumGCThreads   = 12;
constexpr static uint32_t kNumConnections = 300;

// ---------------------------------------------------------------------
// Banded-edge generator (same as graph prefetch test)
// ---------------------------------------------------------------------
static std::vector<std::pair<Vid, Vid>>
gen_banded_edges(uint64_t N, uint64_t E, uint32_t W, uint64_t seed = 42) {
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<uint64_t> Uu(0, N - 1);
  std::uniform_int_distribution<uint32_t> Uw(0, W - 1);

  std::vector<std::pair<Vid, Vid>> edges;
  edges.reserve(E);
  for (uint64_t i = 0; i < E; ++i) {
    Vid u = static_cast<Vid>(Uu(rng));
    Vid v = static_cast<Vid>((u + Uw(rng)) % N);
    if (v == u) v = (u + 1) % N;
    edges.emplace_back(u, v);
  }
  return edges;
}

// ---------------------------------------------------------------------
// A tiny "local header" struct: just what the client cares about.
// We fill this once by dereferencing far-memory headers, then reuse it.
// ---------------------------------------------------------------------
struct LocalHdr {
  uint32_t degree;
  uint16_t inline_len;
};

// Build local header cache by doing far-mem derefs *once*.
static std::vector<LocalHdr>
build_header_cache(GraphAdj &G) {
  const uint64_t n = G.num_vertices();
  std::vector<LocalHdr> cache(n);

  DerefScope s;
  for (uint64_t u = 0; u < n; ++u) {
    auto h = G.header_info(u, s);
    cache[u].degree     = h.degree;
    cache[u].inline_len = h.inline_len;
  }
  return cache;
}

// ---------------------------------------------------------------------
// Baseline 1: naive local client that derefs far-mem headers every time
// ---------------------------------------------------------------------
static uint64_t
local_frontier_sum_with_deref(GraphAdj &G,
                              const std::vector<Vid> &frontier) {
  uint64_t total = 0;
  DerefScope s;
  for (Vid u : frontier) {
    auto h = G.header_info(u, s); // far-mem header deref
    (void)h;
    total += static_cast<uint64_t>(u);
  }
  return total;
}

// ---------------------------------------------------------------------
// Baseline 2: "headers local" client
// Graph still lives in far memory, but we assume all headers were
// fetched/cached once. Now we read them from a local array.
// ---------------------------------------------------------------------
static uint64_t
local_frontier_sum_cached(const std::vector<Vid> &frontier,
                          const std::vector<LocalHdr> &hdr_cache) {
  uint64_t total = 0;
  for (Vid u : frontier) {
    const LocalHdr &h = hdr_cache[u];
    // Touch h so the compiler can't drop it
    total += static_cast<uint64_t>(u) + static_cast<uint64_t>(h.degree & 1u);
  }
  return total;
}

// ---------------------------------------------------------------------
// Remote aggregation via active component
// This calls GraphAdj::remote_degree_sum(frontier).
// ---------------------------------------------------------------------
static uint64_t
remote_frontier_sum(GraphAdj &G,
                    const std::vector<Vid> &frontier) {
  return G.remote_degree_sum(frontier);
}

// ---------------------------------------------------------------------
// Benchmark: naive local vs cached-header local vs remote
// ---------------------------------------------------------------------
static void benchmark_frontier_agg(GraphAdj &G,
                                   const std::vector<LocalHdr> &hdr_cache,
                                   const std::vector<Vid> &frontier,
                                   uint32_t iters) {
  using clk = std::chrono::high_resolution_clock;
  using us  = std::chrono::microseconds;

  // Sanity: All three paths should agree.
  uint64_t naive_once   = local_frontier_sum_with_deref(G, frontier);
  uint64_t cached_once  = local_frontier_sum_cached(frontier, hdr_cache);
  uint64_t remote_once  = remote_frontier_sum(G, frontier);

  cout << "test_graph_compute: naive_sum  = " << naive_once  << "\n";
  cout << "test_graph_compute: cached_sum = " << cached_once << "\n";
  cout << "test_graph_compute: remote_sum = " << remote_once << "\n";

  if (naive_once != cached_once || naive_once != remote_once) {
    cout << "test_graph_compute: MISMATCH between variants!" << endl;
    return;
  }
  cout << "test_graph_compute: PASS (all variants agree)\n";

  // --- Benchmark naive local (far-mem header each time) ---
  auto t0 = clk::now();
  for (uint32_t i = 0; i < iters; ++i) {
    (void)local_frontier_sum_with_deref(G, frontier);
  }
  auto t1 = clk::now();
  double naive_us =
      std::chrono::duration_cast<us>(t1 - t0).count() / double(iters);

  // --- Benchmark cached-header local (headers local) ---
  auto t2 = clk::now();
  for (uint32_t i = 0; i < iters; ++i) {
    (void)local_frontier_sum_cached(frontier, hdr_cache);
  }
  auto t3 = clk::now();
  double cached_us =
      std::chrono::duration_cast<us>(t3 - t2).count() / double(iters);

  // --- Benchmark remote active compute ---
  auto t4 = clk::now();
  for (uint32_t i = 0; i < iters; ++i) {
    (void)remote_frontier_sum(G, frontier);
  }
  auto t5 = clk::now();
  double remote_us =
      std::chrono::duration_cast<us>(t5 - t4).count() / double(iters);

  auto pct = [](double base, double other) {
    return (base > 0.0) ? (1.0 - other / base) * 100.0 : 0.0;
  };

  cout << "Frontier size: " << frontier.size() << "\n";
  cout << "Naive   local (far-mem header per u): " << naive_us  << " us / call\n";
  cout << "Cached  local (headers local)       : " << cached_us << " us / call\n";
  cout << "Remote  agg  (active component)     : " << remote_us << " us / call\n";
  cout << "Remote vs naive   : "
       << (pct(naive_us,  remote_us) >= 0 ? "+" : "")
       << pct(naive_us,  remote_us) << "%\n";
  cout << "Remote vs cached  : "
       << (pct(cached_us, remote_us) >= 0 ? "+" : "")
       << pct(cached_us, remote_us) << "%\n";
}

int argc;

// ---------------------------------------------------------------------
// Main AIFM runtime entry
// ---------------------------------------------------------------------
static void _main(void *arg) {
  char **argv = static_cast<char **>(arg);
  std::string ip_addr_port(argv[1]);
  auto raddr = helpers::str_to_netaddr(ip_addr_port);

  std::unique_ptr<FarMemManager> manager(
      FarMemManagerFactory::build(
          kCacheSize, kNumGCThreads,
          new TCPDevice(raddr, kNumConnections, kFarMemSize)));

  // Build a medium-size graph in far memory.
  const uint64_t N = 200000;
  const uint64_t E = 1200000;
  const uint32_t W = 64;
  auto edges = gen_banded_edges(N, E, W);

  struct AllRemote : RemotingPolicy {
    uint16_t inline_capacity(Vid, uint32_t) const override { return 0; }
  } all_remote;

  // GraphAdj lives entirely in far memory (TCP-backed).
  auto G = std::make_unique<GraphAdj>(manager.get(), N, all_remote);
  G->build_from_edges(edges);

  // Build header cache once: headers are "local" for the cached baseline.
  auto hdr_cache = build_header_cache(*G);

  // Pick a random frontier.
  const size_t frontier_size = 10000;
  std::vector<Vid> frontier;
  frontier.reserve(frontier_size);

  std::mt19937_64 rng(12345);
  std::uniform_int_distribution<uint64_t> U(0, N - 1);
  for (size_t i = 0; i < frontier_size; ++i) {
    frontier.push_back(static_cast<Vid>(U(rng)));
  }

  // Run sanity + microbenchmark.
  const uint32_t iters = 200;
  benchmark_frontier_agg(*G, hdr_cache, frontier, iters);

  cout << "Done.\n";
}

// ---------------------------------------------------------------------
// Process command line and hand off to AIFM runtime
// ---------------------------------------------------------------------
int main(int _argc, char* argv[]) {
  if (_argc < 3) {
    std::cerr << "usage: " << argv[0] << " [cfg_file] [ip_addr:port]\n";
    return -EINVAL;
  }

  char conf_path[strlen(argv[1]) + 1];
  strcpy(conf_path, argv[1]);

  // Shift argv so that _main sees [ip_addr:port, ...] starting at argv[1]
  for (int i = 2; i < _argc; i++) {
    argv[i - 1] = argv[i];
  }
  argc = _argc - 1;

  int ret = runtime_init(conf_path, _main, argv);
  if (ret) {
    std::cerr << "failed to start runtime" << std::endl;
    return ret;
  }
  return 0;
}
// ---------- end of file ----------