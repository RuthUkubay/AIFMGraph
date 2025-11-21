// aifm/test/test_graph_basic.cpp
extern "C" {
#include <runtime/runtime.h>
}

#include "deref_scope.hpp"
#include "device.hpp"
#include "manager.hpp"

// our graph headers (header-only under aifm/inc/)
#include "graph_local.hpp"
#include "graph_far.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

using namespace std;
using namespace far_memory;

/* ---------------- tiny local graph for sanity ---------------- */

static void build_toy_local(CSRLocal& G){
  G.add_edge(0,1);
  G.add_edge(0,2);
  G.add_edge(1,3);
  G.add_edge(2,3);
  G.add_edge(3,4);
  G.add_edge(4,5);
  G.finalize();
}

template <typename CSRFarT>
static void build_toy_far(CSRFarT& G){
  G.add_edge(0,1);
  G.add_edge(0,2);
  G.add_edge(1,3);
  G.add_edge(2,3);
  G.add_edge(3,4);
  G.add_edge(4,5);
}

/* ---------------- scalable so we can see differences ---------------- */

static void build_ring_plus_random_local(CSRLocal& G, int32_t n, int deg = 4) {

  for (int32_t u = 0; u < n; ++u) G.add_edge(u, (u + 1) % n);
  // extra random edges per node
  std::mt19937 rng(123);
  std::uniform_int_distribution<int32_t> dist(0, n - 1);
  for (int32_t u = 0; u < n; ++u) {
    for (int k = 1; k < deg; ++k) G.add_edge(u, dist(rng));
  }
  G.finalize();
}

template <typename CSRFarT>
static void build_ring_plus_random_far(CSRFarT& G, int32_t n, int deg = 4) {
  for (int32_t u = 0; u < n; ++u) G.add_edge(u, (u + 1) % n);
  std::mt19937 rng(123);
  std::uniform_int_distribution<int32_t> dist(0, n - 1);
  for (int32_t u = 0; u < n; ++u) {
    for (int k = 1; k < deg; ++k) G.add_edge(u, dist(rng));
  }
}

/* ---------------- runtime entry ---------------- */

// Use the same style as other AIFM tests
constexpr uint64_t kCacheSize    = 256 * Region::kSize; // local cache for far mem
constexpr uint64_t kFarMemSize   = (1ULL << 33);        // 8 GB FakeDevice
constexpr uint64_t kNumGCThreads = 12;                  // GC threads

// Scale knobs for the large benchmark
static constexpr int32_t  kNVerts   = 1 << 18;                 // 262,144 nodes
static constexpr uint64_t kMaxEdges = (uint64_t)kNVerts * 4;   // ~4 edges/node

static void _main(void *arg) {
  // Build FarMemManager exactly like other AIFM tests (FakeDevice == no TCP server needed)
  std::unique_ptr<FarMemManager> manager(
      FarMemManagerFactory::build(kCacheSize, kNumGCThreads, new FakeDevice(kFarMemSize)));

  /* ---- 0) Tiny sanity: local and far should both say dist[5]=4 ---- */
  {
    // local
    CSRLocal L6(6);
    build_toy_local(L6);
    auto dloc = bfs_local(L6, 0, nullptr);

    // far
    auto off6 = manager->allocate_array<int32_t, 6>();
    auto deg6 = manager->allocate_array<int32_t, 6>();
    auto nbr6 = manager->allocate_array<int32_t, 16>();
    fargraph::CSRFar<6, 16> F6(off6, deg6, nbr6);
    build_toy_far(F6);
    F6.finalize();
    auto dfar = fargraph::bfs_far(F6, 0, nullptr);

    std::cout << "sanity: local dist[5]=" << dloc[5]
              << " | far dist[5]=" << dfar[5] << "\n";
  }

  /* ---- 1) Local benchmark (microseconds, many iterations) ---- */
  {
    CSRLocal L(kNVerts);
    build_ring_plus_random_local(L, kNVerts, /*deg=*/4);

    // warm-up (populate caches, etc.)
    auto warm = bfs_local(L, 0, nullptr); (void)warm;

    const int iters = 50;
    auto t0 = chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
      auto dist = bfs_local(L, 0, nullptr); (void)dist;
    }
    auto t1 = chrono::high_resolution_clock::now();
    auto us = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();

    std::cout << "local: iters=" << iters
              << " total_us=" << us
              << " per_iter_us=" << (double)us / iters
              << "\n";
  }

  /* ---- 2) Far benchmark (AIFM Arrays + one DerefScope per thread) ---- */
  {
    // allocate far arrays with compile-time sizes
    auto off = manager->allocate_array<int32_t, kNVerts>();
    auto deg = manager->allocate_array<int32_t, kNVerts>();
    auto nbr = manager->allocate_array<int32_t, kMaxEdges>();

    // OPTIONAL: try disabling prefetch to see impact (leave enabled first)
    // off.disable_prefetch(); deg.disable_prefetch(); nbr.disable_prefetch();

    // wrap arrays by reference in our CSR graph
    fargraph::CSRFar<kNVerts, kMaxEdges> F(off, deg, nbr);

    // build & finalize (copies once into far arrays with a single DerefScope)
    build_ring_plus_random_far(F, kNVerts, /*deg=*/4);
    F.finalize();

    // warm-up
    auto warm = fargraph::bfs_far(F, 0, nullptr); (void)warm;

    const int iters = 50;
    auto t0 = chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
      auto dist = fargraph::bfs_far(F, 0, nullptr); (void)dist;
    }
    auto t1 = chrono::high_resolution_clock::now();
    auto us = chrono::duration_cast<chrono::microseconds>(t1 - t0).count();

    std::cout << "far:   iters=" << iters
              << " total_us=" << us
              << " per_iter_us=" << (double)us / iters
              << "\n";
  }
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " [cfg_file]\n";
    return -EINVAL;
  }
  int ret = runtime_init(argv[1], _main, NULL);
  if (ret) {
    std::cerr << "failed to start runtime\n";
    return ret;
  }
  return 0;
}
// ---------- end of file ----------