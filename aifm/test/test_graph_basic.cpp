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

/* ---------------- scalable synthetic graph builders ---------------- */

static void build_ring_plus_random_local(CSRLocal& G, int32_t n, int deg = 4) {
  // ring backbone
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
constexpr uint64_t kNumGCThreads = 12;
