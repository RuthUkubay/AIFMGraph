// aifm/test/test_graph_basic.cpp
extern "C" {
#include <runtime/runtime.h>
}

#include "deref_scope.hpp"
#include "device.hpp"
#include "manager.hpp"

// our graph headers (header-only under inc/)
#include "graph_local.hpp"
#include "graph_far.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
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

/* ---------------- far graph (works with templated CSRFar) ---------------- */

template <typename CSRFarT>
static void build_toy_far(CSRFarT& G){
  G.add_edge(0,1);
  G.add_edge(0,2);
  G.add_edge(1,3);
  G.add_edge(2,3);
  G.add_edge(3,4);
  G.add_edge(4,5);
}

/* ---------------- runtime entry ---------------- */

constexpr uint64_t kCacheSize    = 256 * Region::kSize; // same style as other tests
constexpr uint64_t kFarMemSize   = (1ULL << 33);        // 8 GB for FakeDevice bring-up
constexpr uint64_t kNumGCThreads = 12;

// compile-time capacities for far CSR
static constexpr int32_t  kNVerts   = 6;
static constexpr uint64_t kMaxEdges = 16;

static void _main(void *arg) {
  // Build FarMemManager exactly like other AIFM tests
  unique_ptr<FarMemManager> manager(
      FarMemManagerFactory::build(kCacheSize, kNumGCThreads, new FakeDevice(kFarMemSize)));

  // 1) LOCAL baseline
  {
    CSRLocal L(kNVerts);
    build_toy_local(L);

    auto t0 = chrono::high_resolution_clock::now();
    auto dist = bfs_local(L, /*src=*/0, nullptr);
    auto t1 = chrono::high_resolution_clock::now();

    cout << "local: dist[5]=" << dist[5]
         << " ms=" << chrono::duration_cast<chrono::milliseconds>(t1 - t0).count()
         << "\n";
  }

  // 2) FAR CSR on AIFM Array<T,N>
  {
    // allocate arrays with manager, then wrap them by reference in CSRFar
    auto off = manager->allocate_array<int32_t, kNVerts>();
    auto deg = manager->allocate_array<int32_t, kNVerts>();
    auto nbr = manager->allocate_array<int32_t, kMaxEdges>();

    fargraph::CSRFar<kNVerts, kMaxEdges> F(off, deg, nbr);
    build_toy_far(F);
    F.finalize();  // copies CSR into far arrays with one DerefScope

    auto t0 = chrono::high_resolution_clock::now();
    auto dist = fargraph::bfs_far(F, /*src=*/0, nullptr);
    auto t1 = chrono::high_resolution_clock::now();

    cout << "far:   dist[5]=" << dist[5]
         << " ms=" << chrono::duration_cast<chrono::milliseconds>(t1 - t0).count()
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