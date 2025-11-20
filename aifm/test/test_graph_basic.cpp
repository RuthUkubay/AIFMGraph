// aifm/test/graphs/driver.cpp
extern "C" {
#include <runtime/runtime.h>
}

#include "deref_scope.hpp"
#include "device.hpp"
#include "manager.hpp"
#include "pointer.hpp"

#include "graph_local.hpp"
#include "graph_far.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

using namespace std;
using namespace far_memory;

// ---------- toy builders ----------
static void build_toy_local(CSRLocal& G){
  G.add_edge(0,1);
  G.add_edge(0,2);
  G.add_edge(1,3);
  G.add_edge(2,3);
  G.add_edge(3,4);
  G.add_edge(4,5);
  G.finalize();
}

static void build_toy_far(fargraph::CSRFar& G){
  G.add_edge(0,1);
  G.add_edge(0,2);
  G.add_edge(1,3);
  G.add_edge(2,3);
  G.add_edge(3,4);
  G.add_edge(4,5);
}

// ---------- runtime entry ----------
constexpr uint64_t kCacheSize   = 256 * Region::kSize;   // same constants as your example
constexpr uint64_t kFarMemSize  = (1ULL << 33);          // 8GB fake device for bring-up
constexpr uint64_t kNumGCThreads = 12;

static void _main(void *arg) {
  // 1) Build FarMemManager exactly like your example test
  unique_ptr<FarMemManager> manager(
      FarMemManagerFactory::build(kCacheSize, kNumGCThreads, new FakeDevice(kFarMemSize)));

  // 2) LOCAL sanity
  {
    CSRLocal L(6);
    build_toy_local(L);
    auto t0 = chrono::high_resolution_clock::now();
    auto dist = bfs_local(L, 0, nullptr);
    auto t1 = chrono::high_resolution_clock::now();
    cout << "local: dist[5]=" << dist[5]
         << " ms=" << chrono::duration_cast<chrono::milliseconds>(t1-t0).count()
         << "\n";
  }

  // 3) FAR graph on top of manager
  {
    fargraph::CSRFar F(6);
    build_toy_far(F);
    F.finalize(manager.get());   // allocate + copy CSR into far memory

    auto t0 = chrono::high_resolution_clock::now();
    auto dist = fargraph::bfs_far(F, 0, nullptr);
    auto t1 = chrono::high_resolution_clock::now();
    cout << "far:   dist[5]=" << dist[5]
         << " ms=" << chrono::duration_cast<chrono::milliseconds>(t1-t0).count()
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
