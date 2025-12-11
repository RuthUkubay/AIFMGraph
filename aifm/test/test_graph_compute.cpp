// aifm/test/test_graph_compute.cpp
extern "C" {
#include <runtime/runtime.h>
}

#include "device.hpp"
#include "manager.hpp"
#include "deref_scope.hpp"
#include "graph_adj.hpp"

#include <cassert>
#include <cerrno>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

using namespace far_memory;
using std::cout;
using std::endl;

// Use an "all remote" placement so everything goes through far memory.
struct AllRemote : RemotingPolicy {
  uint16_t inline_capacity(Vid, uint32_t) const override { return 0; }
};

constexpr uint64_t kCacheSize    = (64ULL  << 20);  // small is fine
constexpr uint64_t kFarMemSize   = (1ULL   << 30);
constexpr uint32_t kNumGCThreads = 4;

static void _main(void *) {
  // Build a FarMemManager with a FakeDevice (in-process remote server).
  std::unique_ptr<FarMemManager> manager(
      FarMemManagerFactory::build(kCacheSize, kNumGCThreads,
                                  new FakeDevice(kFarMemSize)));

  // Tiny graph; we don't actually care about edges for this smoke test.
  const uint64_t N = 16;
  AllRemote pol;
  GraphAdj G(manager.get(), N, pol);

  // Empty edge list is fine; this just initializes headers to degree=0.
  std::vector<std::pair<Vid, Vid>> edges;
  G.build_from_edges(edges);

  // Frontier whose elements we expect the remote to sum.
  // This matches the "sum frontier[i]" behavior in ServerGraphAgg::compute().
  std::vector<Vid> frontier = {1, 2, 3, 4, 5};
  uint64_t expected = 0;
  for (auto v : frontier) {
    expected += static_cast<uint64_t>(v);
  }

  // Call our graph remote compute path.
  uint64_t remote_sum = G.remote_degree_sum(frontier);

  cout << "test_graph_compute: expected sum = " << expected
       << ", remote_sum = " << remote_sum << endl;

  if (remote_sum != expected) {
    std::cerr << "ERROR: remote_degree_sum mismatch!" << std::endl;
    // You can BUG() or just assert.
    assert(false && "remote_degree_sum mismatch");
  } else {
    cout << "test_graph_compute: PASS" << endl;
  }
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " [cfg_file]\n";
    return -EINVAL;
  }
  return runtime_init(argv[1], _main, nullptr);
}
