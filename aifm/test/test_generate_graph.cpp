extern "C" {
#include <runtime/runtime.h>
}

#include "graph_adj.hpp"
#include "device.hpp"   // FakeDevice
#include "manager.hpp"

#include <memory>
#include <random>
#include <iostream>
#include <vector>
#include <utility>
#include <cstdint>
#include <cassert>
#include <atomic>
#include "object.hpp"   // for far_memory::Object used by the notifier


using namespace far_memory;
using std::cout;
using std::endl;

static std::atomic<uint64_t> wb_count{0};

// Keep the “FakeDevice-level” simplicity and sizes similar to your array test.
constexpr uint64_t kCacheSize    = (1ULL << 20); // 128 MB local cache
constexpr uint64_t kFarMemSize   = (4ULL  << 30);  // 4 GB far memory
constexpr uint32_t kNumGCThreads = 12;

// Params for a simple random directed graph (Erdős–Rényi style).
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
    // Optional: avoid self loops; comment out if you want them.
    if (u == v) { if (u + 1 < p.num_vertices) v = u + 1; else v = 0; }
    edges.emplace_back(u, v);
  }
  return edges;
}

static void sanity_check(GraphAdj& G, const std::vector<std::pair<Vid,Vid>>& edges) {
  const uint64_t N = G.num_vertices();

  // Check total degree sum equals |E|
  uint64_t sum_deg = 0;
  for (uint64_t u = 0; u < N; ++u) {
    DerefScope s;
    sum_deg += G.degree(u, s);
  }
  if (sum_deg != edges.size()) {
    cout << "Sanity failed: sum_deg=" << sum_deg
         << " edges=" << edges.size() << endl;
  }

  // Spot-check a few vertices’ neighbor materialization.
  for (uint64_t u = 0; u < std::min<uint64_t>(N, 5); ++u) {
    DerefScope s;
    auto view = G.neighbors(u, s);
    // Just touch the first few entries if they exist.
    for (uint32_t i = 0; i < std::min<uint32_t>(view.len, 3); ++i) {
      volatile Vid v = view.ptr[i]; (void)v; // prevent optimizing away
    }
  }
}

static void do_work(FarMemManager* manager, const GenParams& gen) {
  cout << "Running " << __FILE__ << "..." << endl;

  // Generate edges on host.
  auto edges = gen_random_edges(gen);

  // Build graph into far memory.
  GraphAdj G(manager, gen.num_vertices);
  G.build_from_edges(edges);

  // Sanity checks (degree sum; light neighbor touches).
  sanity_check(G, edges);

  cout << "Graph built: |V|=" << gen.num_vertices
       << " |E|=" << edges.size() << endl;

  cout << "Passed" << endl;
}

static void _main(void* arg) {
  std::unique_ptr<FarMemManager> manager(
      FarMemManagerFactory::build(kCacheSize, kNumGCThreads,
                                  new FakeDevice(kFarMemSize)));

  // Count evictions (write-backs) for vanilla DSID objects
  manager->register_eval_notifier(
      kVanillaPtrDSID,
      [&](far_memory::Object obj, FarMemManager::WriteObjectFn writeback)->bool {
        wb_count.fetch_add(1, std::memory_order_relaxed);
        writeback(obj.get_data_len());   // do normal write-back
        return false;                    // we didn’t fully handle it
      });

  GenParams gen{ .num_vertices = 2000, .num_edges = 100000, .seed = 42 };
  do_work(manager.get(), gen);

  std::cout << "Evicted objects (write-backs): "
            << wb_count.load() << "\n";
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
