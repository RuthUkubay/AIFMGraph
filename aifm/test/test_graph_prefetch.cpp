// aifm/test/test_graph_prefetch.cpp
extern "C" {
#include <runtime/runtime.h>
}

#include "device.hpp"    // FakeDevice
#include "manager.hpp"
#include "deref_scope.hpp"
#include "graph_adj.hpp" // your pointer-chasing graph header

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <queue>
#include <random>
#include <unordered_set>
#include <vector>

using namespace far_memory;
using std::cout;
using std::endl;

// ---- FarMem knobs (same flavor as your working tests) ----
constexpr uint64_t kCacheSize    = (128ULL << 20); // 128 MB local cache
constexpr uint64_t kFarMemSize   = (4ULL  << 30);  // 4 GB far memory
constexpr uint32_t kNumGCThreads = 12;

// ---- Policies (reuse your earlier shapes) ----
struct AllRemotePolicy : RemotingPolicy {
  uint16_t inline_capacity(Vid, uint32_t) const override { return 0; }
};
struct Inline8Policy : RemotingPolicy {
  uint16_t inline_capacity(Vid, uint32_t deg) const override {
    return deg <= 8 ? static_cast<uint16_t>(deg) : 0;
  }
};
struct Inline16Policy : RemotingPolicy {
  uint16_t inline_capacity(Vid, uint32_t deg) const override {
    // "Wish" for 16; header caps at 8 internally.
    return deg <= 16 ? static_cast<uint16_t>(deg) : 0;
  }
};

// ---- Graph generator (single shared edge list for A/B/C) ----
struct GenParams {
  uint64_t num_vertices;
  uint64_t num_edges;
  uint64_t seed;
};

static std::vector<std::pair<Vid,Vid>>
gen_random_edges(const GenParams& p) {
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

// ---- Placement stats (same definition you used) ----
struct PlacementStats {
  uint64_t verts_inline_any = 0;
  uint64_t verts_remote_any = 0;
  uint64_t bytes_remote     = 0;
};

static PlacementStats
measure_placement(GraphAdj& G) {
  PlacementStats st{};
  const uint64_t N = G.num_vertices();
  for (uint64_t u = 0; u < N; ++u) {
    DerefScope s;
    auto h = G.header_info(u, s);
    if (h.inline_len > 0) st.verts_inline_any++;
    if (h.degree > h.inline_len) {
      st.verts_remote_any++;
      st.bytes_remote += (h.degree - h.inline_len) * sizeof(Vid);
    }
  }
  return st;
}

static void print_stats(const char* name, const PlacementStats& st, uint64_t N) {
  cout << "[" << name << "] vertices=" << N
       << " inline_any=" << st.verts_inline_any
       << " remote_any=" << st.verts_remote_any
       << " bytes_remote=" << st.bytes_remote
       << "\n";
}

// ===== Traversal modes =====

// Mode 1: Baseline BFS (no prefetch)
static std::vector<int> bfs_baseline(GraphAdj& G, Vid src) {
  const uint64_t n = G.num_vertices();
  std::vector<int> dist(n, -1);
  if (src >= n) return dist;

  std::queue<Vid> q;
  dist[src] = 0; q.push(src);

  while (!q.empty()) {
    Vid u = q.front(); q.pop();
    DerefScope s;
    auto nv = G.neighbors(u, s);
    for (uint32_t i = 0; i < nv.inline_len; ++i) {
      Vid v = nv.inline_ptr[i];
      if (dist[v] == -1) { dist[v] = dist[u] + 1; q.push(v); }
    }
    for (uint32_t i = 0; i < nv.tail_len; ++i) {
      Vid v = nv.tail_ptr[i];
      if (dist[v] == -1) { dist[v] = dist[u] + 1; q.push(v); }
    }
  }
  return dist;
}

// Utility: prefetch headers for a set of vertices by coalescing contiguous IDs
static void prefetch_headers_for_set(GraphAdj& G, std::vector<Vid>& ids) {
  if (ids.empty()) return;
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());

  // Emit sequential spans to the prefetcher (stride = 1)
  uint64_t run_start = ids[0];
  uint64_t prev = ids[0];
  uint32_t run_len = 1;

  auto flush_run = [&](void) {
    G.prefetch_headers_span(run_start, run_len);
  };

  for (size_t i = 1; i < ids.size(); ++i) {
    if (ids[i] == prev + 1) {
      ++run_len; prev = ids[i];
    } else {
      flush_run();
      run_start = prev = ids[i];
      run_len = 1;
    }
  }
  flush_run();
}

// Mode 2: BFS with level-wise header prefetch.
// We collect the next wave of vertices, prefetch their headers before descending.
static std::vector<int> bfs_header_prefetch(GraphAdj& G, Vid src) {
  const uint64_t n = G.num_vertices();
  std::vector<int> dist(n, -1);
  if (src >= n) return dist;

  std::vector<Vid> curr, next;
  curr.reserve(1024); next.reserve(1024);
  dist[src] = 0; curr.push_back(src);

  int depth = 0;
  while (!curr.empty()) {
    // Discover next
    next.clear();
    for (Vid u : curr) {
      DerefScope s;
      auto nv = G.neighbors(u, s);
      for (uint32_t i = 0; i < nv.inline_len; ++i) {
        Vid v = nv.inline_ptr[i];
        if (dist[v] == -1) { dist[v] = depth + 1; next.push_back(v); }
      }
      for (uint32_t i = 0; i < nv.tail_len; ++i) {
        Vid v = nv.tail_ptr[i];
        if (dist[v] == -1) { dist[v] = depth + 1; next.push_back(v); }
      }
    }

    // Prefetch headers for 'next' before processing it
    prefetch_headers_for_set(G, next);

    curr.swap(next);
    ++depth;
  }
  return dist;
}

// Mode 3: BFS with header prefetch + tail warm hints under a budget.
// We prefetch headers for the whole next wave, then "warm" tails for the first K vertices.
static std::vector<int> bfs_header_and_tail(GraphAdj& G, Vid src, size_t tail_hint_budget = 8192) {
  const uint64_t n = G.num_vertices();
  std::vector<int> dist(n, -1);
  if (src >= n) return dist;

  std::vector<Vid> curr, next;
  curr.reserve(1024); next.reserve(1024);
  dist[src] = 0; curr.push_back(src);

  int depth = 0;
  while (!curr.empty()) {
    next.clear();
    for (Vid u : curr) {
      DerefScope s;
      auto nv = G.neighbors(u, s);
      for (uint32_t i = 0; i < nv.inline_len; ++i) {
        Vid v = nv.inline_ptr[i];
        if (dist[v] == -1) { dist[v] = depth + 1; next.push_back(v); }
      }
      for (uint32_t i = 0; i < nv.tail_len; ++i) {
        Vid v = nv.tail_ptr[i];
        if (dist[v] == -1) { dist[v] = depth + 1; next.push_back(v); }
      }
    }

    // Prefetch all headers in the next wave.
    prefetch_headers_for_set(G, next);

    // Warm a fraction of tails (budgeted) to hide first-touch stalls.
    // We just take the first few unique IDs from 'next' after de-dup.
    std::vector<Vid> uniq = next;
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    size_t warmed = 0;
    for (Vid v : uniq) {
      if (warmed >= tail_hint_budget) break;
      G.hint_tail_present(v);
      ++warmed;
    }

    curr.swap(next);
    ++depth;
  }
  return dist;
}

// ---- Timing harness ----
template <typename Fn>
static uint64_t time_us(Fn&& f, int iters) {
  using clk = std::chrono::high_resolution_clock;
  auto warm = f(); (void)warm;
  auto t0 = clk::now();
  for (int i = 0; i < iters; ++i) (void)f();
  auto t1 = clk::now();
  return (uint64_t) std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
}

// ---- Driver: build per policy, then run modes ----
static void run_suite(FarMemManager* manager) {
  GenParams gen { .num_vertices = 200000, .num_edges = 1200000, .seed = 1234 };
  auto edges = gen_random_edges(gen);
  cout << "Graph: |V|=" << gen.num_vertices << " |E|=" << edges.size() << "\n";

  struct {
    const char* name;
    RemotingPolicy* pol;
  } cases[] = {
    {"all_remote",      new AllRemotePolicy()},
    {"inline_8",        new Inline8Policy()},
    {"inline_16_cap8",  new Inline16Policy()},
  };

  const int iters = 5;

  for (auto &C : cases) {
    GraphAdj G(manager, gen.num_vertices, *C.pol);
    G.build_from_edges(edges);
    auto st = measure_placement(G);
    print_stats(C.name, st, gen.num_vertices);

    auto t0 = time_us([&]{ return bfs_baseline(G, 0); }, iters);
    auto t1 = time_us([&]{ return bfs_header_prefetch(G, 0); }, iters);
    auto t2 = time_us([&]{ return bfs_header_and_tail(G, 0, /*tail_hint_budget=*/4096); }, iters);

    cout << "BFS[" << C.name << "]  "
         << "baseline_us=" << (double)t0/iters
         << " | header_pf_us=" << (double)t1/iters
         << " | header+tail_pf_us=" << (double)t2/iters
         << "\n";

    delete C.pol;
  }
  cout << "Done.\n";
}

// ---- Shenango/AIFM glue ----
static void _main(void*) {
  std::unique_ptr<FarMemManager> manager(
      FarMemManagerFactory::build(kCacheSize, kNumGCThreads, new FakeDevice(kFarMemSize)));
  run_suite(manager.get());
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " [cfg_file]\n";
    return -EINVAL;
  }
  int ret = runtime_init(argv[1], _main, nullptr);
  if (ret) {
    std::cerr << "failed to start runtime\n";
    return ret;
  }
  return 0;
}
