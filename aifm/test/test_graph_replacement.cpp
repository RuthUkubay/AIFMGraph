// // aifm/test/test_graph_placement.cpp
// extern "C" {
// #include <runtime/runtime.h>
// }

// #include "device.hpp"    // FakeDevice
// #include "manager.hpp"
// #include "deref_scope.hpp"
// #include "graph_adj.hpp" // pointer-chasing graph header (in inc/)
// #include <algorithm>
// #include <chrono>
// #include <iostream>
// #include <memory>
// #include <queue>    // <-- needed for std::queue
// #include <random>
// #include <vector>

// // --------- FarMem knobs (same style as your working tests) ----------
// using namespace far_memory;
// using std::cout;
// using std::endl;

// constexpr uint64_t kCacheSize    = (128ULL << 20); // 128MB local cache
// constexpr uint64_t kFarMemSize   = (4ULL  << 30);  // 4GB far memory
// constexpr uint32_t kNumGCThreads = 12;

// // --------- Graph generator (one shared edge list for apples-to-apples) ----------
// struct GenParams {
//   uint64_t num_vertices;
//   uint64_t num_edges;
//   uint64_t seed;
// };

// static std::vector<std::pair<Vid,Vid>>
// gen_random_edges(const GenParams& p) {
//   std::mt19937_64 rng(p.seed);
//   std::uniform_int_distribution<uint64_t> U(0, p.num_vertices - 1);
//   std::vector<std::pair<Vid,Vid>> edges;
//   edges.reserve(p.num_edges);
//   for (uint64_t i = 0; i < p.num_edges; ++i) {
//     Vid u = static_cast<Vid>(U(rng));
//     Vid v = static_cast<Vid>(U(rng));
//     if (u == v) v = (u + 1 < p.num_vertices) ? u + 1 : 0; // avoid self loop
//     edges.emplace_back(u, v);
//   }
//   return edges;
// }

// // --------- Placement policies ----------
// struct AllRemotePolicy : RemotingPolicy {
//   uint16_t inline_capacity(Vid, uint32_t) const override { return 0; }
// };

// struct Inline8Policy : RemotingPolicy {
//   uint16_t inline_capacity(Vid, uint32_t deg) const override {
//     return deg <= 8 ? static_cast<uint16_t>(deg) : 0;
//   }
// };

// struct Inline16Policy : RemotingPolicy {
//   uint16_t inline_capacity(Vid, uint32_t deg) const override {
//     // “Wish” for 16; header currently caps at 8 inside VertexHdr.
//     return deg <= 16 ? static_cast<uint16_t>(deg) : 0;
//   }
// };

// // --------- Stats helpers ----------
// struct PlacementStats {
//   uint64_t verts_inline_any = 0;   // vertices with inline_len > 0
//   uint64_t verts_remote_any = 0;   // vertices with (degree - inline_len) > 0
//   uint64_t bytes_remote     = 0;   // total tail bytes
// };

// static PlacementStats
// measure_placement(GraphAdj& G) {
//   PlacementStats st{};
//   const uint64_t N = G.num_vertices();
//   for (uint64_t u = 0; u < N; ++u) {
//     DerefScope s;
//     auto h = G.header_info(u, s);
//     if (h.inline_len > 0) st.verts_inline_any++;
//     if (h.degree > h.inline_len) {
//       st.verts_remote_any++;
//       st.bytes_remote += (h.degree - h.inline_len) * sizeof(Vid);
//     }
//   }
//   return st;
// }

// static void print_stats(const char* name, const PlacementStats& st, uint64_t N) {
//   cout << "[" << name << "] vertices=" << N
//        << " inline_any=" << st.verts_inline_any
//        << " remote_any=" << st.verts_remote_any
//        << " bytes_remote=" << st.bytes_remote
//        << "\n";
// }

// // --------- Minimal BFS over GraphAdj (reads inline and tail spans) ----------
// static std::vector<int> bfs(GraphAdj &G, Vid src) {
//   const uint64_t n = G.num_vertices();
//   std::vector<int> dist(n, -1);
//   if (src >= n) return dist;

//   std::queue<Vid> q;
//   dist[src] = 0; 
//   q.push(src);

//   while (!q.empty()) {
//     Vid u = q.front(); 
//     q.pop();

//     DerefScope scope;
//     G.for_each_neighbor(u, scope, [&](Vid v) {
//       if (dist[v] == -1) {
//         dist[v] = dist[u] + 1;
//         q.push(v);
//       }
//     });
//   }
//   return dist;
// }

// // --------- BFS timing ----------
// static void run_bfs_and_time(GraphAdj& G, Vid src, int iters, const char* tag) {
//   using clk = std::chrono::high_resolution_clock;
//   auto warm = bfs(G, src); (void)warm;

//   auto t0 = clk::now();
//   for (int i = 0; i < iters; ++i) {
//     auto dist = bfs(G, src); (void)dist;
//   }
//   auto t1 = clk::now();
//   auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();

//   cout << "BFS[" << tag << "] iters=" << iters
//        << " total_us=" << us
//        << " per_iter_us=" << (double)us / iters
//        << "\n";
// }

// // --------- Work driver: build once per policy and compare ----------
// static void do_work(FarMemManager* manager) {
//   GenParams gen { .num_vertices = 200000, .num_edges = 1200000, .seed = 42 };
//   auto edges = gen_random_edges(gen);
//   cout << "Graph: |V|=" << gen.num_vertices << " |E|=" << edges.size() << "\n";

//   // 1) All-remote baseline
//   {
//     AllRemotePolicy pol;
//     GraphAdj G(manager, gen.num_vertices, pol);
//     G.build_from_edges(edges);

//     auto st = measure_placement(G);
//     print_stats("all_remote", st, gen.num_vertices);
//     run_bfs_and_time(G, /*src=*/0, /*iters=*/5, "all_remote");
//   }

//   // 2) Inline up to 8
//   {
//     Inline8Policy pol;
//     GraphAdj G(manager, gen.num_vertices, pol);
//     G.build_from_edges(edges);

//     auto st = measure_placement(G);
//     print_stats("inline_8", st, gen.num_vertices);
//     run_bfs_and_time(G, /*src=*/0, /*iters=*/5, "inline_8");
//   }

//   // 3) Inline up to 16 (will cap at 8 in current header; useful A/B if you raise cap later)
//   {
//     Inline16Policy pol;
//     GraphAdj G(manager, gen.num_vertices, pol);
//     G.build_from_edges(edges);

//     auto st = measure_placement(G);
//     print_stats("inline_16(cap_8)", st, gen.num_vertices);
//     run_bfs_and_time(G, /*src=*/0, /*iters=*/5, "inline_16");
//   }

//   cout << "Done.\n";
// }

// // --------- Shenango/AIFM runtime glue ----------
// static void _main(void* /*arg*/) {
//   std::unique_ptr<FarMemManager> manager(
//       FarMemManagerFactory::build(kCacheSize, kNumGCThreads, new FakeDevice(kFarMemSize)));
//   do_work(manager.get());
// }

// int main(int argc, char* argv[]) {
//   if (argc < 2) {
//     std::cerr << "usage: " << argv[0] << " [cfg_file]\n";
//     return -EINVAL;
//   }
//   int ret = runtime_init(argv[1], _main, nullptr);
//   if (ret) {
//     std::cerr << "failed to start runtime\n";
//     return ret;
//   }
//   return 0;
// }
