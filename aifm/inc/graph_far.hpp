#pragma once
#include <cstdint>
#include <vector>
#include <queue>
#include <cstring>

#include "array.hpp"        // far_memory::Array<T, N>
#include "deref_scope.hpp"  // far_memory::DerefScope
#include "manager.hpp"      // FarMemManager

namespace fargraph {
using i32 = int32_t;

/* Far-memory CSR with compile-time capacities:
 * - N_VERTS: number of vertices (fixed at compile-time)
 * - MAX_EDGES: capacity for edges (upper bound; we assert E <= MAX_EDGES at finalize)
 *
 * Layout:
 *   off[u] : start index into nbr
 *   deg[u] : out-degree
 *   nbr[off[u] .. off[u]+deg[u]-1] : neighbor IDs (0..N_VERTS-1)
 *
 * AIFM usage:
 *   - Arrays are far_memory::Array<T, N> allocated via manager->allocate_array<T, N>().
 *   - Access uses ONE live DerefScope per thread (scope-based at()/at_mut()).
 */
template <i32 N_VERTS, uint64_t MAX_EDGES>
struct CSRFar {
  static_assert(N_VERTS > 0, "N_VERTS must be > 0");
  static_assert(MAX_EDGES > 0, "MAX_EDGES must be > 0");

  // public for convenience in tests
  static constexpr i32 kN = N_VERTS;

  // Far arrays
  far_memory::Array<i32, N_VERTS> off;
  far_memory::Array<i32, N_VERTS> deg;
  far_memory::Array<i32, MAX_EDGES> nbr;

  // Builder state (host-side) before finalize()
  std::vector<std::vector<i32>> adj_tmp;

  CSRFar() : adj_tmp(N_VERTS) {}

  inline void add_edge(i32 u, i32 v) {
    // simple bounds checks (IDs must be 0..N_VERTS-1)
    if (u < 0 || u >= N_VERTS || v < 0 || v >= N_VERTS) return;
    adj_tmp[u].push_back(v);
  }

  // Allocate far arrays and copy CSR (checks MAX_EDGES).
  inline void finalize(far_memory::FarMemManager* mm) {
    // Allocate far arrays with compile-time sizes (matches your array test)
    off = mm->allocate_array<i32, N_VERTS>();
    deg = mm->allocate_array<i32, N_VERTS>();
    nbr = mm->allocate_array<i32, MAX_EDGES>();

    // Build CSR locally
    std::vector<i32> off_l(N_VERTS), deg_l(N_VERTS);
    uint64_t E = 0;
    for (i32 u = 0; u < N_VERTS; ++u) {
      off_l[u] = static_cast<i32>(E);
      deg_l[u] = static_cast<i32>(adj_tmp[u].size());
      E += static_cast<uint64_t>(deg_l[u]);
    }
    // guard against overflow of capacity
    // (hard fail now; or you can clamp / drop edges if you prefer)
    if (E > MAX_EDGES) {
      // Minimal fail-fast; replace with your logging if needed.
      fprintf(stderr, "CSRFar::finalize: E=%lu exceeds MAX_EDGES=%lu\n",
              static_cast<unsigned long>(E),
              static_cast<unsigned long>(MAX_EDGES));
      abort();
    }

    // Copy into far arrays with ONE live scope
    far_memory::DerefScope scope;
    for (i32 u = 0; u < N_VERTS; ++u) {
      off.at_mut(scope, u) = off_l[u];
      deg.at_mut(scope, u) = deg_l[u];
    }

    // Flatten neighbors and copy
    uint64_t idx = 0;
    for (i32 u = 0; u < N_VERTS; ++u) {
      for (i32 v : adj_tmp[u]) {
        nbr.at_mut(scope, static_cast<i32>(idx++)) = v;
      }
    }

    // Drop builder
    adj_tmp.clear();
    adj_tmp.shrink_to_fit();
  }

  // Fetch neighbor span for u using the SAME scope.
  inline void neighbor_span(i32 u, far_memory::DerefScope& scope,
                            i32& start, i32& len) const {
    start = off.at(scope, u);
    len   = deg.at(scope, u);
  }
};

// BFS over far CSR (keys are 0..N_VERTS-1)
template <i32 N_VERTS, uint64_t MAX_EDGES>
inline std::vector<int> bfs_far(const CSRFar<N_VERTS, MAX_EDGES>& G,
                                int src, std::vector<int>* parent = nullptr) {
  const int n = N_VERTS;
  std::vector<int> dist(n, -1);
  if (parent) parent->assign(n, -1);

  std::queue<int> q;
  if (src < 0 || src >= n) return dist;
  dist[src] = 0;
  if (parent) (*parent)[src] = src;
  q.push(src);

  far_memory::DerefScope scope;  // ONE live scope

  while (!q.empty()) {
    int u = q.front(); q.pop();

    i32 start = 0, len = 0;
    G.neighbor_span(u, scope, start, len);

    for (i32 i = 0; i < len; ++i) {
      int v = G.nbr.at(scope, static_cast<i32>(start + i));
      if (dist[v] == -1) {
        dist[v] = dist[u] + 1;
        if (parent) (*parent)[v] = u;
        q.push(v);
      }
    }
  }
  return dist;
}

} // namespace fargraph
