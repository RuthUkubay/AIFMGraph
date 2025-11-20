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
 * - N_VERTS: number of vertices (fixed at compile time)
 * - MAX_EDGES: capacity for edges (upper bound; enforce at finalize)
 *
 * AIFM notes:
 *  - Array<T,N> must be constructed with FarMemManager* (no default/move).
 *  - at()/at_mut() are non-const -> don't call them through const objects.
 *  - Keep exactly one live DerefScope per thread when accessing far memory.
 */
template <i32 N_VERTS, uint64_t MAX_EDGES>
struct CSRFar {
  static_assert(N_VERTS > 0, "N_VERTS must be > 0");
  static_assert(MAX_EDGES > 0, "MAX_EDGES must be > 0");
  static constexpr i32 kN = N_VERTS;

  // Far arrays (constructed with manager in ctor init-list)
  far_memory::Array<i32, N_VERTS>   off;
  far_memory::Array<i32, N_VERTS>   deg;
  far_memory::Array<i32, MAX_EDGES> nbr;

  // Host-side builder before finalize()
  std::vector<std::vector<i32>> adj_tmp;

  // IMPORTANT: Array<T,N> requires FarMemManager* here
  explicit CSRFar(far_memory::FarMemManager* mm)
      : off(mm), deg(mm), nbr(mm), adj_tmp(N_VERTS) {}

  inline void add_edge(i32 u, i32 v) {
    if (u < 0 || u >= N_VERTS || v < 0 || v >= N_VERTS) return;
    adj_tmp[u].push_back(v);
  }

  // Build CSR locally and copy into far arrays.
  inline void finalize() {
    std::vector<i32> off_l(N_VERTS), deg_l(N_VERTS);
    uint64_t E = 0;
    for (i32 u = 0; u < N_VERTS; ++u) {
      off_l[u] = static_cast<i32>(E);
      deg_l[u] = static_cast<i32>(adj_tmp[u].size());
      E += static_cast<uint64_t>(deg_l[u]);
    }
    if (E > MAX_EDGES) {
      fprintf(stderr,
              "CSRFar::finalize: edges=%lu exceed MAX_EDGES=%lu\n",
              (unsigned long)E, (unsigned long)MAX_EDGES);
      abort();
    }

    far_memory::DerefScope scope;  // one live scope
    for (i32 u = 0; u < N_VERTS; ++u) {
      off.at_mut(scope, u) = off_l[u];
      deg.at_mut(scope, u) = deg_l[u];
    }
    uint64_t idx = 0;
    for (i32 u = 0; u < N_VERTS; ++u) {
      for (i32 v : adj_tmp[u]) {
        nbr.at_mut(scope, static_cast<i32>(idx++)) = v;
      }
    }
    adj_tmp.clear(); adj_tmp.shrink_to_fit();
  }

  // Non-const: at() is non-const
  inline void neighbor_span(i32 u, far_memory::DerefScope& scope,
                            i32& start, i32& len) {
    start = off.at(scope, u);
    len   = deg.at(scope, u);
  }
};

// BFS over far CSR
template <i32 N_VERTS, uint64_t MAX_EDGES>
inline std::vector<int> bfs_far(CSRFar<N_VERTS, MAX_EDGES>& G, // non-const
                                int src, std::vector<int>* parent = nullptr) {
  const int n = N_VERTS;
  std::vector<int> dist(n, -1);
  if (parent) parent->assign(n, -1);

  if (src < 0 || src >= n) return dist;

  std::queue<int> q;
  dist[src] = 0;
  if (parent) (*parent)[src] = src;
  q.push(src);

  far_memory::DerefScope scope; // one live scope

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
