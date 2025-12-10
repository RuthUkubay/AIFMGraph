#pragma once

extern "C" {
#include <runtime/thread.h>
}

#include "deref_scope.hpp"
#include "pointer.hpp"
#include "manager.hpp"
#include "array.hpp"

#include <cstdint>
#include <vector>
#include <utility>
#include <cassert>

namespace far_memory {

// Compact vertex id
using Vid = uint32_t;

/**
 * Per-vertex header kept in local cache when hot.
 * degree : number of neighbors
 * nbrs   : remoteable byte array holding 'degree' Vid entries (contiguous)
 *
 * 'nbrs' is marked mutable so we can call .deref() from const methods
 * (the handle's deref is non-const, but it produces a const void* for reads).
 */
struct VertexHdr {
  uint32_t               degree{0};
  mutable GenericUniquePtr nbrs;
};

/**
 * Thin wrapper over GenericArray to store VertexHdr in far-mem objects,
 * one object per vertex. Each slot() returns the handle to that object.
 */
class VertexArray : public GenericArray {
public:
  inline VertexArray(FarMemManager* mgr, uint64_t n_vertices)
    : GenericArray(mgr, /*item_size=*/sizeof(VertexHdr),
                        /*num_items=*/n_vertices) {}

  // Non-const context: return the handle directly.
  inline GenericUniquePtr* slot(uint64_t i) { return at(false, i); }

  // Const context: still return a non-const handle so we can call deref().
  inline GenericUniquePtr* slot(uint64_t i) const {
    return const_cast<VertexArray*>(this)->at(false, i);
  }

  inline uint64_t size() const { return kNumItems_; }
};

/**
 * GraphAdj: adjacency-list graph over AIFM.
 * Layout: one VertexHdr object per vertex; its 'nbrs' points to a remoteable
 * byte array of neighbor Vids.
 */
class GraphAdj {
public:
  inline GraphAdj(FarMemManager* mgr, uint64_t n_vertices)
      : mgr_(mgr), verts_(mgr, n_vertices) {}

  inline uint64_t num_vertices() const { return verts_.size(); }

  /**
   * Build from edge list (directed). Assumes vertices are in [0, N).
   * Two-pass: (1) count degrees, (2) allocate & fill neighbor arrays.
   */
  inline void build_from_edges(const std::vector<std::pair<Vid, Vid>>& edges) {
    const uint64_t N = verts_.size();

    // Pass 1: host-side degree counts
    std::vector<uint32_t> deg(N, 0);
    for (auto [u, v] : edges) {
      assert(u < N && v < N);
      ++deg[u];
    }

    // Pass 2a: write headers & allocate neighbor arrays
    {
      DerefScope scope;
      for (uint64_t u = 0; u < N; ++u) {
        auto& vh = deref_vertex(scope, u);  // mutable header mapping
        vh.degree = deg[u];
        if (vh.degree == 0) {
          vh.nbrs = GenericUniquePtr{};     // empty
          continue;
        }
        const uint16_t bytes = static_cast<uint16_t>(vh.degree * sizeof(Vid));
        vh.nbrs = mgr_->allocate_generic_unique_ptr(kVanillaPtrDSID, bytes);
      }
    }

    // Pass 2b: fill neighbor arrays
    std::vector<uint32_t> cur(N, 0);
    {
      DerefScope scope;
      for (auto [u, v] : edges) {
        auto& vh = deref_vertex(scope, u); // mutable header -> can deref_mut
        if (vh.degree == 0) continue;
        auto* base = static_cast<uint8_t*>(vh.nbrs.deref_mut(scope));
        reinterpret_cast<Vid*>(base)[cur[u]++] = v;
      }
    }
  }

  struct NeighborView {
    const Vid* ptr{nullptr};
    uint32_t   len{0};
  };

  /** Read-only neighbor view for vertex u (valid while 'scope' lives). */
  inline NeighborView neighbors(uint64_t u, DerefScope& scope) const {
    const auto& vh = const_deref_vertex(scope, u);   // read header
    if (vh.degree == 0) return {nullptr, 0};
    // In a const method we still need a non-const handle to call deref().
    auto* h = verts_.slot(u);
    const auto* base = static_cast<const uint8_t*>(h->deref(scope));
    return { reinterpret_cast<const Vid*>(base), vh.degree };
  }

  /** Degree of vertex u. */
  inline uint32_t degree(uint64_t u, DerefScope& scope) const {
    return const_deref_vertex(scope, u).degree;
  }

private:
  FarMemManager* mgr_{nullptr};
  VertexArray    verts_;

  // Map the header for mutation
  inline VertexHdr& deref_vertex(const DerefScope& scope, uint64_t i) {
    void* p = verts_.slot(i)->deref_mut(scope);
    return *reinterpret_cast<VertexHdr*>(p);
  }

  // Map the header read-only
  inline const VertexHdr& const_deref_vertex(const DerefScope& scope, uint64_t i) const {
    auto* h = verts_.slot(i);            // non-const handle in const method
    const void* p = h->deref(scope);     // returns const void*
    return *reinterpret_cast<const VertexHdr*>(p);
  }
};

} // namespace far_memory
