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

using Vid = uint32_t;

// Make nbrs 'mutable' so we can deref in const methods (logical read-only).
struct VertexHdr {
  uint32_t              degree{0};
  mutable GenericUniquePtr nbrs;   // null if degree==0
};

class VertexArray : public GenericArray {
public:
  inline VertexArray(FarMemManager* mgr, uint64_t n_vertices)
    : GenericArray(mgr, /*item_size=*/sizeof(VertexHdr),
                        /*num_items=*/n_vertices) {}

  inline GenericUniquePtr*       slot(uint64_t i)       { return at(false, i); }
  inline const GenericUniquePtr* slot(uint64_t i) const { return const_cast<VertexArray*>(this)->at(false, i); }
  inline uint64_t                size() const           { return kNumItems_; }
};

class GraphAdj {
public:
  inline GraphAdj(FarMemManager* mgr, uint64_t n_vertices)
    : mgr_(mgr), verts_(mgr, n_vertices) {}

  inline uint64_t num_vertices() const { return verts_.size(); }

  inline void build_from_edges(const std::vector<std::pair<Vid,Vid>>& edges) {
    const uint64_t N = verts_.size();

    // Phase 1: degree counts
    std::vector<uint32_t> deg(N, 0);
    for (auto [u,v] : edges) { assert(u < N && v < N); ++deg[u]; }

    // Phase 2: write headers + allocate adjacency arrays
    {
      DerefScope scope;
      for (uint64_t u = 0; u < N; ++u) {
        auto& vh = deref_vertex(scope, u);              // MUTABLE header
        vh.degree = deg[u];
        if (vh.degree == 0) { vh.nbrs = GenericUniquePtr{}; continue; }
        const uint16_t bytes = static_cast<uint16_t>(vh.degree * sizeof(Vid));
        vh.nbrs = mgr_->allocate_generic_unique_ptr(kVanillaPtrDSID, bytes);
      }
    }

    // Phase 3: fill neighbors (mutate arrays)
    std::vector<uint32_t> cur(N, 0);
    {
      DerefScope scope;
      for (auto [u,v] : edges) {
        auto& vh = deref_vertex(scope, u);              // MUTABLE header (so nbrs is non-const)
        if (vh.degree == 0) continue;
        auto* base = static_cast<uint8_t*>(vh.nbrs.deref_mut(scope));
        reinterpret_cast<Vid*>(base)[cur[u]++] = v;
      }
    }
  }

  struct NeighborView { const Vid* ptr{nullptr}; uint32_t len{0}; };

  inline NeighborView neighbors(uint64_t u, DerefScope& scope) const {
    const auto& vh = const_deref_vertex(scope, u);      // READ header
    if (vh.degree == 0) return {nullptr, 0};
    // verts_.slot(u) returns const GenericUniquePtr* in a const method.
    // Cast away const to invoke non-const deref(); result is const void*.
    const auto* slot = const_cast<GenericUniquePtr*>(verts_.slot(u));
    const auto* base = static_cast<const uint8_t*>(slot->deref(scope));
    return { reinterpret_cast<const Vid*>(base), vh.degree };
  }

  inline uint32_t degree(uint64_t u, DerefScope& scope) const {
    return const_deref_vertex(scope, u).degree;
  }

private:
  FarMemManager* mgr_{nullptr};
  VertexArray    verts_;

  inline VertexHdr& deref_vertex(const DerefScope& scope, uint64_t i) {
    void* p = verts_.slot(i)->deref_mut(scope);         // MUTABLE header
    return *reinterpret_cast<VertexHdr*>(p);
  }
  inline const VertexHdr& const_deref_vertex(const DerefScope& scope, uint64_t i) const {
    // Need non-const handle to call deref(); returned data pointer is const.
    auto* slot = const_cast<GenericUniquePtr*>(verts_.slot(i));
    const void* p = slot->deref(scope);                 // READ-ONLY header
    return *reinterpret_cast<const VertexHdr*>(p);
  }
};

} // namespace far_memory
