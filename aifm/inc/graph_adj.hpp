#pragma once
extern "C" { #include <runtime/thread.h> }

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

struct VertexHdr {
  uint32_t        degree{0};
  GenericUniquePtr nbrs;
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

    std::vector<uint32_t> deg(N, 0);
    for (auto [u,v] : edges) { assert(u < N && v < N); ++deg[u]; }

    {
      DerefScope scope;
      for (uint64_t u = 0; u < N; ++u) {
        auto& vh = deref_vertex(scope, u);
        vh.degree = deg[u];
        if (vh.degree == 0) { vh.nbrs = GenericUniquePtr{}; continue; }
        const uint16_t bytes = static_cast<uint16_t>(vh.degree * sizeof(Vid));
        vh.nbrs = mgr_->allocate_generic_unique_ptr(kVanillaPtrDSID, bytes);
      }
    }

    std::vector<uint32_t> cur(N, 0);
    {
      DerefScope scope;
      for (auto [u,v] : edges) {
        const auto& vh = const_deref_vertex(scope, u);
        if (vh.degree == 0) continue;
        auto* base = static_cast<uint8_t*>(vh.nbrs.deref(scope));
        reinterpret_cast<Vid*>(base)[cur[u]++] = v;
      }
    }
  }

  struct NeighborView { const Vid* ptr{nullptr}; uint32_t len{0}; };

  inline NeighborView neighbors(uint64_t u, DerefScope& scope) const {
    const auto& vh = const_deref_vertex(scope, u);
    if (vh.degree == 0 || !vh.nbrs.valid()) return {nullptr, 0};
    const auto* base = static_cast<const uint8_t*>(vh.nbrs.deref(scope));
    return { reinterpret_cast<const Vid*>(base), vh.degree };
  }

  inline uint32_t degree(uint64_t u, DerefScope& scope) const {
    return const_deref_vertex(scope, u).degree;
  }

private:
  FarMemManager* mgr_{nullptr};
  VertexArray    verts_;

  inline VertexHdr& deref_vertex(const DerefScope& scope, uint64_t i) {
    void* p = verts_.slot(i)->deref(scope);
    return *reinterpret_cast<VertexHdr*>(p);
  }
  inline const VertexHdr& const_deref_vertex(const DerefScope& scope, uint64_t i) const {
    const void* p = verts_.slot(i)->deref(scope);
    return *reinterpret_cast<const VertexHdr*>(p);
  }
};

} // namespace far_memory
