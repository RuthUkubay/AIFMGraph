#include "graph_adj.hpp"
#include <cstring>   // optional (memcpy), not used here but handy

namespace far_memory {

/* ---------------- VertexArray ---------------- */

VertexArray::VertexArray(FarMemManager* mgr, uint64_t n_vertices)
: GenericArray(mgr, /*item_size=*/sizeof(Vertex), /*num_items=*/n_vertices) {}

inline GenericUniquePtr* VertexArray::slot(uint64_t i) {
  return at(/*nt=*/false, i);
}

inline const GenericUniquePtr* VertexArray::slot(uint64_t i) const {
  return const_cast<VertexArray*>(this)->at(false, i);
}

/* ---------------- GraphAdj ---------------- */

GraphAdj::GraphAdj(FarMemManager* mgr, uint64_t n_vertices)
: mgr_(mgr), n_(n_vertices), verts_(mgr_, n_vertices) {}

/**
 * build_from_edges
 * Phase 1: degree count (CPU-only).
 * Phase 2: deref header u; set degree; allocate contiguous Vid[degree].
 * Phase 3: fill each Vid[] by deref’ing the neighbor array and appending v.
 * Scope discipline: short-lived scopes bound pin time; no nesting.
 */
void GraphAdj::build_from_edges(const std::vector<std::pair<Vid,Vid>>& edges) {
  // Phase 1: degree count
  std::vector<uint32_t> deg(n_, 0);
  for (auto [u, v] : edges) {
    assert(u < n_ && v < n_);
    ++deg[u];
  }

  // Phase 2: init headers + allocate neighbor arrays
  {
    DerefScope scope;
    for (uint64_t u = 0; u < n_; ++u) {
      Vertex& vh = deref_vertex(scope, u);  // pins header u during this iteration
      vh.degree = deg[u];
      if (vh.degree == 0) {
        vh.nbrs = GenericUniquePtr{};
        continue;
      }
      vh.nbrs = mgr_->allocate_generic_unique_ptr(
          kVanillaPtrDSID, static_cast<uint64_t>(vh.degree) * sizeof(Vid));
    }
    // scope ends → headers unpinned; evictor free to act if needed
  }

  // Phase 3: fill neighbor arrays
  std::vector<uint32_t> cur(n_, 0);
  {
    DerefScope scope;
    for (auto [u, v] : edges) {
      const Vertex& vh = const_deref_vertex(scope, u);
      if (vh.degree == 0) continue;

      // Fault-in & pin u’s neighbor array, then append v
      auto* base = static_cast<uint8_t*>(vh.nbrs.deref(scope));
      auto* arr  = reinterpret_cast<Vid*>(base);
      arr[cur[u]++] = v;
    }
    // scope ends → arrays unpinned; eligible for eviction
  }
}

/**
 * neighbors(u, scope)
 * Return a view into u’s adjacency list. Pointer valid until scope ends.
 * Aligns with AIFM: object fetched on first touch; local loads thereafter.
 */
GraphAdj::NeighborView GraphAdj::neighbors(uint64_t u, DerefScope& scope) const {
  const Vertex& vh = const_deref_vertex(scope, u); // pins header briefly
  if (vh.degree == 0 || !vh.nbrs.valid())
    return {nullptr, 0};
  const auto* base = static_cast<const uint8_t*>(vh.nbrs.deref(scope)); // pins array
  return { reinterpret_cast<const Vid*>(base), vh.degree };
}

/** degree(u, scope) — metadata-only: keeps header hot, no array touch. */
uint32_t GraphAdj::degree(uint64_t u, DerefScope& scope) const {
  return const_deref_vertex(scope, u).degree;
}

/** Prefetch a strided set of headers; useful for metadata look-ahead. */
void GraphAdj::prefetch_vertices(uint64_t start, uint64_t step, uint32_t num) {
  verts_.static_prefetch(/*start=*/start, /*step=*/step, /*num=*/num);
}

/* --------- private helpers: header deref under a scope --------- */

Vertex& GraphAdj::deref_vertex(const DerefScope& scope, uint64_t i) {
  GenericUniquePtr* slotp = verts_.slot(i);
  void* p = slotp->deref(scope);                     // runtime fetch+pin if needed
  return *reinterpret_cast<Vertex*>(p);
}

const Vertex& GraphAdj::const_deref_vertex(const DerefScope& scope, uint64_t i) const {
  const GenericUniquePtr* slotp = verts_.slot(i);
  const void* p = slotp->deref(scope);
  return *reinterpret_cast<const Vertex*>(p);
}

} // namespace far_memory
