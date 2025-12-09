#pragma once

// Scope status is managed by Shenango + AIFM.
// Scope pins objects.
// Evictor must skip pinned objects;
extern "C" {
#include <runtime/thread.h>
}

#include "deref_scope.hpp"   // Status {OutofScope, InScopeV0, InScopeV1, GC}
#include "pointer.hpp"       // GenericUniquePtr (remoteable object handle)
#include "prefetcher.hpp"    // For optional header prefetch
#include "manager.hpp"       // FarMemManager: alloc/fetch/evict orchestrator

#include <cstdint>
#include <utility>
#include <vector>
#include <cassert>

namespace far_memory {

using Vid = uint32_t;

/**
 * Vertex
 * ------
 * Tiny, frequently-touched header:
 *  - degree: quick metadata (skip empty, bound loops) without faulting payload.
 *  - nbrs: remoteable pointer to contiguous Vid[] (size = degree * sizeof(Vid)).
 * Aligns with AIFM: fetch/evict whole objects (adj list) rather than pages.
 */
struct Vertex {
  uint32_t degree{0};
  GenericUniquePtr nbrs; // null when degree==0; deref(scope) → Vid* pinned for scope life.
};

/**
 * VertexArray
 * -----------
 * Runtime-sized array of per-vertex headers, mirroring GenericArray’s pattern:
 * each slot is a GenericUniquePtr to a Vertex. Prefetch headers without
 * touching large neighbor arrays; track hotness per-slot.
 */
class VertexArray : public GenericArray {
public:
  VertexArray(FarMemManager* mgr, uint64_t n_vertices);
  inline GenericUniquePtr* slot(uint64_t i);
  inline const GenericUniquePtr* slot(uint64_t i) const;
};

/**
 * GraphAdj
 * --------
 * Minimal adjacency-list graph over far memory.
 * Design:
 *  - Headers stay small & hot; adjacency arrays fault in on demand.
 *  - All data access goes through DerefScope to pin during use.
 *  - No nested scopes (paper constraint).
 */
class GraphAdj {
public:
  GraphAdj(FarMemManager* mgr, uint64_t n_vertices);

  uint64_t num_vertices() const { return n_; }

  // Build from directed edge list (u,v), 0 ≤ u,v < n_
  void build_from_edges(const std::vector<std::pair<Vid,Vid>>& edges);

  struct NeighborView {
    const Vid* ptr{nullptr};
    uint32_t   len{0};
  };

  // Get neighbors(u) under caller-owned scope (pins until scope ends).
  NeighborView neighbors(uint64_t u, DerefScope& scope) const;

  // Degree without touching the big array (keeps header hot).
  uint32_t degree(uint64_t u, DerefScope& scope) const;

  // Optional: prefetch a pattern of vertex headers (not neighbor arrays).
  void prefetch_vertices(uint64_t start, uint64_t step, uint32_t num);

private:
  // Runtime wiring
  FarMemManager* mgr_{nullptr}; // orchestrates alloc / fetch / evict (user-space)
  uint64_t n_{0};               // vertex count (host-side)
  VertexArray verts_;           // runtime-sized header array (per-slot hotness/prefetch)

  // Helpers: consistent header deref under a scope (pins for scope life).
  // Vertex& deref_vertex(const DerefScope& scope, uint64_t i);
  // const Vertex& const_deref_vertex(const DerefScope& scope, uint64_t i) const;
};

} // namespace far_memory
