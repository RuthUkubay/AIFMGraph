// inc/graph_adj.hpp
#pragma once

extern "C" {
#include <runtime/thread.h>
}

#include "deref_scope.hpp"
#include "pointer.hpp"
#include "manager.hpp"
#include "array.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

namespace far_memory {

using Vid = uint32_t;

/* Placement policy: how many neighbors to keep inline per vertex */
struct RemotingPolicy {
  virtual ~RemotingPolicy() = default;
  virtual uint16_t inline_capacity(Vid u, uint32_t degree) const = 0;
};

/* Per-vertex header */
struct VertexHdr {
  uint32_t degree{0};
  uint16_t inline_len{0};

  static constexpr uint16_t kInlineCap = 8;
  Vid inline_small[kInlineCap]{};     // tiny adjacency in the header
  mutable GenericUniquePtr tail;      // remaining neighbors, if any
};

/* Array of headers (one object per vertex) */
class VertexArray : public GenericArray {
public:
  inline VertexArray(FarMemManager* mgr, uint64_t n_vertices)
      : GenericArray(mgr, sizeof(VertexHdr), n_vertices) {}

  inline GenericUniquePtr* slot(uint64_t i) { return at(false, i); }
  inline GenericUniquePtr* slot(uint64_t i) const {
    return const_cast<VertexArray*>(this)->at(false, i);
  }
  inline uint64_t size() const { return kNumItems_; }
};

/* Graph: adjacency lists over AIFM */
class GraphAdj {
public:
  inline GraphAdj(FarMemManager* mgr, uint64_t n_vertices,
                  const RemotingPolicy& policy)
      : mgr_(mgr), policy_(policy), verts_(mgr, n_vertices) {}

  inline uint64_t num_vertices() const { return verts_.size(); }

  inline void build_from_edges(const std::vector<std::pair<Vid, Vid>>& edges) {
    const uint64_t N = verts_.size();

    // 1) degree count
    std::vector<uint32_t> deg(N, 0);
    for (auto [u, v] : edges) { (void)v; assert(u < N); ++deg[u]; }

    // 2a) write headers & allocate tails
    {
      DerefScope scope;
      for (uint64_t u = 0; u < N; ++u) {
        auto &vh = deref_vertex(scope, u);

        // full zero-init (safe for POD; VertexHdr is POD here)
        vh = VertexHdr{};

        vh.degree = deg[u];
        if (vh.degree == 0) continue;

        const uint16_t wish = policy_.inline_capacity((Vid)u, vh.degree);
        vh.inline_len = std::min<uint16_t>(wish, VertexHdr::kInlineCap);

        const uint32_t tail_deg = vh.degree - vh.inline_len;
        if (tail_deg > 0) {
          const uint16_t bytes = static_cast<uint16_t>(tail_deg * sizeof(Vid));
          vh.tail = mgr_->allocate_generic_unique_ptr(kVanillaPtrDSID, bytes);
        } else {
          vh.tail = GenericUniquePtr{};
        }
      }
    }

    // 2b) fill inline then tail
    std::vector<uint32_t> cur(N, 0);
    {
      DerefScope scope;
      for (auto [u, v] : edges) {
        auto &vh = deref_vertex(scope, u);
        if (vh.degree == 0) continue;

        if (cur[u] < vh.inline_len) {
          vh.inline_small[cur[u]++] = v;
        } else {
          const uint32_t tail_idx = cur[u] - vh.inline_len;
          const uint32_t tail_deg = vh.degree - vh.inline_len;
          if (tail_deg > 0) {
            auto *base = static_cast<uint8_t*>(vh.tail.deref_mut(scope));
            reinterpret_cast<Vid*>(base)[tail_idx] = v;
          }
          ++cur[u];
        }
      }
    }
  }

  struct NeighborView {
    const Vid* inline_ptr{nullptr};
    uint32_t   inline_len{0};
    const Vid* tail_ptr{nullptr};
    uint32_t   tail_len{0};
  };

  inline NeighborView neighbors(uint64_t u, DerefScope& scope) const {
    const auto &vh = const_deref_vertex(scope, u);
    NeighborView nv;
    nv.inline_ptr = vh.inline_small;
    nv.inline_len = vh.inline_len;

    const uint32_t tail_deg =
        (vh.degree > vh.inline_len) ? (vh.degree - vh.inline_len) : 0;
    if (tail_deg > 0) {
      const void *tail_base = vh.tail.deref(scope);   // only if tail exists
      nv.tail_ptr = reinterpret_cast<const Vid*>(tail_base);
      nv.tail_len = tail_deg;
    }
    return nv;
  }

  struct HeaderInfo { uint32_t degree; uint16_t inline_len; };
  inline HeaderInfo header_info(uint64_t u, DerefScope &scope) const {
    const auto &vh = const_deref_vertex(scope, u);
    return HeaderInfo{vh.degree, vh.inline_len};
  }

  inline uint32_t degree(uint64_t u, DerefScope& scope) const {
    return const_deref_vertex(scope, u).degree;
  }

  /* Prefetch helpers */
  inline void prefetch_headers_span(uint64_t start, uint32_t num) {
    if (num == 0) return;
    verts_.static_prefetch(start, /*step=*/1, num);
  }

  inline void enable_header_static_prefetch(uint32_t distance) {
    // stride-1 scan; ‘distance’ lookahead
    verts_.static_prefetch(/*start=*/0, /*step=*/1, /*num=*/distance);
  }

  inline void disable_header_prefetch() {
    // cancel/disable any static prefetch configured on this array
    verts_.disable_prefetch();
  }

  inline void hint_tail_present(uint64_t u) {
    DerefScope s;
    const auto &vh = const_deref_vertex(s, u);
    if (vh.degree > vh.inline_len) (void)vh.tail.deref(s);
  }

private:
  FarMemManager* mgr_{nullptr};
  const RemotingPolicy& policy_;
  VertexArray    verts_;

  inline VertexHdr& deref_vertex(const DerefScope& scope, uint64_t i) {
    void* p = verts_.slot(i)->deref_mut(scope);
    return *reinterpret_cast<VertexHdr*>(p);
  }
  inline const VertexHdr& const_deref_vertex(const DerefScope& scope, uint64_t i) const {
    auto* h = verts_.slot(i);
    const void* p = h->deref(scope);
    return *reinterpret_cast<const VertexHdr*>(p);
  }
};

} // namespace far_memory
