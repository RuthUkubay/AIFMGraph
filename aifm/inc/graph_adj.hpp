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
  Vid inline_small[kInlineCap]{}; // tiny adjacency in the header

  // Tail is split into multiple far-memory chunks so we never exceed
  // a small per-object size. Each chunk records its size in bytes.
  struct TailChunk {
    mutable GenericUniquePtr ptr;  // <-- mutable so const methods can deref()
    uint32_t bytes{0};             // multiple of sizeof(Vid)
  };
  mutable std::vector<TailChunk> tail_chunks;
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

  /* Build from directed edge list. Robust via chunked tails. */
  inline void build_from_edges(const std::vector<std::pair<Vid, Vid>>& edges) {
    const uint64_t N = verts_.size();

    // 1) degree count
    std::vector<uint32_t> deg(N, 0);
    for (auto [u, v] : edges) { (void)v; assert(u < N); ++deg[u]; }

    // 2a) write headers & allocate tails (chunked)
    {
      DerefScope scope;
      for (uint64_t u = 0; u < N; ++u) {
        auto &vh = deref_vertex(scope, u);
        vh = VertexHdr{};           // value-init (clears tail_chunks etc.)
        vh.degree = deg[u];

        if (vh.degree == 0) continue;

        const uint16_t wish = policy_.inline_capacity((Vid)u, vh.degree);
        vh.inline_len = std::min<uint16_t>(wish, VertexHdr::kInlineCap);

        const uint32_t tail_deg = vh.degree - vh.inline_len;
        vh.tail_chunks.clear();
        if (tail_deg > 0) {
          // Keep chunk sizes modest (e.g., 60 KiB) to be safe.
          static constexpr uint32_t kMaxChunkBytes = 60 * 1024;
          uint32_t bytes_total = tail_deg * sizeof(Vid);
          while (bytes_total > 0) {
            const uint32_t this_bytes =
                (bytes_total > kMaxChunkBytes) ? kMaxChunkBytes : bytes_total;
            VertexHdr::TailChunk c;
            c.bytes = this_bytes;
            c.ptr = mgr_->allocate_generic_unique_ptr(kVanillaPtrDSID,
                                                      static_cast<uint16_t>(c.bytes));
            vh.tail_chunks.emplace_back(std::move(c));
            bytes_total -= this_bytes;
          }
        }
      }
    }

    // 2b) fill inline then tail (across chunks)
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
          write_tail_vid(vh, scope, tail_idx, v);
          ++cur[u];
        }
      }
    }
  }

  /* Header info for stats */
  struct HeaderInfo { uint32_t degree; uint16_t inline_len; };
  inline HeaderInfo header_info(uint64_t u, DerefScope &scope) const {
    const auto &vh = const_deref_vertex(scope, u);
    return HeaderInfo{vh.degree, vh.inline_len};
  }

  inline uint32_t degree(uint64_t u, DerefScope& scope) const {
    return const_deref_vertex(scope, u).degree;
  }

  /* ===== Safe iterator over all neighbors (inline + all tail chunks) ===== */
  template <typename F>
  inline void for_each_neighbor(uint64_t u, DerefScope& scope, F&& f) const {
    const auto &vh = const_deref_vertex(scope, u);
    // inline part
    for (uint32_t i = 0; i < vh.inline_len; ++i) {
      f(vh.inline_small[i]);
    }
    // tail chunks
    const uint32_t tail_deg =
        (vh.degree > vh.inline_len) ? (vh.degree - vh.inline_len) : 0;
    if (tail_deg == 0) return;

    uint32_t emitted = 0;
    for (const auto& c : vh.tail_chunks) {
      if (emitted >= tail_deg) break;
      const void* raw = c.ptr.deref(scope); // OK: ptr is mutable; deref() returns const void*
      const auto* base = static_cast<const uint8_t*>(raw);
      const Vid*   vptr = reinterpret_cast<const Vid*>(base);
      const uint32_t entries = c.bytes / sizeof(Vid);
      const uint32_t todo = std::min(entries, tail_deg - emitted);
      for (uint32_t i = 0; i < todo; ++i) f(vptr[i]);
      emitted += todo;
    }
  }

  /* ===== Back-compat: neighbors() view used by older tests =====
     Returns inline span and the FIRST tail chunk (if any) as a span.
     Tests that just "touch a few entries" will work with this.           */
  struct NeighborView {
    const Vid* inline_ptr{nullptr};
    uint32_t   inline_len{0};
    const Vid* tail_ptr{nullptr};
    uint32_t   tail_len{0}; // length of the first tail chunk only
  };

  inline NeighborView neighbors(uint64_t u, DerefScope& scope) const {
    const auto &vh = const_deref_vertex(scope, u);
    NeighborView nv;
    nv.inline_ptr = vh.inline_small;
    nv.inline_len = vh.inline_len;

    const uint32_t tail_deg =
        (vh.degree > vh.inline_len) ? (vh.degree - vh.inline_len) : 0;
    if (tail_deg > 0 && !vh.tail_chunks.empty()) {
      const auto &c0 = vh.tail_chunks[0];
      const void* raw = c0.ptr.deref(scope);
      const auto* base = static_cast<const uint8_t*>(raw);
      nv.tail_ptr = reinterpret_cast<const Vid*>(base);
      const uint32_t entries = c0.bytes / sizeof(Vid);
      nv.tail_len = std::min(entries, tail_deg);
    }
    return nv;
  }

  /* Prefetch helpers (headers only; tails on demand) */
  inline void prefetch_headers_span(uint64_t start, uint32_t num) {
    if (num == 0) return;
    verts_.static_prefetch(start, /*step=*/1, num);
  }
  inline void enable_header_static_prefetch(uint32_t distance) {
    verts_.static_prefetch(/*start=*/0, /*step=*/1, /*num=*/distance);
  }
  inline void disable_header_prefetch() {
    verts_.disable_prefetch();
  }
  inline void hint_tail_present(uint64_t u) {
    DerefScope s;
    const auto &vh = const_deref_vertex(s, u);
    if (!vh.tail_chunks.empty()) (void)vh.tail_chunks[0].ptr.deref(s);
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

  // map logical tail index -> chunk,offset and write
  inline void write_tail_vid(VertexHdr& vh, const DerefScope& scope,
                             uint32_t tail_idx, Vid v) {
    uint32_t idx = tail_idx;
    for (auto &c : vh.tail_chunks) {
      const uint32_t entries = c.bytes / sizeof(Vid);
      if (idx < entries) {
        void* raw = c.ptr.deref_mut(scope);
        auto* base = static_cast<uint8_t*>(raw);
        reinterpret_cast<Vid*>(base)[idx] = v;
        return;
      }
      idx -= entries;
    }
    assert(false && "tail_idx out of range");
  }
};

} // namespace far_memory
