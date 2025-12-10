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

/* Policy: how many neighbors to keep inline per vertex */
struct RemotingPolicy {
  virtual ~RemotingPolicy() = default;
  virtual uint16_t inline_capacity(Vid u, uint32_t degree) const = 0;
};

/* Per-vertex header — POD only (safe for far-memory moves) */
struct VertexHdr {
  uint32_t degree{0};
  uint16_t inline_len{0};

  static constexpr uint16_t kInlineCap = 8;
  Vid inline_small[kInlineCap]{};   // small adjacency in the header

  struct TailChunk {
    mutable GenericUniquePtr ptr;   // deref() is non-const, so handle must be mutable
    uint16_t bytes{0};              // <= 60 KiB, multiple of sizeof(Vid)
  };
  static constexpr uint16_t kMaxTailChunks = 8;
  uint16_t num_chunks{0};
  TailChunk chunks[kMaxTailChunks]{};
};

/* Array of headers (one far object per vertex) */
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

/* Graph: adjacency lists over AIFM (pointer-chasing layout) */
class GraphAdj {
public:
  inline GraphAdj(FarMemManager* mgr, uint64_t n_vertices,
                  const RemotingPolicy& policy)
      : mgr_(mgr), policy_(policy), verts_(mgr, n_vertices) {}

  inline uint64_t num_vertices() const { return verts_.size(); }

  /* Build from directed edges; tail is chunked (no std::vector in far mem) */
  inline void build_from_edges(const std::vector<std::pair<Vid, Vid>>& edges) {
    const uint64_t N = verts_.size();

    // 1) degree count
    std::vector<uint32_t> deg(N, 0);
    for (auto [u, v] : edges) { (void)v; assert(u < N); ++deg[u]; }

    // 2a) headers & tail chunks
    {
      DerefScope scope;
      for (uint64_t u = 0; u < N; ++u) {
        auto &vh = deref_vertex(scope, u);
        vh = VertexHdr{};                       // value-init (POD safe)
        vh.degree = deg[u];
        if (vh.degree == 0) continue;

        const uint16_t wish = policy_.inline_capacity((Vid)u, vh.degree);
        vh.inline_len = std::min<uint16_t>(wish, VertexHdr::kInlineCap);

        const uint32_t tail_deg = vh.degree - vh.inline_len;
        vh.num_chunks = 0;

        if (tail_deg > 0) {
          static constexpr uint32_t kMaxChunkBytes = 60 * 1024; // 60 KiB
          uint32_t bytes_left = tail_deg * sizeof(Vid);
          while (bytes_left > 0) {
            assert(vh.num_chunks < VertexHdr::kMaxTailChunks &&
                   "degree requires more tail chunks than kMaxTailChunks");
            const uint32_t take = std::min<uint32_t>(bytes_left, kMaxChunkBytes);
            auto &c = vh.chunks[vh.num_chunks++];
            c.bytes = static_cast<uint16_t>(take);
            c.ptr   = mgr_->allocate_generic_unique_ptr(kVanillaPtrDSID, c.bytes);
            bytes_left -= take;
          }
        }
      }
    }

    // 2b) fill inline then tail (mapped chunk by chunk)
    std::vector<uint32_t> cur(N, 0);
    {
      DerefScope scope;
      for (auto [u, v] : edges) {
        auto &vh = deref_vertex(scope, u);
        if (vh.degree == 0) continue;

        if (cur[u] < vh.inline_len) {
          vh.inline_small[cur[u]++] = v;
        } else {
          write_tail_vid(vh, scope, cur[u] - vh.inline_len, v);
          ++cur[u];
        }
      }
    }
  }

  /* Header info (for stats) */
  struct HeaderInfo { uint32_t degree; uint16_t inline_len; };
  inline HeaderInfo header_info(uint64_t u, DerefScope &scope) const {
    const auto &vh = const_deref_vertex(scope, u);
    return HeaderInfo{vh.degree, vh.inline_len};
  }

  inline uint32_t degree(uint64_t u, DerefScope& scope) const {
    return const_deref_vertex(scope, u).degree;
  }

  /* Iterate all neighbors (header + all tail chunks) */
  template <typename F>
  inline void for_each_neighbor(uint64_t u, DerefScope& scope, F&& f) const {
    const auto &vh = const_deref_vertex(scope, u);

    for (uint32_t i = 0; i < vh.inline_len; ++i) f(vh.inline_small[i]);

    const uint32_t tail_deg =
        (vh.degree > vh.inline_len) ? (vh.degree - vh.inline_len) : 0;
    if (tail_deg == 0) return;

    uint32_t seen = 0;
    for (uint16_t k = 0; k < vh.num_chunks && seen < tail_deg; ++k) {
      const auto &c = vh.chunks[k];
      const void* raw  = c.ptr.deref(scope);              // const deref
      const auto* base = static_cast<const uint8_t*>(raw);
      const Vid* vptr  = reinterpret_cast<const Vid*>(base);
      const uint32_t entries = c.bytes / sizeof(Vid);
      const uint32_t todo = std::min(entries, tail_deg - seen);
      for (uint32_t i = 0; i < todo; ++i) f(vptr[i]);
      seen += todo;
    }
  }

  /* Prefetch helpers (headers only) */
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

  inline void write_tail_vid(VertexHdr& vh, const DerefScope& scope,
                             uint32_t tail_idx, Vid v) {
    uint32_t idx = tail_idx;
    for (uint16_t k = 0; k < vh.num_chunks; ++k) {
      auto &c = vh.chunks[k];
      const uint32_t entries = c.bytes / sizeof(Vid);
      if (idx < entries) {
        void* raw  = c.ptr.deref_mut(scope);
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
