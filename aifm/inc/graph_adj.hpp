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

#include <limits>


namespace far_memory {

using Vid = uint32_t;

// ------ Graph active component protocol on top of RemDevice -------

// New data structure type and instance ID for the graph active component.
// IMPORTANT: we'll use the same values on the server side.
static constexpr uint8_t kGraphAggDSType = 3;  // distinct from kVanillaPtrDSType
static constexpr uint8_t kGraphDSID      = 5;  // graph instance id for compute()

// Opcodes for graph-specific remote compute().
enum GraphOpcode : uint8_t {
  kGraphOpDegreeSum = 1,  // sum(degree(u)) over a frontier
};

// Request header for degree sum.
// Wire format for compute() input_buf:
//   [ GraphDegreeSumReqHdr ][ frontier_len * Vid ]
struct __attribute__((packed)) GraphDegreeSumReqHdr {
  uint32_t frontier_len;
};

static_assert(sizeof(GraphDegreeSumReqHdr) == 4,
              "Unexpected padding in GraphDegreeSumReqHdr");

// Parameters sent at construct() time to initialize the graph component.
struct __attribute__((packed)) GraphAggParams {
  uint64_t num_vertices;
};

static_assert(sizeof(GraphAggParams) == 8,
              "Unexpected padding in GraphAggParams");






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
  Vid inline_small[kInlineCap]{};      // small adjacency in-place
  mutable GenericUniquePtr tail;       // remaining neighbors (if any)
};

/* Array of headers (one far-mem object per vertex) */
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



  // inline GraphAdj(FarMemManager* mgr, uint64_t n_vertices,
  //                 const RemotingPolicy& policy)
  //     : mgr_(mgr), policy_(policy), verts_(mgr, n_vertices) {
  //   // Initialize the graph active component on the remote memory server.
  //   GraphAggParams params;
  //   params.num_vertices = n_vertices;

  //   FarMemDevice *dev = mgr_->get_device();
  //   assert(dev != nullptr);

  //   dev->construct(/*ds_type=*/kGraphAggDSType,
  //                  /*ds_id=*/kGraphDSID,
  //                  /*param_len=*/sizeof(params),
  //                  /*params=*/reinterpret_cast<uint8_t*>(&params));
  // }




  inline uint64_t num_vertices() const { return verts_.size(); }

  /* Build from (u,v) edges. */
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
        vh = VertexHdr{};                 // value-init (safe zeroing)
        vh.degree = deg[u];
        if (vh.degree == 0) continue;

        const uint16_t wish = policy_.inline_capacity((Vid)u, vh.degree);
        vh.inline_len = std::min<uint16_t>(wish, VertexHdr::kInlineCap);

        const uint32_t tail_deg = vh.degree - vh.inline_len;
        if (tail_deg > 0) {
          // NOTE: 16-bit size to match vanilla pointer expectations.
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

  inline void prefetch_headers_span(uint64_t start, uint32_t num) {
  if (num == 0) return;
  verts_.static_prefetch(/*start=*/start, /*step=*/1, /*num=*/num);
}

  // Compute sum of degrees (or vertex IDs for now) for a frontier via
  // a remote active component.
  //
  // NOTE: This lazily constructs the remote graph aggregation component
  //       the first time it’s called, and reuses it thereafter. That way
  //       tests that *don’t* use remote compute (like test_graph_prefetch)
  //       never touch DSID kGraphDSID and don’t trip Server::construct().
  inline uint64_t remote_degree_sum(const std::vector<Vid>& frontier) const {
    const uint32_t n = static_cast<uint32_t>(frontier.size());
    if (n == 0) return 0;

    // --- One-time registration of the graph active component on this device ---
    {
      static bool registered = false;
      if (!registered) {
        FarMemDevice* dev = mgr_->get_device();

        // Must match the struct in server_graph.hpp
        struct __attribute__((packed)) GraphAggParams {
          uint64_t num_vertices;
        } params;

        params.num_vertices = verts_.size();

        dev->construct(/*ds_type=*/kGraphAggDSType,
                       /*ds_id=*/kGraphDSID,
                       /*param_len=*/sizeof(params),
                       /*params=*/reinterpret_cast<uint8_t*>(&params));
        registered = true;
      }
    }

    // --- Build input buffer: [frontier_len][frontier...] ---
    std::vector<uint8_t> in;
    in.resize(sizeof(uint32_t) + n * sizeof(Vid));
    uint8_t* p = in.data();
    std::memcpy(p, &n, sizeof(uint32_t));
    p += sizeof(uint32_t);
    std::memcpy(p, frontier.data(), n * sizeof(Vid));

    uint16_t in_len  = static_cast<uint16_t>(in.size());
    uint16_t out_len = 0;
    uint8_t out_buf[sizeof(uint64_t)] = {0};

    FarMemDevice* dev = mgr_->get_device();
    dev->compute(/*ds_id=*/kGraphDSID,
                 /*opcode=*/kGraphOpDegreeSum,
                 /*input_len=*/in_len,
                 /*input_buf=*/in.data(),
                 /*output_len=*/&out_len,
                 /*output_buf=*/out_buf);

    assert(out_len == sizeof(uint64_t));
    uint64_t total = 0;
    std::memcpy(&total, out_buf, sizeof(uint64_t));
    return total;
  }






  /* Backwards-compatible struct view for callers that used neighbors(). */
  struct NeighborView {
    const Vid* inline_ptr{nullptr};
    uint32_t   inline_len{0};
    const Vid* tail_ptr{nullptr};
    uint32_t   tail_len{0};
  };

  /* Optional view API (kept for compatibility with older tests). */
  inline NeighborView neighbors(uint64_t u, DerefScope& scope) const {
    const auto &vh = const_deref_vertex(scope, u);
    NeighborView nv;
    nv.inline_ptr = vh.inline_small;
    nv.inline_len = vh.inline_len;

    const uint32_t tail_deg =
        (vh.degree > vh.inline_len) ? (vh.degree - vh.inline_len) : 0;
    if (tail_deg > 0) {
      const void *tail_base = vh.tail.deref(scope);   // const-safe read
      nv.tail_ptr = reinterpret_cast<const Vid*>(tail_base);
      nv.tail_len = tail_deg;
    }
    return nv;
  }
  // --- add inside public: of GraphAdj (near other helpers) ---
// inside class GraphAdj (public:)
inline void warm_tail_prefix(uint64_t u, uint32_t k, DerefScope& scope) const {
  const auto &vh = const_deref_vertex(scope, u);
  const uint32_t tail_deg =
      (vh.degree > vh.inline_len) ? (vh.degree - vh.inline_len) : 0;
  if (tail_deg == 0 || k == 0) return;

  const void *base = vh.tail.deref(scope);
  const Vid *tp = reinterpret_cast<const Vid*>(base);
  const uint32_t warm = (k < tail_deg) ? k : tail_deg;

  // bounded touches to nudge fetch; volatile prevents the compiler from eliding
  volatile Vid sink = 0;
  for (uint32_t i = 0; i < warm; ++i) sink ^= tp[i];
  (void)sink;
}

inline void hint_tail_present(uint64_t u) const {
  DerefScope s;
  const auto &vh = const_deref_vertex(s, u);
  if (vh.degree > vh.inline_len) {
    (void)vh.tail.deref(s);   // map the tail once, then drop the scope
  }
}




  /* Header info for stats / warm touches */
  struct HeaderInfo { uint32_t degree; uint16_t inline_len; };
  inline HeaderInfo header_info(uint64_t u, DerefScope &scope) const {
    const auto &vh = const_deref_vertex(scope, u);
    return HeaderInfo{vh.degree, vh.inline_len};
  }

  inline uint32_t degree(uint64_t u, DerefScope& scope) const {
    return const_deref_vertex(scope, u).degree;
  }

  /* Safe neighbor iterator (works for any placement). */
  template <typename F>
  inline void for_each_neighbor(uint64_t u, DerefScope& scope, F&& fn) const {
    const auto &vh = const_deref_vertex(scope, u);

    // inline chunk
    for (uint32_t i = 0; i < vh.inline_len; ++i)
      fn(vh.inline_small[i]);

    // tail chunk (if any)
    const uint32_t tail_deg =
        (vh.degree > vh.inline_len) ? (vh.degree - vh.inline_len) : 0;
    if (tail_deg > 0) {
      const void *base = vh.tail.deref(scope);
      const Vid *tp = reinterpret_cast<const Vid*>(base);
      for (uint32_t i = 0; i < tail_deg; ++i)
        fn(tp[i]);
    }
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