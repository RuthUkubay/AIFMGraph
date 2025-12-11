#pragma once

extern "C" {
#include <base/assert.h>
#include <base/compiler.h>
#include <base/stddef.h>
}

#include "server.hpp"

#include <cstdint>
#include <cstring>

namespace far_memory {

// IMPORTANT: these must match the values in your GraphAdj file.
static constexpr uint8_t kGraphAggDSType = 2;  // same as in GraphAdj
static constexpr uint8_t kGraphDSID      = 2;  // same as in GraphAdj

// Opcodes (duplicate here so server can decode them).
enum GraphOpcode : uint8_t {
  kGraphOpDegreeSum = 1,
};

// Request / param structs (keep layout identical to the client side).
struct __attribute__((packed)) GraphDegreeSumReqHdr {
  uint32_t frontier_len;
};

struct __attribute__((packed)) GraphAggParams {
  uint64_t num_vertices;
};

class ServerGraphAgg : public ServerDS {
public:
  ServerGraphAgg(uint8_t param_len, uint8_t *params) {
    assert(param_len == sizeof(GraphAggParams));
    GraphAggParams p;
    std::memcpy(&p, params, sizeof(p));
    num_vertices_ = p.num_vertices;
  }

  // We don't use these for the graph active component (all graph data is
  // still stored via the vanilla pointer DS), so they can just BUG().
  void read_object(uint8_t, const uint8_t*, uint16_t*, uint8_t*) override {
    BUG();
  }

  void write_object(uint8_t, const uint8_t*, uint16_t, const uint8_t*) override {
    BUG();
  }

  bool remove_object(uint8_t, const uint8_t*) override {
    BUG();
    return false;
  }

  void compute(uint8_t opcode, uint16_t input_len,
               const uint8_t *input_buf, uint16_t *output_len,
               uint8_t *output_buf) override {
    switch (opcode) {
      case kGraphOpDegreeSum: {
        assert(input_len >= sizeof(GraphDegreeSumReqHdr));
        auto *hdr = reinterpret_cast<const GraphDegreeSumReqHdr*>(input_buf);
        const uint32_t frontier_len = hdr->frontier_len;

        const uint8_t *p = input_buf + sizeof(GraphDegreeSumReqHdr);
        const size_t remaining = input_len - sizeof(GraphDegreeSumReqHdr);
        assert(remaining >= static_cast<size_t>(frontier_len) * sizeof(uint32_t));

        const uint32_t *frontier =
            reinterpret_cast<const uint32_t*>(p);

        uint64_t total_deg = 0;

        // TODO: hook this up to real graph degrees from far memory.
        //
        // For now, you can use this as a pipeline sanity check
        // by temporarily treating frontier[i] as the "degree" itself:
        //
          for (uint32_t i = 0; i < frontier_len; ++i)
            total_deg += frontier[i];
        
        // Then later replace with a real degree lookup.

        // for (uint32_t i = 0; i < frontier_len; ++i) {
        //   (void)frontier[i];  // replace with real degree lookup later
        // }

        std::memcpy(output_buf, &total_deg, sizeof(total_deg));
        *output_len = sizeof(total_deg);
        break;
      }

      default:
        // Unknown graph opcode.
        *output_len = 0;
        break;
    }
  }

private:
  uint64_t num_vertices_ = 0;
};

class ServerGraphAggFactory : public ServerDSFactory {
public:
  ServerDS *build(uint8_t param_len, uint8_t *params) override {
    return new ServerGraphAgg(param_len, params);
  }
};

} // namespace far_memory
