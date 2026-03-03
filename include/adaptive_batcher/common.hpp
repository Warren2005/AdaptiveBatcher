#pragma once

#include <cstdint>
#include <cstring>

namespace adaptive_batcher {

// ─── Action space ─────────────────────────────────────────────────────────────

enum class BatchAction : uint8_t {
    IMMEDIATE = 0,  //   0 μs — dispatch immediately
    FAST      = 1,  //  50 μs
    NORMAL    = 2,  // 100 μs
    SLOW      = 3,  // 200 μs
    LAZY      = 4,  // 500 μs
};

constexpr uint32_t ACTION_WINDOW_US[5] = {0, 50, 100, 200, 500};
constexpr int NUM_ACTIONS  = 5;
constexpr int NUM_FEATURES = 10;  // padded to 16 in hot path

// ─── Feature vector ───────────────────────────────────────────────────────────
// 16 floats: [0..9] real features, [10..15] zero padding for AVX2 alignment.

struct alignas(64) MarketFeatures {
    float v[16];

    MarketFeatures() { std::memset(v, 0, sizeof(v)); }
};

static_assert(sizeof(MarketFeatures) == 64, "MarketFeatures must be one cache line");

// ─── Request — opaque order to be batched ────────────────────────────────────

struct Request {
    uint64_t id;          // caller-assigned request ID
    int64_t  timestamp_ns;
    uint64_t payload[6];  // opaque payload (symbol, qty, price, etc.)
};

// ─── BookState — snapshot from VectorBook for feature extraction ──────────────

struct BookState {
    int64_t  best_bid_ticks;   // 0 if no bid
    int64_t  best_ask_ticks;   // 0 if no ask
    int64_t  bid_quantity;     // quantity at BBO
    int64_t  ask_quantity;     // quantity at BBO
    int64_t  tick_size;        // ticks per slot
    int64_t  timestamp_ns;     // event timestamp
    char     msg_type;         // 'A'=add, 'X'=cancel, 'E'=execute, 'U'=replace
};

} // namespace adaptive_batcher
