#pragma once

#include "common.hpp"

#include <array>
#include <cstdint>
#include <deque>

namespace adaptive_batcher {

// ─── FeatureExtractor ────────────────────────────────────────────────────────
// Maintains rolling state for all 10 market microstructure features.
// Updated O(1) per event. snapshot() returns normalized, padded vector.
//
// Feature layout in MarketFeatures::v[]:
//   [0] spread_bps          — bid-ask spread in basis points
//   [1] volatility          — Welford online variance of mid-price returns
//   [2] message_rate        — messages per second (1s sliding window)
//   [3] time_of_day         — minutes since 9:30am (normalized 0..1 over 6.5h)
//   [4] queue_depth         — total pending requests (set externally)
//   [5] price_momentum      — (mid_now - mid_5s_ago) / mid_5s_ago
//   [6] order_imbalance     — (bid_qty - ask_qty) / (bid_qty + ask_qty)
//   [7] cancel_ratio        — cancels / (adds + cancels) in last 1000 msgs
//   [8] batch_success_rate  — EMA of (actual_window ≤ configured_window)
//   [9] time_since_last_batch — ns since last dispatch, normalized [0..1]
//   [10..15] zero padding

class FeatureExtractor {
public:
    FeatureExtractor();

    // Update rolling state from a book event. O(1).
    void on_market_event(const BookState& state) noexcept;

    // Called after each batch dispatch to update batch_success_rate and
    // time_since_last_batch baseline.
    void on_batch_dispatched(uint32_t window_us_configured,
                             uint32_t window_us_actual) noexcept;

    // Set external queue depth (sum across all batch buffers).
    void set_queue_depth(uint32_t depth) noexcept { queue_depth_ = depth; }

    // Return normalized, padded feature vector (slots [10..15] = 0).
    MarketFeatures snapshot() const noexcept;

    // For testing: direct access to raw (un-normalized) features
    float raw_spread_bps()           const noexcept;
    float raw_volatility()           const noexcept;
    float raw_message_rate()         const noexcept;
    float raw_order_imbalance()      const noexcept;
    float raw_cancel_ratio()         const noexcept;
    float raw_price_momentum()       const noexcept;
    float raw_time_since_last_batch_ms() const noexcept;

private:
    // ── spread_bps ─────────────────────────────────────────────────────────
    int64_t last_bid_ticks_ = 0;
    int64_t last_ask_ticks_ = 0;
    int64_t last_tick_size_ = 1;

    // ── volatility (Welford online variance of log returns) ────────────────
    double  vol_mean_   = 0.0;
    double  vol_M2_     = 0.0;
    int64_t vol_count_  = 0;
    double  last_mid_   = 0.0;

    // ── message_rate (sliding 1s window, O(1) amortized) ────────────────────
    static constexpr int kMsgRateBuf = 4096;
    std::array<int64_t, kMsgRateBuf> msg_times_{};
    int     msg_head_  = 0;  // next write slot
    int     msg_tail_  = 0;  // oldest in-window slot
    int     msg_window_count_ = 0;  // entries within 1s — O(1) to read

    // ── time_of_day ────────────────────────────────────────────────────────
    int64_t last_timestamp_ns_ = 0;

    // ── queue_depth (set externally) ────────────────────────────────────────
    uint32_t queue_depth_ = 0;

    // ── price_momentum (5s lookback, O(1) amortized) ─────────────────────────
    static constexpr int kMomBuf = 512;
    std::array<std::pair<int64_t, double>, kMomBuf> mid_history_{};
    int    mom_head_  = 0;  // next write slot
    int    mom_tail_  = 0;  // oldest in-window slot
    int    mom_count_ = 0;  // total entries in buffer
    double mom_ref_mid_ = 0.0;  // oldest mid within 5s window — O(1) to read

    // ── order_imbalance ─────────────────────────────────────────────────────
    int64_t last_bid_qty_ = 0;
    int64_t last_ask_qty_ = 0;

    // ── cancel_ratio (last 1000 messages) ───────────────────────────────────
    static constexpr int kCancelBuf = 1024;  // power-of-2 for masking
    std::array<char, kCancelBuf> cancel_history_{};
    int    cancel_head_  = 0;
    int    cancel_count_ = 0;
    int    cancel_adds_  = 0;
    int    cancel_cancels_ = 0;

    // ── batch_success_rate (EMA, α=0.1) ────────────────────────────────────
    float batch_success_rate_ = 1.0f;  // start optimistic
    static constexpr float kBatchAlpha = 0.1f;

    // ── time_since_last_batch ───────────────────────────────────────────────
    int64_t last_batch_ns_ = 0;
};

} // namespace adaptive_batcher
