#include "adaptive_batcher/market_features.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace adaptive_batcher {

// ─── Feature indices ─────────────────────────────────────────────────────────
static constexpr int F_SPREAD      = 0;
static constexpr int F_VOLATILITY  = 1;
static constexpr int F_MSG_RATE    = 2;
static constexpr int F_TIME_OF_DAY = 3;
static constexpr int F_QUEUE_DEPTH = 4;
static constexpr int F_MOMENTUM    = 5;
static constexpr int F_IMBALANCE   = 6;
static constexpr int F_CANCEL_RATE = 7;
static constexpr int F_BATCH_SR    = 8;
static constexpr int F_SINCE_BATCH = 9;

// Market open: 9:30am = 9.5 hours = 34,200 seconds after midnight
static constexpr int64_t kMarketOpenSec   = 34200LL;
static constexpr double  kSessionLenSec   = 23400.0;  // 6.5 hours
static constexpr int64_t kMomentumWindow  = 5'000'000'000LL;  // 5 seconds in ns
static constexpr int64_t kMsgRateWindow   = 1'000'000'000LL;  // 1 second in ns
static constexpr int64_t kMaxSinceBatch   = 1'000'000'000LL;  // 1 second cap

FeatureExtractor::FeatureExtractor() {
    msg_times_.fill(0);
    mid_history_.fill({0, 0.0});
    cancel_history_.fill(0);
}

void FeatureExtractor::on_market_event(const BookState& state) noexcept {
    int64_t now = state.timestamp_ns;
    last_timestamp_ns_ = now;

    // ── Update BBO ──────────────────────────────────────────────────────────
    if (state.best_bid_ticks > 0) last_bid_ticks_ = state.best_bid_ticks;
    if (state.best_ask_ticks > 0) last_ask_ticks_ = state.best_ask_ticks;
    if (state.tick_size > 0)      last_tick_size_ = state.tick_size;
    last_bid_qty_ = state.bid_quantity;
    last_ask_qty_ = state.ask_quantity;

    // ── Volatility: Welford online variance of log-return ───────────────────
    if (last_bid_ticks_ > 0 && last_ask_ticks_ > 0) {
        double mid = 0.5 * (last_bid_ticks_ + last_ask_ticks_);
        if (last_mid_ > 0.0 && mid > 0.0) {
            double ret = std::log(mid / last_mid_);
            ++vol_count_;
            double delta  = ret - vol_mean_;
            vol_mean_    += delta / vol_count_;
            double delta2 = ret - vol_mean_;
            vol_M2_      += delta * delta2;
        }
        last_mid_ = (last_mid_ > 0.0)
                    ? 0.5 * (last_bid_ticks_ + last_ask_ticks_)
                    : 0.5 * (last_bid_ticks_ + last_ask_ticks_);

        // ── Price momentum: sliding 5s window (O(1) amortized) ─────────────
        // Evict expired entries from tail
        {
            const int64_t cutoff = now - kMomentumWindow;
            while (mom_count_ > 0) {
                auto [old_ts, old_mid] = mid_history_[mom_tail_ & (kMomBuf - 1)];
                if (old_ts <= cutoff) { ++mom_tail_; --mom_count_; }
                else break;
            }
        }
        // Oldest in-window entry becomes the momentum reference
        if (mom_count_ > 0) {
            mom_ref_mid_ = mid_history_[mom_tail_ & (kMomBuf - 1)].second;
        }
        // Push new (timestamp, mid)
        mid_history_[mom_head_ & (kMomBuf - 1)] = {now, mid};
        ++mom_head_;
        ++mom_count_;
    }

    // ── Message rate: sliding 1s window (O(1) amortized) ────────────────────
    // Evict expired entries from tail
    {
        const int64_t cutoff = now - kMsgRateWindow;
        while (msg_window_count_ > 0) {
            int64_t oldest = msg_times_[msg_tail_ & (kMsgRateBuf - 1)];
            if (oldest <= cutoff) { ++msg_tail_; --msg_window_count_; }
            else break;
        }
    }
    // Push new timestamp
    msg_times_[msg_head_ & (kMsgRateBuf - 1)] = now;
    ++msg_head_;
    ++msg_window_count_;

    // ── Cancel ratio: last 1000 messages ────────────────────────────────────
    char msg_type = state.msg_type;
    bool is_cancel = (msg_type == 'X' || msg_type == 'D');
    bool is_add    = (msg_type == 'A' || msg_type == 'F');

    // Evict oldest entry
    int oldest_idx = cancel_head_ % kCancelBuf;
    char old_type = cancel_history_[oldest_idx];
    if (cancel_count_ >= kCancelBuf) {
        if (old_type == 'C') --cancel_cancels_;
        else if (old_type == 'A') --cancel_adds_;
    } else {
        ++cancel_count_;
    }

    cancel_history_[oldest_idx] = is_cancel ? 'C' : (is_add ? 'A' : 0);
    cancel_head_ = (cancel_head_ + 1) % kCancelBuf;

    if (is_cancel) ++cancel_cancels_;
    if (is_add)    ++cancel_adds_;
}

void FeatureExtractor::on_batch_dispatched(uint32_t window_us_configured,
                                            uint32_t window_us_actual) noexcept {
    float success = (window_us_actual <= window_us_configured) ? 1.0f : 0.0f;
    batch_success_rate_ = (1.0f - kBatchAlpha) * batch_success_rate_ +
                           kBatchAlpha * success;
    last_batch_ns_ = last_timestamp_ns_;
}

float FeatureExtractor::raw_spread_bps() const noexcept {
    if (last_bid_ticks_ <= 0 || last_ask_ticks_ <= 0) return 0.0f;
    double mid = 0.5 * (last_bid_ticks_ + last_ask_ticks_);
    if (mid <= 0.0) return 0.0f;
    double spread_ticks = static_cast<double>(last_ask_ticks_ - last_bid_ticks_);
    // spread in bps = (spread / mid) * 10000
    return static_cast<float>(spread_ticks / mid * 10000.0);
}

float FeatureExtractor::raw_volatility() const noexcept {
    if (vol_count_ < 2) return 0.0f;
    return static_cast<float>(std::sqrt(vol_M2_ / (vol_count_ - 1)));
}

float FeatureExtractor::raw_message_rate() const noexcept {
    return static_cast<float>(msg_window_count_);  // O(1): maintained incrementally
}

float FeatureExtractor::raw_order_imbalance() const noexcept {
    int64_t total = last_bid_qty_ + last_ask_qty_;
    if (total <= 0) return 0.0f;
    return static_cast<float>(last_bid_qty_ - last_ask_qty_) /
           static_cast<float>(total);
}

float FeatureExtractor::raw_cancel_ratio() const noexcept {
    int denom = cancel_adds_ + cancel_cancels_;
    if (denom <= 0) return 0.0f;
    return static_cast<float>(cancel_cancels_) / static_cast<float>(denom);
}

float FeatureExtractor::raw_price_momentum() const noexcept {
    // O(1): mom_ref_mid_ is the oldest mid within 5s, maintained incrementally
    if (mom_count_ < 2 || mom_ref_mid_ <= 0.0 || last_mid_ <= 0.0) return 0.0f;
    return static_cast<float>((last_mid_ - mom_ref_mid_) / mom_ref_mid_);
}

float FeatureExtractor::raw_time_since_last_batch_ms() const noexcept {
    if (last_batch_ns_ <= 0 || last_timestamp_ns_ <= 0) return 1000.0f;
    int64_t delta = last_timestamp_ns_ - last_batch_ns_;
    if (delta < 0) delta = 0;
    if (delta > kMaxSinceBatch) delta = kMaxSinceBatch;
    return static_cast<float>(delta) / 1e6f;  // convert to ms
}

MarketFeatures FeatureExtractor::snapshot() const noexcept {
    MarketFeatures f;  // zero-initialized

    // [0] spread_bps — normalize: 0..50 bps → 0..1 (cap at 50)
    float spread = raw_spread_bps();
    f.v[F_SPREAD] = std::min(spread / 50.0f, 1.0f);

    // [1] volatility — normalize: cap at 0.01 (1%) return std dev
    float vol = raw_volatility();
    f.v[F_VOLATILITY] = std::min(vol / 0.01f, 1.0f);

    // [2] message_rate — normalize: 0..10000 msgs/s → 0..1
    float msg_rate = raw_message_rate();
    f.v[F_MSG_RATE] = std::min(msg_rate / 10000.0f, 1.0f);

    // [3] time_of_day — minutes since 9:30am, normalize over 6.5h session
    if (last_timestamp_ns_ > 0) {
        int64_t sec_since_midnight = last_timestamp_ns_ / 1'000'000'000LL;
        double session_frac = static_cast<double>(sec_since_midnight - kMarketOpenSec)
                              / kSessionLenSec;
        session_frac = std::max(0.0, std::min(1.0, session_frac));
        f.v[F_TIME_OF_DAY] = static_cast<float>(session_frac);
    }

    // [4] queue_depth — normalize: 0..1280 (5 buffers × 256) → 0..1
    f.v[F_QUEUE_DEPTH] = std::min(static_cast<float>(queue_depth_) / 1280.0f, 1.0f);

    // [5] price_momentum — range approx [-1, +1], already normalized
    float mom = raw_price_momentum();
    f.v[F_MOMENTUM] = std::max(-1.0f, std::min(1.0f, mom * 100.0f));  // ×100 for bps-scale

    // [6] order_imbalance — already in [-1, +1]
    f.v[F_IMBALANCE] = raw_order_imbalance();

    // [7] cancel_ratio — already in [0, 1]
    f.v[F_CANCEL_RATE] = raw_cancel_ratio();

    // [8] batch_success_rate — already in [0, 1] (EMA)
    f.v[F_BATCH_SR] = batch_success_rate_;

    // [9] time_since_last_batch — normalize: 0..1000ms → 0..1
    float tsb = raw_time_since_last_batch_ms();
    f.v[F_SINCE_BATCH] = std::min(tsb / 1000.0f, 1.0f);

    // [10..15] remain zero (padding for AVX2)
    return f;
}

} // namespace adaptive_batcher
