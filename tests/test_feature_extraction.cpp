#include "adaptive_batcher/market_features.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace adaptive_batcher;

// Helper: create a simple book state
static BookState make_state(int64_t bid, int64_t ask,
                             int64_t bid_qty, int64_t ask_qty,
                             int64_t ts_ns, char msg_type = 'A',
                             int64_t tick = 1) {
    return BookState{bid, ask, bid_qty, ask_qty, tick, ts_ns, msg_type};
}

// Market open is at 9:30am = 34200s since midnight → timestamp 34200 * 1e9 ns
static constexpr int64_t kOpenNs = 34200LL * 1'000'000'000LL;

TEST(FeatureExtractor, SpreadBps) {
    FeatureExtractor fe;
    // bid = 10000 ticks, ask = 10002 ticks → spread = 2 ticks
    // mid = 10001, spread_bps = 2/10001 * 10000 ≈ 1.999 bps
    fe.on_market_event(make_state(10000, 10002, 100, 100, kOpenNs));
    float raw = fe.raw_spread_bps();
    EXPECT_NEAR(raw, 1.9998f, 0.01f);

    MarketFeatures f = fe.snapshot();
    // normalized: spread_bps / 50
    EXPECT_NEAR(f.v[0], raw / 50.0f, 0.001f);
}

TEST(FeatureExtractor, OrderImbalance) {
    FeatureExtractor fe;
    // bid_qty=300, ask_qty=100 → imbalance = (300-100)/(300+100) = 0.5
    fe.on_market_event(make_state(10000, 10002, 300, 100, kOpenNs));
    float raw = fe.raw_order_imbalance();
    EXPECT_NEAR(raw, 0.5f, 0.001f);

    // Symmetric → 0
    FeatureExtractor fe2;
    fe2.on_market_event(make_state(10000, 10002, 200, 200, kOpenNs));
    EXPECT_NEAR(fe2.raw_order_imbalance(), 0.0f, 0.001f);

    // All bid → 1
    FeatureExtractor fe3;
    fe3.on_market_event(make_state(10000, 10002, 500, 0, kOpenNs));
    EXPECT_NEAR(fe3.raw_order_imbalance(), 1.0f, 0.001f);
}

TEST(FeatureExtractor, CancelRatio) {
    FeatureExtractor fe;
    // Send 5 adds then 5 cancels
    for (int i = 0; i < 5; ++i) {
        fe.on_market_event(make_state(10000, 10002, 100, 100, kOpenNs + i, 'A'));
    }
    for (int i = 0; i < 5; ++i) {
        fe.on_market_event(make_state(10000, 10002, 100, 100, kOpenNs + 5 + i, 'X'));
    }
    float ratio = fe.raw_cancel_ratio();
    EXPECT_NEAR(ratio, 0.5f, 0.01f);
}

TEST(FeatureExtractor, BatchSuccessRate) {
    FeatureExtractor fe;
    // Feed some events to init
    fe.on_market_event(make_state(10000, 10002, 100, 100, kOpenNs));

    // Start at 1.0 (optimistic)
    MarketFeatures f = fe.snapshot();
    EXPECT_NEAR(f.v[8], 1.0f, 0.001f);

    // Report a failure (actual > configured)
    fe.on_batch_dispatched(100, 200);
    f = fe.snapshot();
    // EMA update: 0.9 * 1.0 + 0.1 * 0.0 = 0.9
    EXPECT_NEAR(f.v[8], 0.9f, 0.001f);

    // Report a success
    fe.on_batch_dispatched(100, 50);
    f = fe.snapshot();
    // EMA: 0.9 * 0.9 + 0.1 * 1.0 = 0.91
    EXPECT_NEAR(f.v[8], 0.91f, 0.001f);
}

TEST(FeatureExtractor, TimeOfDay) {
    FeatureExtractor fe;
    // At market open (9:30am = 34200s)
    fe.on_market_event(make_state(10000, 10002, 100, 100, kOpenNs));
    MarketFeatures f = fe.snapshot();
    // 0 seconds into session → 0.0
    EXPECT_NEAR(f.v[3], 0.0f, 0.01f);

    // At 12:45pm = 9.5 + 3.25 hours = 46500s from midnight
    // session fraction = (46500 - 34200) / 23400 = 12300/23400 ≈ 0.526
    int64_t noon_ns = 46500LL * 1'000'000'000LL;
    FeatureExtractor fe2;
    fe2.on_market_event(make_state(10000, 10002, 100, 100, noon_ns));
    f = fe2.snapshot();
    EXPECT_NEAR(f.v[3], 12300.0f / 23400.0f, 0.01f);
}

TEST(FeatureExtractor, QueueDepth) {
    FeatureExtractor fe;
    fe.on_market_event(make_state(10000, 10002, 100, 100, kOpenNs));
    fe.set_queue_depth(640);  // 50% of max (5 * 256 = 1280)
    MarketFeatures f = fe.snapshot();
    EXPECT_NEAR(f.v[4], 0.5f, 0.001f);
}

TEST(FeatureExtractor, PaddingZero) {
    FeatureExtractor fe;
    fe.on_market_event(make_state(10000, 10002, 100, 100, kOpenNs));
    MarketFeatures f = fe.snapshot();
    for (int i = 10; i < 16; ++i) {
        EXPECT_EQ(f.v[i], 0.0f) << "padding slot " << i << " should be zero";
    }
}

TEST(FeatureExtractor, IncrementalConsistency) {
    // Apply the same sequence of events to two extractors; snapshots must match.
    FeatureExtractor fe1, fe2;

    // 20 events
    for (int i = 0; i < 20; ++i) {
        int64_t ts   = kOpenNs + i * 1'000'000LL;  // 1ms apart
        int64_t bid  = 10000 + i;
        int64_t ask  = 10002 + i;
        char    type = (i % 3 == 2) ? 'X' : 'A';

        BookState s = make_state(bid, ask, 100 + i, 100 - i, ts, type);
        fe1.on_market_event(s);
        fe2.on_market_event(s);
    }

    MarketFeatures f1 = fe1.snapshot();
    MarketFeatures f2 = fe2.snapshot();

    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(f1.v[i], f2.v[i]) << "feature " << i << " mismatch";
    }
}

TEST(FeatureExtractor, MessageRate) {
    FeatureExtractor fe;
    // Send 500 messages within 1s window
    for (int i = 0; i < 500; ++i) {
        int64_t ts = kOpenNs + static_cast<int64_t>(i) * 1'000'000LL;  // 1ms apart
        fe.on_market_event(make_state(10000, 10002, 100, 100, ts));
    }
    // All 500 events are within 1 second
    float rate = fe.raw_message_rate();
    EXPECT_GT(rate, 100.0f);  // at least 100 messages in window
}
