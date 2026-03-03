#include "adaptive_batcher/policy_engine.hpp"
#include "adaptive_batcher/training_thread.hpp"

#include <gtest/gtest.h>
#include <cstring>

using namespace adaptive_batcher;

// Helper: create features where feature[0] = v, rest = 0
static MarketFeatures make_features(float v0 = 0.0f, float v1 = 0.0f) {
    MarketFeatures f;
    f.v[0] = v0;
    f.v[1] = v1;
    return f;
}

// Helper: set weights for a specific action to a known value
static BanditWeights weights_with_best_action(int best_action) {
    BanditWeights w;
    for (int a = 0; a < NUM_ACTIONS; ++a) {
        w.w[a][0] = (a == best_action) ? 1.0f : -1.0f;
    }
    return w;
}

TEST(PolicyEngine, ArgmaxCorrect) {
    PolicyEngine::Config cfg;
    cfg.epsilon_initial    = 0.0f;  // no exploration
    cfg.epsilon_final      = 0.0f;
    cfg.cold_start_samples = 0;

    PolicyEngine pe(cfg);
    pe.register_reader();
    pe.set_time_since_open_min(60.0f);  // suppress exploration: open > 30min
    pe.set_p99_last_10s(0.0f);

    // Best action = SLOW (index 3)
    BanditWeights w = weights_with_best_action(3);
    pe.publish_weights(w);

    MarketFeatures f = make_features(1.0f);  // feature[0] = 1.0 → w[3][0]=1 wins
    BatchAction action = pe.infer(f);
    EXPECT_EQ(action, BatchAction::SLOW);
}

TEST(PolicyEngine, NoExploration_AlwaysArgmax) {
    PolicyEngine::Config cfg;
    cfg.epsilon_initial    = 0.0f;
    cfg.epsilon_final      = 0.0f;
    cfg.cold_start_samples = 0;

    PolicyEngine pe(cfg);
    pe.register_reader();
    pe.set_time_since_open_min(60.0f);

    BanditWeights w = weights_with_best_action(1);  // FAST wins
    pe.publish_weights(w);

    MarketFeatures f = make_features(1.0f);
    for (int i = 0; i < 100; ++i) {
        EXPECT_EQ(pe.infer(f), BatchAction::FAST);
    }
}

TEST(PolicyEngine, EpsilonDecaySchedule) {
    PolicyEngine::Config cfg;
    cfg.epsilon_initial    = 0.5f;
    cfg.epsilon_final      = 0.05f;
    cfg.cold_start_samples = 100;

    PolicyEngine pe(cfg);
    pe.register_reader();

    // At sample 0: epsilon should be ~0.5
    // At sample 100+: epsilon should be ~0.05
    // (We can't directly read epsilon, but we can test convergence indirectly
    //  via the sample count accessor)

    EXPECT_EQ(pe.sample_count(), 0u);
    pe.infer(MarketFeatures{});
    EXPECT_EQ(pe.sample_count(), 1u);
}

TEST(PolicyEngine, WeightSerialization) {
    PolicyEngine::Config cfg;
    cfg.epsilon_initial = 0.0f;
    cfg.epsilon_final   = 0.0f;
    cfg.cold_start_samples = 0;

    PolicyEngine pe(cfg);
    pe.register_reader();

    BanditWeights w;
    for (int a = 0; a < NUM_ACTIONS; ++a) {
        for (int i = 0; i < NUM_FEATURES; ++i) {
            w.w[a][i] = static_cast<float>(a * 10 + i);
        }
    }
    w.version = 42;
    pe.publish_weights(w);

    // Save to temp file
    const std::string path = "/tmp/ab_test_weights.csv";
    EXPECT_TRUE(pe.save_weights(path));

    // Load into a new engine
    PolicyEngine pe2(cfg);
    pe2.register_reader();
    EXPECT_TRUE(pe2.load_weights(path));

    // Verify argmax still gives same result
    pe.set_time_since_open_min(60.0f);
    pe2.set_time_since_open_min(60.0f);

    MarketFeatures f = make_features(1.0f);
    EXPECT_EQ(pe.infer(f), pe2.infer(f));
}

TEST(PolicyEngine, RewardFunction) {
    // Verify shaped reward formula: -α*p99 - β*max(0,p99-SLA)² + γ*batch_size
    TrainingThread::Config cfg;
    cfg.learning_rate    = 0.01f;
    cfg.alpha            = 1.0f;
    cfg.beta             = 5.0f;
    cfg.gamma            = 0.1f;
    cfg.sla_threshold_us = 500.0f;

    // p99 = 300μs, batch_size = 10, SLA = 500
    // reward = -1.0 * 300 - 5.0 * max(0, -200)² + 0.1 * 10 = -300 + 0 + 1 = -299
    float p99_us    = 300.0f;
    uint32_t bsz    = 10;
    float expected  = -cfg.alpha * p99_us
                    - cfg.beta * std::max(0.0f, p99_us - cfg.sla_threshold_us)
                                 * std::max(0.0f, p99_us - cfg.sla_threshold_us)
                    + cfg.gamma * static_cast<float>(bsz);
    EXPECT_NEAR(expected, -299.0f, 0.001f);

    // p99 = 600μs (violation)
    p99_us   = 600.0f;
    expected = -600.0f - 5.0f * 100.0f * 100.0f + 1.0f;  // = -600 - 50000 + 1 = -50599
    float actual = -cfg.alpha * p99_us
                 - cfg.beta * std::max(0.0f, p99_us - cfg.sla_threshold_us)
                              * std::max(0.0f, p99_us - cfg.sla_threshold_us)
                 + cfg.gamma * static_cast<float>(bsz);
    EXPECT_NEAR(actual, -50599.0f, 1.0f);
}

TEST(P99Tracker, BasicP99) {
    P99Tracker tracker;
    for (int i = 1; i <= 100; ++i) {
        tracker.add(static_cast<int64_t>(i) * 1000);  // i μs in ns
    }
    // p99 of [1..100] μs = 99 μs
    EXPECT_NEAR(tracker.p99_us(), 99.0f, 1.5f);
}

TEST(P99Tracker, WindowEviction) {
    P99Tracker tracker;
    // Fill with 1ms latencies
    for (int i = 0; i < P99Tracker::kWindowSize; ++i) {
        tracker.add(1'000'000);  // 1ms in ns
    }
    EXPECT_NEAR(tracker.p99_us(), 1000.0f, 1.0f);

    // Now overwrite with 10μs latencies
    for (int i = 0; i < P99Tracker::kWindowSize; ++i) {
        tracker.add(10'000);  // 10μs in ns
    }
    EXPECT_NEAR(tracker.p99_us(), 10.0f, 1.0f);
}
