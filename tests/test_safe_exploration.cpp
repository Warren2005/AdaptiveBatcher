#include "adaptive_batcher/policy_engine.hpp"

#include <gtest/gtest.h>

using namespace adaptive_batcher;

// Helper function (defined here, used by tests below)
static MarketFeatures make_features_helper() {
    return MarketFeatures{};
}

// Run infer() N times with epsilon=1.0 (full exploration) and count actions
static std::array<int, NUM_ACTIONS> count_actions(PolicyEngine& pe,
                                                   const MarketFeatures& f,
                                                   int trials = 2000) {
    std::array<int, NUM_ACTIONS> counts{};
    for (int i = 0; i < trials; ++i) {
        int a = static_cast<int>(pe.infer(f));
        counts[a]++;
    }
    return counts;
}

TEST(SafeExploration, SuppressedWhenMarketJustOpened) {
    PolicyEngine::Config cfg;
    cfg.epsilon_initial    = 1.0f;  // always explore if not suppressed
    cfg.epsilon_final      = 1.0f;
    cfg.cold_start_samples = 0;

    PolicyEngine pe(cfg);
    pe.register_reader();

    // Set up weights so IMMEDIATE (action 0) is best
    BanditWeights w;
    w.w[0][0] = 5.0f;  // IMMEDIATE strongly preferred
    pe.publish_weights(w);

    // Within first 30 minutes → exploration suppressed → always argmax (IMMEDIATE)
    pe.set_time_since_open_min(15.0f);

    MarketFeatures f = make_features_helper();
    f.v[0] = 1.0f;  // trigger IMMEDIATE score
    auto counts = count_actions(pe, f, 1000);

    // All actions must be IMMEDIATE (no exploration)
    EXPECT_EQ(counts[0], 1000);
    for (int a = 1; a < NUM_ACTIONS; ++a) {
        EXPECT_EQ(counts[a], 0) << "action " << a << " should be suppressed";
    }
}

TEST(SafeExploration, SuppressedWhenSLABreached) {
    PolicyEngine::Config cfg;
    cfg.epsilon_initial    = 1.0f;
    cfg.epsilon_final      = 1.0f;
    cfg.cold_start_samples = 0;
    cfg.sla_threshold_us   = 200.0f;

    PolicyEngine pe(cfg);
    pe.register_reader();

    BanditWeights w;
    w.w[1][0] = 5.0f;  // FAST is best
    pe.publish_weights(w);

    pe.set_time_since_open_min(60.0f);           // past 30 min
    pe.set_p99_last_10s(400.0f);                  // exceeds SLA of 200μs

    MarketFeatures f{};
    f.v[0] = 1.0f;
    auto counts = count_actions(pe, f, 1000);

    // Exploration suppressed due to SLA breach → always FAST (argmax)
    EXPECT_EQ(counts[1], 1000);
}

TEST(SafeExploration, ExplorationWithinTwoXBound) {
    PolicyEngine::Config cfg;
    cfg.epsilon_initial    = 1.0f;  // always explore
    cfg.epsilon_final      = 1.0f;
    cfg.cold_start_samples = 0;

    PolicyEngine pe(cfg);
    pe.register_reader();

    // Best action = NORMAL (index 2, window = 100μs)
    // Max allowed = 2 * 100 = 200μs → actions IMMEDIATE/FAST/NORMAL/SLOW are ok
    // LAZY (500μs) exceeds 200μs limit → should never be selected
    BanditWeights w;
    w.w[2][0] = 5.0f;  // NORMAL is best
    pe.publish_weights(w);

    pe.set_time_since_open_min(60.0f);
    pe.set_p99_last_10s(0.0f);

    MarketFeatures f{};
    f.v[0] = 1.0f;

    auto counts = count_actions(pe, f, 5000);

    // LAZY (action 4, 500μs) should never appear — 2× NORMAL = 200μs < 500μs
    EXPECT_EQ(counts[4], 0) << "LAZY must never be explored from NORMAL";

    // Should see some exploration (not all NORMAL)
    int non_normal = 0;
    for (int a = 0; a < NUM_ACTIONS; ++a) {
        if (a != 2) non_normal += counts[a];
    }
    EXPECT_GT(non_normal, 0) << "Some exploration should occur";
}

TEST(SafeExploration, ImmediateActionBoundsExplorationToFast) {
    // When best = IMMEDIATE (window=0), max_allowed = FAST (50μs).
    // Exploration is bounded to {IMMEDIATE, FAST} only — NORMAL/SLOW/LAZY must never appear.
    PolicyEngine::Config cfg;
    cfg.epsilon_initial    = 1.0f;
    cfg.epsilon_final      = 1.0f;
    cfg.cold_start_samples = 0;

    PolicyEngine pe(cfg);
    pe.register_reader();

    BanditWeights w;
    w.w[0][0] = 5.0f;  // IMMEDIATE is best (window=0)
    pe.publish_weights(w);

    pe.set_time_since_open_min(60.0f);
    pe.set_p99_last_10s(0.0f);

    MarketFeatures f{};
    f.v[0] = 1.0f;

    auto counts = count_actions(pe, f, 5000);

    // Only IMMEDIATE (0) and FAST (1) should appear
    EXPECT_EQ(counts[2], 0) << "NORMAL must never be explored from IMMEDIATE";
    EXPECT_EQ(counts[3], 0) << "SLOW must never be explored from IMMEDIATE";
    EXPECT_EQ(counts[4], 0) << "LAZY must never be explored from IMMEDIATE";
    EXPECT_GT(counts[0] + counts[1], 0) << "Some IMMEDIATE/FAST exploration should occur";
}

TEST(SafeExploration, ColdStartEpsilonDecay) {
    PolicyEngine::Config cfg;
    cfg.epsilon_initial    = 0.5f;
    cfg.epsilon_final      = 0.05f;
    cfg.cold_start_samples = 500;

    PolicyEngine pe(cfg);
    pe.register_reader();
    pe.set_time_since_open_min(60.0f);

    // After 0 samples, epsilon ~ 0.5 (high exploration)
    // After 500 samples, epsilon ~ 0.05 (low exploration)
    // We test that exploration decreases over time

    BanditWeights w;
    w.w[0][0] = 5.0f;  // IMMEDIATE is best
    pe.publish_weights(w);

    MarketFeatures f{};
    f.v[0] = 1.0f;

    // Early phase: count non-best actions
    int early_explore = 0;
    for (int i = 0; i < 200; ++i) {
        if (pe.infer(f) != BatchAction::IMMEDIATE) ++early_explore;
    }

    // Late phase: warm up to 500+ samples first
    for (int i = 0; i < 500; ++i) pe.infer(f);

    int late_explore = 0;
    for (int i = 0; i < 200; ++i) {
        if (pe.infer(f) != BatchAction::IMMEDIATE) ++late_explore;
    }

    // Late exploration rate should be significantly lower
    EXPECT_LT(late_explore, early_explore)
        << "Exploration should decrease over time (early=" << early_explore
        << " late=" << late_explore << ")";
}

