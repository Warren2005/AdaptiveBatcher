#pragma once

#include "common.hpp"

#include <vectorbook/rcu.h>

#include <atomic>
#include <cstdint>
#include <string>

namespace adaptive_batcher {

// ─── PolicyEngine::Config (at namespace scope for Clang compatibility) ───────
struct PolicyEngineConfig {
    float epsilon_initial    = 0.5f;
    float epsilon_final      = 0.05f;
    int   cold_start_samples = 500;
    float sla_threshold_us   = 500.0f;
};

// ─── BanditWeights ───────────────────────────────────────────────────────────
// Linear weights for all 5 actions. Published via EpochRcu.

struct alignas(64) BanditWeights {
    float    w[NUM_ACTIONS][16];  // [5 actions × 16 floats (10 real + 6 padding)]
    uint64_t version;             // incremented on each SGD update

    BanditWeights() : version(0) {
        // Uniform initialization: all weights = 0
        for (int a = 0; a < NUM_ACTIONS; ++a)
            for (int i = 0; i < 16; ++i)
                w[a][i] = 0.0f;
    }
};

// ─── PolicyEngine ────────────────────────────────────────────────────────────
// ε-greedy contextual bandit with AVX2 dot product inference.
// Inference budget: < 100ns.

class PolicyEngine {
public:
    using Config = PolicyEngineConfig;

    explicit PolicyEngine(const Config& cfg = Config{});
    ~PolicyEngine();

    // Hot path: select an action given current features. < 100ns.
    // Reads weights via RCU (single acquire load).
    BatchAction infer(const MarketFeatures& features) noexcept;

    // Called by training thread after SGD update.
    // Publishes new weights via EpochRcu (blocks until readers quiesce).
    void publish_weights(const BanditWeights& new_weights);

    // Serialize / deserialize weights (versioned CSV, 5 rows × 10 cols).
    bool load_weights(const std::string& path);
    bool save_weights(const std::string& path) const;

    // State accessible to training thread for exploration suppression.
    void set_p99_last_10s(float p99_us) noexcept {
        p99_last_10s_.store(p99_us, std::memory_order_relaxed);
    }
    void set_time_since_open_min(float minutes) noexcept {
        time_since_open_min_.store(minutes, std::memory_order_relaxed);
    }

    // Reset weights to uniform initialization.
    void reset_weights();

    // Access RCU for training thread (single writer).
    vectorbook::EpochRcu<BanditWeights>& rcu() noexcept { return rcu_; }

    // Register calling thread as an RCU reader (must call before infer()).
    void register_reader() { rcu_.register_reader(); }

    // For testing: access current sample count.
    uint64_t sample_count() const noexcept {
        return sample_count_.load(std::memory_order_relaxed);
    }

private:
    // Compute score for action a given features (AVX2 dot product).
    float score(int a, const float* feat) const noexcept;

    // Return current epsilon based on sample count decay schedule.
    float current_epsilon() const noexcept;

    // Check exploration suppression constraints.
    bool should_suppress_exploration() const noexcept;

    Config cfg_;

    // Weight management via RCU
    vectorbook::EpochRcu<BanditWeights> rcu_;

    // Atomic state for exploration suppression (written by training thread)
    alignas(64) std::atomic<float>    p99_last_10s_{0.0f};
    alignas(64) std::atomic<float>    time_since_open_min_{0.0f};
    alignas(64) std::atomic<uint64_t> sample_count_{0};
};

} // namespace adaptive_batcher
