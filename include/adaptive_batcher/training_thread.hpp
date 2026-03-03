#pragma once

#include "common.hpp"
#include "experience_queue.hpp"
#include "policy_engine.hpp"

#include <atomic>
#include <array>
#include <cstdint>
#include <thread>

namespace adaptive_batcher {

// ─── TrainingThread::Config (at namespace scope for Clang compatibility) ─────
struct TrainingThreadConfig {
    float learning_rate      = 0.01f;
    float alpha              = 1.0f;
    float beta               = 5.0f;
    float gamma              = 0.1f;
    float sla_threshold_us   = 500.0f;
    float fallback_threshold = 2.0f;
    int   fallback_seconds   = 60;
};

// ─── P99Tracker — sliding window p99 from last N latency samples ──────────────
// Simple sorted-insertion approach over a fixed ring buffer of size N.

class P99Tracker {
public:
    static constexpr int kWindowSize = 1000;

    void add(int64_t latency_ns) noexcept;
    float p99_us() const noexcept;
    float p50_us() const noexcept;
    int   count()  const noexcept { return count_; }

private:
    std::array<int64_t, kWindowSize> buf_{};
    int count_  = 0;
    int filled_ = 0;  // number of valid entries (≤ kWindowSize)
    mutable bool sorted_ = false;
    mutable std::array<int64_t, kWindowSize> sorted_buf_{};

    void ensure_sorted() const noexcept;
};

// ─── TrainingThread ──────────────────────────────────────────────────────────
// Async SGD loop + RCU weight publication.
// Reads from ExperienceQueue (SPSC), matches batch_id → latency via LatencyFeedback.
// Posts updated p99 atomically for PolicyEngine exploration suppression.

class TrainingThread {
public:
    using Config = TrainingThreadConfig;

    TrainingThread(PolicyEngine& policy,
                   ExperienceQueue& exp_queue,
                   LatencyFeedback& lat_queue,
                   const Config& cfg = Config{});
    ~TrainingThread();

    void start();
    void stop();

    // Current p99 in microseconds (read by policy engine for exploration suppression).
    float current_p99_us() const noexcept {
        return p99_atomic_.load(std::memory_order_relaxed);
    }

    // Baseline p99 for fallback comparison (fixed 100μs window).
    void record_baseline_latency(int64_t latency_ns);

    bool fallback_active() const noexcept {
        return fallback_active_.load(std::memory_order_relaxed);
    }

private:
    void run();
    void process_experience(const Experience& exp, int64_t latency_ns, uint32_t batch_size);
    float compute_reward(float p99_us, uint32_t batch_size) const noexcept;
    void sgd_update(const Experience& exp, float reward);
    void check_fallback();

    PolicyEngine&    policy_;
    ExperienceQueue& exp_queue_;
    LatencyFeedback& lat_queue_;
    Config           cfg_;

    // In-flight map: batch_id % kInFlightSize → Experience
    static constexpr int kInFlightSize = 1024;
    std::array<Experience, kInFlightSize> in_flight_{};
    std::array<bool,       kInFlightSize> in_flight_valid_{};

    // Working copy of weights (updated by SGD, published via RCU)
    BanditWeights working_weights_;

    // P99 tracking
    P99Tracker adaptive_tracker_;
    P99Tracker baseline_tracker_;
    alignas(64) std::atomic<float> p99_atomic_{0.0f};

    // Fallback state
    alignas(64) std::atomic<bool>  fallback_active_{false};
    int  consecutive_fallback_seconds_ = 0;
    int64_t last_fallback_check_ns_   = 0;

    // Thread management
    std::thread thread_;
    alignas(64) std::atomic<bool> running_{false};
};

} // namespace adaptive_batcher
