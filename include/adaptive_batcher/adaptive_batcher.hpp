#pragma once

#include "common.hpp"
#include "experience_queue.hpp"
#include "market_features.hpp"
#include "policy_engine.hpp"
#include "batch_buffer.hpp"
#include "training_thread.hpp"

#include <atomic>
#include <chrono>
#include <string>
#include <thread>

namespace adaptive_batcher {

// ─── Config ──────────────────────────────────────────────────────────────────

struct Config {
    // Reward shaping
    float alpha = 1.0f;             // p99 penalty weight
    float beta  = 5.0f;             // SLA violation penalty weight
    float gamma = 0.1f;             // batch size reward weight

    // Policy
    float sla_threshold_us   = 500.0f;
    float learning_rate      = 0.01f;
    float epsilon_initial    = 0.5f;
    float epsilon_final      = 0.05f;
    int   cold_start_samples = 500;

    // Files
    std::string weights_path = "data/weights/latest.csv";

    // Fallback
    float fallback_threshold = 2.0f;
    int   fallback_seconds   = 60;
};

// ─── AdaptiveBatcher ─────────────────────────────────────────────────────────
// Top-level orchestrator. Ties together feature extraction, policy inference,
// batch buffering, and async SGD training.
//
// Threading model:
//   - submit() and on_market_event() called from a SINGLE producer thread.
//   - on_batch_dispatched() may be called from the timer thread.
//   - Training runs in a dedicated background thread.

class AdaptiveBatcher {
public:
    explicit AdaptiveBatcher(const Config& cfg = {});
    ~AdaptiveBatcher();

    // Not copyable or movable
    AdaptiveBatcher(const AdaptiveBatcher&) = delete;
    AdaptiveBatcher& operator=(const AdaptiveBatcher&) = delete;

    // Start training thread and timer thread.
    void start();

    // Graceful shutdown: flush all pending batches, save weights, stop threads.
    void stop();

    // Hot path entry point (single producer thread assumed).
    // Extracts features, selects action, pushes to batch buffer, records experience.
    void submit(const Request& req);

    // Update market features from an order book event.
    void on_market_event(const BookState& state);

    // Called by batch dispatch callback with observed latency.
    // Posts latency record to training thread via LatencyFeedback queue.
    void on_batch_dispatched(uint64_t batch_id, int64_t latency_ns,
                              uint32_t batch_size = 1);

    // Access statistics (thread-safe, approximate).
    float current_p99_us() const noexcept;
    bool  fallback_active() const noexcept;

private:
    void timer_loop();

    Config           cfg_;

    ExperienceQueue  exp_queue_;
    LatencyFeedback  lat_queue_;

    FeatureExtractor extractor_;
    PolicyEngine     policy_;
    BatchBufferArray buffers_;
    TrainingThread   trainer_;

    std::thread      timer_thread_;
    alignas(64) std::atomic<bool> running_{false};

    int64_t market_open_ns_ = 0;  // set on first on_market_event()
    int64_t last_submit_ns_ = 0;
};

} // namespace adaptive_batcher
