#include "adaptive_batcher/training_thread.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>

namespace adaptive_batcher {

// ─── P99Tracker ──────────────────────────────────────────────────────────────

void P99Tracker::add(int64_t latency_ns) noexcept {
    buf_[count_ % kWindowSize] = latency_ns;
    ++count_;
    filled_ = std::min(count_, kWindowSize);
    sorted_ = false;
}

void P99Tracker::ensure_sorted() const noexcept {
    if (sorted_) return;
    std::copy(buf_.begin(), buf_.begin() + filled_, sorted_buf_.begin());
    std::sort(sorted_buf_.begin(), sorted_buf_.begin() + filled_);
    sorted_ = true;
}

float P99Tracker::p99_us() const noexcept {
    if (filled_ == 0) return 0.0f;
    ensure_sorted();
    int idx = static_cast<int>(0.99 * (filled_ - 1));
    return static_cast<float>(sorted_buf_[idx]) / 1000.0f;  // ns → μs
}

float P99Tracker::p50_us() const noexcept {
    if (filled_ == 0) return 0.0f;
    ensure_sorted();
    int idx = filled_ / 2;
    return static_cast<float>(sorted_buf_[idx]) / 1000.0f;
}

// ─── TrainingThread ──────────────────────────────────────────────────────────

TrainingThread::TrainingThread(PolicyEngine& policy,
                                ExperienceQueue& exp_queue,
                                LatencyFeedback& lat_queue,
                                const Config& cfg)
    : policy_(policy)
    , exp_queue_(exp_queue)
    , lat_queue_(lat_queue)
    , cfg_(cfg)
{
    // Load current weights as working copy
    vectorbook::RcuReadGuard<BanditWeights> guard(policy_.rcu());
    if (guard.get()) {
        working_weights_ = *guard.get();
    }

    in_flight_valid_.fill(false);
}

TrainingThread::~TrainingThread() {
    stop();
}

void TrainingThread::start() {
    running_.store(true, std::memory_order_release);
    thread_ = std::thread(&TrainingThread::run, this);
}

void TrainingThread::stop() {
    running_.store(false, std::memory_order_release);
    if (thread_.joinable()) thread_.join();
}

void TrainingThread::run() {
    // Register as RCU reader (training thread reads weights to validate)
    policy_.register_reader();

    while (running_.load(std::memory_order_acquire)) {
        bool did_work = false;

        // ── Drain latency feedback queue ─────────────────────────────────
        while (auto opt = lat_queue_.pop()) {
            did_work = true;
            auto [batch_id, latency_ns, batch_size] = *opt;

            // Record in adaptive tracker
            adaptive_tracker_.add(latency_ns);
            float p99 = adaptive_tracker_.p99_us();
            p99_atomic_.store(p99, std::memory_order_relaxed);
            policy_.set_p99_last_10s(p99);

            // Match batch_id → in-flight experience
            int slot = static_cast<int>(batch_id % kInFlightSize);
            if (in_flight_valid_[slot] && in_flight_[slot].batch_id == batch_id) {
                auto& exp = in_flight_[slot];
                // Check staleness (> 1s)
                using namespace std::chrono;
                auto now_ns = duration_cast<nanoseconds>(
                    steady_clock::now().time_since_epoch()).count();
                if (now_ns - exp.timestamp_ns > 1'000'000'000LL) {
                    // Stale — discard
                    in_flight_valid_[slot] = false;
                    continue;
                }

                exp.reward = compute_reward(p99, batch_size);
                exp.reward_ready = true;
                process_experience(exp, latency_ns, batch_size);
                in_flight_valid_[slot] = false;
            }
        }

        // ── Drain experience queue (store for later matching) ─────────────
        while (auto opt = exp_queue_.pop()) {
            did_work = true;
            Experience exp = *opt;
            if (exp.batch_id == UINT64_MAX) continue;  // force-flush, no signal

            int slot = static_cast<int>(exp.batch_id % kInFlightSize);
            in_flight_[slot]       = exp;
            in_flight_valid_[slot] = true;
        }

        // ── Periodic fallback check ────────────────────────────────────────
        check_fallback();

        if (!did_work) {
            // Brief sleep to avoid busy-spinning
            std::this_thread::sleep_for(std::chrono::microseconds(10));
        }
    }
}

void TrainingThread::process_experience(const Experience& exp,
                                         int64_t /*latency_ns*/,
                                         uint32_t /*batch_size*/) {
    sgd_update(exp, exp.reward);

    // Publish updated weights via RCU
    working_weights_.version++;
    policy_.publish_weights(working_weights_);
}

float TrainingThread::compute_reward(float p99_us, uint32_t batch_size) const noexcept {
    float r = -cfg_.alpha * p99_us;
    float violation = p99_us - cfg_.sla_threshold_us;
    if (violation > 0.0f) {
        r -= cfg_.beta * violation * violation;
    }
    r += cfg_.gamma * static_cast<float>(batch_size);
    return r;
}

void TrainingThread::sgd_update(const Experience& exp, float reward) {
    int a = static_cast<int>(exp.action);
    const float* feat = exp.context.v;

    // Compute current prediction: w_a · features
    float pred = 0.0f;
    for (int i = 0; i < 16; ++i) {
        pred += working_weights_.w[a][i] * feat[i];
    }

    float error = reward - pred;
    float lr    = cfg_.learning_rate;

    // SGD update: w_a[i] += η * error * features[i]
    for (int i = 0; i < NUM_FEATURES; ++i) {
        working_weights_.w[a][i] += lr * error * feat[i];
    }
    // Padding slots [10..15] stay zero
}

void TrainingThread::record_baseline_latency(int64_t latency_ns) {
    baseline_tracker_.add(latency_ns);
}

void TrainingThread::check_fallback() {
    using namespace std::chrono;
    auto now_ns = duration_cast<nanoseconds>(
        steady_clock::now().time_since_epoch()).count();

    if (now_ns - last_fallback_check_ns_ < 1'000'000'000LL) return;  // check every 1s
    last_fallback_check_ns_ = now_ns;

    float adaptive_p99  = adaptive_tracker_.p99_us();
    float baseline_p99  = baseline_tracker_.p99_us();

    if (baseline_p99 > 0.0f &&
        adaptive_p99 > cfg_.fallback_threshold * baseline_p99) {
        ++consecutive_fallback_seconds_;
    } else {
        consecutive_fallback_seconds_ = 0;
    }

    if (consecutive_fallback_seconds_ >= cfg_.fallback_seconds) {
        fallback_active_.store(true, std::memory_order_relaxed);
        // Reset weights to uniform
        working_weights_ = BanditWeights{};
        policy_.reset_weights();
        consecutive_fallback_seconds_ = 0;
    }
}

} // namespace adaptive_batcher
