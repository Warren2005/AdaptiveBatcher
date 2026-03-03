#include "adaptive_batcher/adaptive_batcher.hpp"

#include <chrono>
#include <thread>

namespace adaptive_batcher {

static int64_t now_ns() noexcept {
    using namespace std::chrono;
    return duration_cast<nanoseconds>(
        steady_clock::now().time_since_epoch()).count();
}

AdaptiveBatcher::AdaptiveBatcher(const Config& cfg)
    : cfg_(cfg)
    , policy_({cfg.epsilon_initial, cfg.epsilon_final,
               cfg.cold_start_samples, cfg.sla_threshold_us})
    , buffers_([this](const Request* reqs, uint32_t n, const FlushResult& res) {
        // Dispatch callback: called by timer thread or producer thread
        // For now, record a synthetic latency (real VectorBook dispatch not wired)
        int64_t t = now_ns();
        int64_t latency = t - (n > 0 ? reqs[0].timestamp_ns : t);
        if (latency < 0) latency = 0;
        on_batch_dispatched(res.batch_id, latency, res.batch_size);
    })
    , trainer_(policy_, exp_queue_, lat_queue_,
               {cfg.learning_rate, cfg.alpha, cfg.beta, cfg.gamma,
                cfg.sla_threshold_us, cfg.fallback_threshold,
                cfg.fallback_seconds})
{
    // Register producer thread as RCU reader
    policy_.register_reader();

    // Load weights if available
    if (!cfg.weights_path.empty()) {
        policy_.load_weights(cfg.weights_path);
    }
}

AdaptiveBatcher::~AdaptiveBatcher() {
    stop();
}

void AdaptiveBatcher::start() {
    running_.store(true, std::memory_order_release);
    trainer_.start();
    timer_thread_ = std::thread(&AdaptiveBatcher::timer_loop, this);
}

void AdaptiveBatcher::stop() {
    if (!running_.load(std::memory_order_acquire)) return;
    running_.store(false, std::memory_order_release);

    trainer_.stop();
    if (timer_thread_.joinable()) timer_thread_.join();

    // Flush remaining batches
    buffers_.force_flush_all(now_ns());

    // Save weights
    if (!cfg_.weights_path.empty()) {
        policy_.save_weights(cfg_.weights_path);
    }
}

void AdaptiveBatcher::submit(const Request& req) {
    // Update queue depth for feature extraction
    extractor_.set_queue_depth(buffers_.total_depth());

    // Step 1: extract features
    MarketFeatures features = extractor_.snapshot();

    // Step 2: select action
    BatchAction action = policy_.infer(features);

    // Step 3: push to batch buffer
    int64_t t = now_ns();
    uint64_t batch_id = buffers_.push(action, req, t);

    // Step 4: record experience (SPSC: single producer thread)
    Experience exp{};
    exp.context      = features;
    exp.action       = action;
    exp.batch_id     = batch_id;
    exp.timestamp_ns = t;
    exp.reward       = 0.0f;
    exp.reward_ready = false;
    exp_queue_.push(exp);

    last_submit_ns_ = t;
}

void AdaptiveBatcher::on_market_event(const BookState& state) {
    if (market_open_ns_ == 0) {
        market_open_ns_ = state.timestamp_ns;
    }
    extractor_.on_market_event(state);

    // Update time-since-open for exploration suppression
    double elapsed_min =
        static_cast<double>(state.timestamp_ns - market_open_ns_) / 60e9;
    policy_.set_time_since_open_min(static_cast<float>(elapsed_min));
}

void AdaptiveBatcher::on_batch_dispatched(uint64_t batch_id,
                                           int64_t latency_ns,
                                           uint32_t batch_size) {
    LatencyRecord lr{batch_id, latency_ns, batch_size};
    lat_queue_.push(lr);
}

float AdaptiveBatcher::current_p99_us() const noexcept {
    return trainer_.current_p99_us();
}

bool AdaptiveBatcher::fallback_active() const noexcept {
    return trainer_.fallback_active();
}

void AdaptiveBatcher::timer_loop() {
    while (running_.load(std::memory_order_acquire)) {
        int64_t t = now_ns();
        extractor_.set_queue_depth(buffers_.total_depth());
        buffers_.tick(t);
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }
}

} // namespace adaptive_batcher
