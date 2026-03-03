#include "adaptive_batcher/adaptive_batcher.hpp"
#include "adaptive_batcher/batch_buffer.hpp"
#include "adaptive_batcher/market_features.hpp"

#include <benchmark/benchmark.h>

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

using namespace adaptive_batcher;

// ─── Baseline: Fixed-window batcher ──────────────────────────────────────────

class FixedBatcher {
public:
    explicit FixedBatcher(uint32_t window_us) : window_us_(window_us) {}

    void submit(const Request& req) {
        if (buf_.empty()) open_time_ = now_ns();
        buf_.push_back(req);

        if (buf_.size() >= 256 || (now_ns() - open_time_) >= window_us_ * 1000LL) {
            dispatch();
        }
    }

    int64_t last_latency_ns() const { return last_latency_ns_; }
    size_t  total_dispatched() const { return total_dispatched_; }

private:
    void dispatch() {
        int64_t t     = now_ns();
        last_latency_ns_ = (buf_.empty() ? 0 : t - buf_.front().timestamp_ns);
        total_dispatched_ += buf_.size();
        buf_.clear();
    }

    static int64_t now_ns() {
        using namespace std::chrono;
        return duration_cast<nanoseconds>(steady_clock::now().time_since_epoch()).count();
    }

    uint32_t              window_us_;
    std::vector<Request>  buf_;
    int64_t               open_time_     = 0;
    int64_t               last_latency_ns_ = 0;
    size_t                total_dispatched_ = 0;
};

// ─── Benchmarks ───────────────────────────────────────────────────────────────

static Request make_req(uint64_t id) {
    Request r{};
    r.id = id;
    using namespace std::chrono;
    r.timestamp_ns = duration_cast<nanoseconds>(
        steady_clock::now().time_since_epoch()).count();
    return r;
}

static void BM_FixedImmediate(benchmark::State& state) {
    FixedBatcher batcher(0);
    uint64_t id = 0;
    for (auto _ : state) {
        batcher.submit(make_req(++id));
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_FixedImmediate);

static void BM_Fixed50us(benchmark::State& state) {
    FixedBatcher batcher(50);
    uint64_t id = 0;
    for (auto _ : state) {
        batcher.submit(make_req(++id));
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Fixed50us);

static void BM_Fixed100us(benchmark::State& state) {
    FixedBatcher batcher(100);
    uint64_t id = 0;
    for (auto _ : state) {
        batcher.submit(make_req(++id));
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Fixed100us);

static void BM_Fixed200us(benchmark::State& state) {
    FixedBatcher batcher(200);
    uint64_t id = 0;
    for (auto _ : state) {
        batcher.submit(make_req(++id));
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Fixed200us);

static void BM_Fixed500us(benchmark::State& state) {
    FixedBatcher batcher(500);
    uint64_t id = 0;
    for (auto _ : state) {
        batcher.submit(make_req(++id));
    }
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Fixed500us);

static void BM_Adaptive(benchmark::State& state) {
    Config cfg;
    cfg.weights_path     = "";  // no file I/O in benchmark
    cfg.epsilon_initial  = 0.1f;
    cfg.epsilon_final    = 0.05f;
    cfg.cold_start_samples = 100;

    AdaptiveBatcher ab(cfg);
    ab.start();

    // Feed some market events to init features
    BookState bs{100000, 100002, 500, 300, 1,
                 34200LL * 1'000'000'000LL + 3'600'000'000'000LL, 'A'};
    for (int i = 0; i < 50; ++i) {
        bs.timestamp_ns += 1'000'000;
        ab.on_market_event(bs);
    }

    uint64_t id = 0;
    for (auto _ : state) {
        bs.timestamp_ns += 100'000;
        ab.on_market_event(bs);
        ab.submit(make_req(++id));
    }

    ab.stop();
    state.SetItemsProcessed(state.iterations());
}
BENCHMARK(BM_Adaptive)->MinTime(1.0);

BENCHMARK_MAIN();
