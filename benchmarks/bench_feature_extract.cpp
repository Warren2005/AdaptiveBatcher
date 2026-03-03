#include "adaptive_batcher/market_features.hpp"

#include <benchmark/benchmark.h>

using namespace adaptive_batcher;

static BookState make_state() {
    return BookState{100000, 100002, 500, 300, 1,
                     34200LL * 1'000'000'000LL + 3'600'000'000'000LL, 'A'};
}

static void BM_OnMarketEvent(benchmark::State& state) {
    FeatureExtractor fe;
    BookState bs = make_state();
    for (auto _ : state) {
        bs.timestamp_ns += 1000;
        fe.on_market_event(bs);
        benchmark::DoNotOptimize(fe);
    }
}
BENCHMARK(BM_OnMarketEvent)->MinTime(2.0);

static void BM_Snapshot(benchmark::State& state) {
    FeatureExtractor fe;
    BookState bs = make_state();
    // Pre-populate some history
    for (int i = 0; i < 100; ++i) {
        bs.timestamp_ns += 1'000'000;
        fe.on_market_event(bs);
    }
    for (auto _ : state) {
        MarketFeatures f = fe.snapshot();
        benchmark::DoNotOptimize(f);
    }
}
BENCHMARK(BM_Snapshot)->MinTime(2.0);

static void BM_OnEventPlusSnapshot(benchmark::State& state) {
    FeatureExtractor fe;
    BookState bs = make_state();
    for (auto _ : state) {
        bs.timestamp_ns += 1000;
        fe.on_market_event(bs);
        MarketFeatures f = fe.snapshot();
        benchmark::DoNotOptimize(f);
    }
}
BENCHMARK(BM_OnEventPlusSnapshot)->MinTime(2.0);

BENCHMARK_MAIN();
