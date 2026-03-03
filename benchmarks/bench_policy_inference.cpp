#include "adaptive_batcher/policy_engine.hpp"

#include <benchmark/benchmark.h>

using namespace adaptive_batcher;

static void BM_PolicyInfer(benchmark::State& state) {
    PolicyEngine::Config cfg;
    cfg.epsilon_initial    = 0.0f;
    cfg.epsilon_final      = 0.0f;
    cfg.cold_start_samples = 0;

    PolicyEngine pe(cfg);
    pe.register_reader();
    pe.set_time_since_open_min(60.0f);

    BanditWeights w;
    for (int a = 0; a < NUM_ACTIONS; ++a)
        for (int i = 0; i < 16; ++i)
            w.w[a][i] = static_cast<float>(a * 0.1f + i * 0.01f);
    pe.publish_weights(w);

    MarketFeatures f;
    for (int i = 0; i < NUM_FEATURES; ++i) f.v[i] = static_cast<float>(i) / NUM_FEATURES;

    for (auto _ : state) {
        BatchAction action = pe.infer(f);
        benchmark::DoNotOptimize(action);
    }
}
BENCHMARK(BM_PolicyInfer)->MinTime(2.0);

static void BM_PolicyInfer_WithExploration(benchmark::State& state) {
    PolicyEngine::Config cfg;
    cfg.epsilon_initial    = 0.1f;
    cfg.epsilon_final      = 0.1f;
    cfg.cold_start_samples = 0;

    PolicyEngine pe(cfg);
    pe.register_reader();
    pe.set_time_since_open_min(60.0f);

    BanditWeights w;
    pe.publish_weights(w);

    MarketFeatures f;
    for (int i = 0; i < NUM_FEATURES; ++i) f.v[i] = 0.5f;

    for (auto _ : state) {
        BatchAction action = pe.infer(f);
        benchmark::DoNotOptimize(action);
    }
}
BENCHMARK(BM_PolicyInfer_WithExploration)->MinTime(2.0);

BENCHMARK_MAIN();
