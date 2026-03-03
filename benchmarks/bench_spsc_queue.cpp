#include "adaptive_batcher/experience_queue.hpp"

#include <benchmark/benchmark.h>

using namespace adaptive_batcher;

static Experience make_exp() {
    Experience e{};
    e.batch_id = 1;
    e.action   = BatchAction::NORMAL;
    for (int i = 0; i < 10; ++i) e.context.v[i] = 0.5f;
    return e;
}

static void BM_SpscPushPop(benchmark::State& state) {
    ExperienceQueue q;
    Experience exp = make_exp();
    for (auto _ : state) {
        q.push(exp);
        auto got = q.pop();
        benchmark::DoNotOptimize(got);
    }
}
BENCHMARK(BM_SpscPushPop)->MinTime(2.0);

static void BM_SpscPush(benchmark::State& state) {
    ExperienceQueue q;
    Experience exp = make_exp();
    for (auto _ : state) {
        if (!q.push(exp)) {
            // drain half to avoid blocking
            for (int i = 0; i < 2048; ++i) q.pop();
            q.push(exp);
        }
        benchmark::DoNotOptimize(q);
    }
}
BENCHMARK(BM_SpscPush)->MinTime(2.0);

BENCHMARK_MAIN();
