#include "adaptive_batcher/policy_engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>

#if defined(ADAPTIVE_BATCHER_AVX2)
#include <immintrin.h>
#endif

namespace adaptive_batcher {

PolicyEngine::PolicyEngine(const Config& cfg)
    : cfg_(cfg)
    , rcu_(new BanditWeights{})
{
}

PolicyEngine::~PolicyEngine() = default;

float PolicyEngine::score(int a, const float* feat) const noexcept {
    // Weights are accessed via RCU guard in infer(); here we get raw ptr.
    // This private helper is called with ptr already held.
    // NOTE: caller is responsible for RCU read guard.
    // This function should not be called directly — infer() wraps it.
    // It exists for testability; policy_engine tests call infer() instead.
    (void)a; (void)feat;
    return 0.0f;  // placeholder; actual scoring in infer()
}

float PolicyEngine::current_epsilon() const noexcept {
    uint64_t n = sample_count_.load(std::memory_order_relaxed);
    if (n >= static_cast<uint64_t>(cfg_.cold_start_samples)) {
        return cfg_.epsilon_final;
    }
    float frac = static_cast<float>(n) / static_cast<float>(cfg_.cold_start_samples);
    return cfg_.epsilon_initial + frac * (cfg_.epsilon_final - cfg_.epsilon_initial);
}

bool PolicyEngine::should_suppress_exploration() const noexcept {
    float time_open = time_since_open_min_.load(std::memory_order_relaxed);
    if (time_open < 30.0f) return true;  // within first 30 minutes

    float p99 = p99_last_10s_.load(std::memory_order_relaxed);
    if (p99 > 0.0f && p99 > cfg_.sla_threshold_us) return true;

    return false;
}

BatchAction PolicyEngine::infer(const MarketFeatures& features) noexcept {
    sample_count_.fetch_add(1, std::memory_order_relaxed);

    // Acquire RCU read lock
    vectorbook::RcuReadGuard<BanditWeights> guard(rcu_);
    const BanditWeights* w = guard.get();
    if (!w) return BatchAction::NORMAL;  // cold start fallback

    const float* feat = features.v;

    // ── Compute scores for all 5 actions (AVX2 dot product) ─────────────────
    float scores[NUM_ACTIONS];

#if defined(ADAPTIVE_BATCHER_AVX2)
    for (int a = 0; a < NUM_ACTIONS; ++a) {
        // Load 16 floats in two 8-wide registers
        __m256 fw0 = _mm256_loadu_ps(w->w[a]);       // w[0..7]
        __m256 fw1 = _mm256_loadu_ps(w->w[a] + 8);   // w[8..15]
        __m256 ff0 = _mm256_loadu_ps(feat);           // f[0..7]
        __m256 ff1 = _mm256_loadu_ps(feat + 8);       // f[8..15]

        __m256 prod0 = _mm256_mul_ps(fw0, ff0);
        __m256 prod1 = _mm256_mul_ps(fw1, ff1);
        __m256 sum   = _mm256_add_ps(prod0, prod1);

        // Horizontal sum of 8 floats
        __m128 lo   = _mm256_castps256_ps128(sum);
        __m128 hi   = _mm256_extractf128_ps(sum, 1);
        __m128 s128 = _mm_add_ps(lo, hi);
        __m128 shuf = _mm_movehdup_ps(s128);
        __m128 sums = _mm_add_ps(s128, shuf);
        shuf        = _mm_movehl_ps(shuf, sums);
        sums        = _mm_add_ss(sums, shuf);
        scores[a]   = _mm_cvtss_f32(sums);
    }
#else
    // Scalar fallback
    for (int a = 0; a < NUM_ACTIONS; ++a) {
        float s = 0.0f;
        for (int i = 0; i < 16; ++i) s += w->w[a][i] * feat[i];
        scores[a] = s;
    }
#endif

    // ── Argmax ───────────────────────────────────────────────────────────────
    int best = 0;
    for (int a = 1; a < NUM_ACTIONS; ++a) {
        if (scores[a] > scores[best]) best = a;
    }

    // ── ε-greedy exploration ──────────────────────────────────────────────────
    float eps = current_epsilon();
    if (eps > 0.0f && !should_suppress_exploration()) {
        thread_local std::mt19937 rng(std::random_device{}());
        thread_local std::uniform_real_distribution<float> udist(0.0f, 1.0f);

        if (udist(rng) < eps) {
            // Constraint: never explore into window > 2× current optimal window
            uint32_t best_window = ACTION_WINDOW_US[best];
            uint32_t max_allowed = (best_window == 0) ? ACTION_WINDOW_US[NUM_ACTIONS - 1]
                                                      : best_window * 2;

            // Collect valid exploration actions
            int candidates[NUM_ACTIONS];
            int ncand = 0;
            for (int a = 0; a < NUM_ACTIONS; ++a) {
                if (ACTION_WINDOW_US[a] <= max_allowed) {
                    candidates[ncand++] = a;
                }
            }

            if (ncand > 0) {
                std::uniform_int_distribution<int> idist(0, ncand - 1);
                best = candidates[idist(rng)];
            }
        }
    }

    return static_cast<BatchAction>(best);
}

void PolicyEngine::publish_weights(const BanditWeights& new_weights) {
    auto* copy = new BanditWeights(new_weights);
    rcu_.publish(copy);
}

void PolicyEngine::reset_weights() {
    publish_weights(BanditWeights{});
}

bool PolicyEngine::load_weights(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return false;

    BanditWeights w;
    std::string line;
    int row = 0;
    while (std::getline(f, line) && row < NUM_ACTIONS) {
        // Skip comment lines
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string tok;
        int col = 0;
        while (std::getline(ss, tok, ',') && col < NUM_FEATURES) {
            w.w[row][col] = std::stof(tok);
            ++col;
        }
        ++row;
    }
    // Read version from last line if present
    if (std::getline(f, line) && !line.empty() && line[0] == '#') {
        // version embedded as "# version=N"
        auto pos = line.find('=');
        if (pos != std::string::npos) {
            w.version = std::stoull(line.substr(pos + 1));
        }
    }

    publish_weights(w);
    return (row == NUM_ACTIONS);
}

bool PolicyEngine::save_weights(const std::string& path) const {
    vectorbook::RcuReadGuard<BanditWeights> guard(
        const_cast<vectorbook::EpochRcu<BanditWeights>&>(rcu_));
    const BanditWeights* w = guard.get();
    if (!w) return false;

    std::ofstream f(path);
    if (!f.is_open()) return false;

    f << "# AdaptiveBatcher weights v" << w->version << "\n";
    for (int a = 0; a < NUM_ACTIONS; ++a) {
        for (int i = 0; i < NUM_FEATURES; ++i) {
            f << w->w[a][i];
            if (i + 1 < NUM_FEATURES) f << ',';
        }
        f << '\n';
    }
    f << "# version=" << w->version << '\n';
    return true;
}

} // namespace adaptive_batcher
