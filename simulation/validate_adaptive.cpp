// validate_adaptive.cpp
//
// Single-threaded, deterministic validation using SIMULATED time.
//
// No timer thread. No mutex on fixed batchers. No stack overflow risk.
// Every "tick" advances a simulated nanosecond counter; all flushes are
// driven inline from the main loop. Only the SGD training thread runs
// concurrently (it owns separate queues and has its own synchronisation).
//
// Policies compared on the same 15,000-order stream:
//   Fixed-IMMEDIATE / FAST / NORMAL / SLOW / LAZY
//   Adaptive (ε-greedy contextual bandit, online SGD)
//
// Output: per-regime p50 / p99 / p99.9 table + data/validation_results.csv

#include "adaptive_batcher/batch_buffer.hpp"
#include "adaptive_batcher/common.hpp"
#include "adaptive_batcher/experience_queue.hpp"
#include "adaptive_batcher/market_features.hpp"
#include "adaptive_batcher/policy_engine.hpp"
#include "adaptive_batcher/training_thread.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

using namespace adaptive_batcher;

// ─── Regime ───────────────────────────────────────────────────────────────────

enum class Regime : int { HIGH_VOL = 0, LOW_VOL = 1 };

static BookState book_state(Regime r, int64_t sim_ts) {
    if (r == Regime::HIGH_VOL)
        // Wide spread (20 ticks), heavy bid imbalance, cancel events
        return {99990, 100010, 800, 200, 1, sim_ts, 'X'};
    else
        // Tight spread (2 ticks), balanced book, add events
        return {99999, 100001, 400, 410, 1, sim_ts, 'A'};
}

// ─── Sample ───────────────────────────────────────────────────────────────────

struct Sample {
    int64_t latency_ns;
    Regime  regime;
    int     action;  // -1 for fixed
};

static float pct(std::vector<int64_t> v, double p) {
    if (v.empty()) return 0.f;
    std::sort(v.begin(), v.end());
    return static_cast<float>(v[static_cast<size_t>(p / 100.0 * (v.size()-1))])
           / 1000.f;  // ns → μs
}

static void report(const std::string& name,
                    std::vector<Sample>& s,
                    std::ofstream& csv) {
    auto lats = [&](bool filter, Regime r) {
        std::vector<int64_t> v;
        for (auto& x : s)
            if (!filter || x.regime == r) v.push_back(x.latency_ns);
        return v;
    };
    auto all = lats(false, Regime::HIGH_VOL);
    auto hv  = lats(true,  Regime::HIGH_VOL);
    auto lv  = lats(true,  Regime::LOW_VOL);

    std::printf("%-28s %7.1f %8.1f %10.1f %12.1f %12.1f\n",
                name.c_str(),
                pct(all,50), pct(all,99), pct(all,99.9),
                pct(hv,99), pct(lv,99));

    csv << '"' << name << '"' << ','
        << pct(all,50) << ',' << pct(all,99) << ','
        << pct(all,99.9) << ',' << pct(hv,99) << ',' << pct(lv,99) << '\n';
}

// ─── Fixed-window policy (single-threaded, push-based flush) ─────────────────
// Flush fires when an order arrives AFTER the window has expired.
// Latency = (flush_time - submit_time) for each order in the batch.

struct FixedPolicy {
    std::string     name;
    int64_t         window_ns;   // 0 = immediate
    std::vector<Sample> samples;

    struct Slot { int64_t submit_ns; Regime regime; };
    std::vector<Slot> buf;
    int64_t open_ns = 0;

    void submit(int64_t sim_ns, Regime r) {
        if (buf.empty()) open_ns = sim_ns;
        buf.push_back({sim_ns, r});
        bool full    = (int)buf.size() >= 256;
        bool expired = window_ns == 0 || (sim_ns - open_ns) >= window_ns;
        if (full || expired) flush(sim_ns);
    }

    // Call after each simulated tick so timer-based windows fire correctly.
    void tick(int64_t sim_ns) {
        if (!buf.empty() && (sim_ns - open_ns) >= window_ns && window_ns > 0)
            flush(sim_ns);
    }

    void flush(int64_t sim_ns) {
        for (auto& sl : buf)
            samples.push_back({sim_ns - sl.submit_ns, sl.regime, -1});
        buf.clear();
    }
};

// ─── Adaptive policy components (heap-allocated to avoid stack overflow) ──────

struct AdaptiveComponents {
    std::unique_ptr<ExperienceQueue> exp_q;
    std::unique_ptr<LatencyFeedback> lat_q;
    std::unique_ptr<PolicyEngine>    policy;
    std::unique_ptr<TrainingThread>  trainer;
    std::unique_ptr<FeatureExtractor> extractor;

    // Per-action pending buffer for simulated-time flush
    struct Pending { int64_t submit_ns; Regime regime; int action; };
    std::array<std::vector<Pending>, NUM_ACTIONS> pending;
    std::array<int64_t, NUM_ACTIONS> open_ns;

    std::vector<Sample> samples;
    uint64_t next_id = 1;
    uint64_t next_bid = 1;

    // In-flight: batch_id → (regime, action, open_ns)
    struct InFlight { Regime regime; int action; };
    std::array<InFlight, 4096> in_flight{};

    explicit AdaptiveComponents() {
        exp_q     = std::make_unique<ExperienceQueue>();
        lat_q     = std::make_unique<LatencyFeedback>();

        PolicyEngineConfig pcfg;
        pcfg.epsilon_initial    = 0.4f;
        pcfg.epsilon_final      = 0.05f;
        pcfg.cold_start_samples = 300;
        pcfg.sla_threshold_us   = 200.f;
        policy = std::make_unique<PolicyEngine>(pcfg);
        policy->register_reader();
        policy->set_time_since_open_min(60.f);  // past suppression window

        TrainingThreadConfig tcfg;
        tcfg.learning_rate    = 0.05f;
        tcfg.alpha            = 0.01f;
        tcfg.beta             = 0.0f;
        tcfg.gamma            = 0.0f;
        tcfg.sla_threshold_us = 200.f;
        trainer = std::make_unique<TrainingThread>(*policy, *exp_q, *lat_q, tcfg);

        extractor = std::make_unique<FeatureExtractor>();
        open_ns.fill(-1);
    }

    // Select action and queue the order for simulated-time dispatch.
    int submit(int64_t sim_ns, Regime regime) {
        extractor->set_queue_depth(pending_depth());
        MarketFeatures feat = extractor->snapshot();
        BatchAction    act  = policy->infer(feat);
        int            ai   = static_cast<int>(act);

        if (pending[ai].empty()) open_ns[ai] = sim_ns;
        pending[ai].push_back({sim_ns, regime, ai});

        // Record experience (training thread picks it up asynchronously)
        Experience exp{};
        exp.context      = feat;
        exp.action       = act;
        exp.batch_id     = next_bid;
        exp.timestamp_ns = sim_ns;
        exp_q->push(exp);

        // Store in-flight mapping
        in_flight[next_bid % 4096] = {regime, ai};
        next_bid++;

        // Flush IMMEDIATE immediately
        if (ai == 0) tick(sim_ns);
        return ai;
    }

    // Drive timer-based flushes (call after each simulated tick).
    void tick(int64_t sim_ns) {
        for (int a = 0; a < NUM_ACTIONS; ++a) {
            if (pending[a].empty()) continue;
            int64_t window_ns = static_cast<int64_t>(ACTION_WINDOW_US[a]) * 1000LL;
            bool expired = (window_ns == 0) ||
                           (sim_ns - open_ns[a]) >= window_ns;
            bool full    = (int)pending[a].size() >= 256;
            if (expired || full) flush(a, sim_ns);
        }
    }

    void flush(int a, int64_t sim_ns) {
        if (pending[a].empty()) return;
        uint64_t bid = next_bid++;  // new batch ID for this flush

        int64_t worst_latency = 0;
        for (auto& p : pending[a]) {
            int64_t lat = sim_ns - p.submit_ns;
            worst_latency = std::max(worst_latency, lat);
            samples.push_back({lat, p.regime, a});
        }

        // Post to training thread
        LatencyRecord lr{bid, worst_latency, (uint32_t)pending[a].size()};
        lat_q->push(lr);

        pending[a].clear();
        open_ns[a] = -1;
    }

    void flush_all(int64_t sim_ns) {
        for (int a = 0; a < NUM_ACTIONS; ++a) flush(a, sim_ns);
    }

    void start()  { trainer->start(); }
    void stop()   { trainer->stop(); }

    uint32_t pending_depth() const {
        uint32_t d = 0;
        for (auto& v : pending) d += (uint32_t)v.size();
        return d;
    }
};

// ─── Main ─────────────────────────────────────────────────────────────────────

int main() {
    struct Phase { Regime regime; int n_orders; const char* label; };
    const std::array<Phase,3> phases = {{
        {Regime::HIGH_VOL, 5000, "HIGH_VOL — wide spread, heavy cancel flow"},
        {Regime::LOW_VOL,  5000, "LOW_VOL  — tight spread, balanced adds"},
        {Regime::HIGH_VOL, 5000, "HIGH_VOL — regime reverts, bandit must re-adapt"},
    }};

    // Inter-order gap: 100μs simulated, plus 5 market events of 0.5ms each
    // gives ~2.5ms of simulated market time between orders.
    constexpr int64_t kOrderGapNs  = 100'000LL;     // 100μs
    constexpr int64_t kEventGapNs  = 500'000LL;     // 0.5ms per market event
    constexpr int     kEventsPerOrder = 5;
    constexpr int64_t kTickNs      = 10'000LL;       // 10μs tick resolution

    // Fixed policies (stack-allocated; single-threaded, no sync needed)
    std::array<FixedPolicy, 5> fixed = {{
        {"Fixed-IMMEDIATE (0μs)",       0LL},
        {"Fixed-FAST (50μs)",       50'000LL},
        {"Fixed-NORMAL (100μs)",   100'000LL},
        {"Fixed-SLOW (200μs)",     200'000LL},
        {"Fixed-LAZY (500μs)",     500'000LL},
    }};

    // Adaptive components (heap-allocated; ~850KB total)
    auto adaptive = std::make_unique<AdaptiveComponents>();
    adaptive->start();

    int64_t sim_ns = (34200LL + 3600LL) * 1'000'000'000LL;  // 1hr into session

    std::printf("\nAdaptiveBatcher — End-to-End Validation\n");
    std::printf("========================================\n");
    std::printf("15,000 orders · 3 regime phases · 6 policies\n");
    std::printf("Simulated time · 100μs inter-order gap · 10μs tick resolution\n\n");

    for (auto& phase : phases) {
        std::printf("▶  Phase: %s (%d orders)\n", phase.label, phase.n_orders);

        // Warm up feature extractor for this regime (100 events)
        for (int w = 0; w < 100; ++w) {
            sim_ns += kEventGapNs;
            adaptive->extractor->on_market_event(book_state(phase.regime, sim_ns));
        }

        for (int i = 0; i < phase.n_orders; ++i) {
            // Feed market events to feature extractor; tick timers during each gap
            for (int e = 0; e < kEventsPerOrder; ++e) {
                int64_t event_end = sim_ns + kEventGapNs;
                for (int64_t t = sim_ns + kTickNs; t <= event_end; t += kTickNs) {
                    adaptive->tick(t);
                    for (auto& fp : fixed) fp.tick(t);
                }
                sim_ns = event_end;
                adaptive->extractor->on_market_event(book_state(phase.regime, sim_ns));
            }

            // Submit to adaptive policy
            adaptive->submit(sim_ns, phase.regime);

            // Submit to all fixed policies
            for (auto& fp : fixed) fp.submit(sim_ns, phase.regime);

            // Advance simulated time by order gap and fire periodic ticks
            int64_t end_ns = sim_ns + kOrderGapNs;
            for (int64_t t = sim_ns + kTickNs; t <= end_ns; t += kTickNs) {
                adaptive->tick(t);
                for (auto& fp : fixed) fp.tick(t);
            }
            sim_ns = end_ns;
        }
    }

    // Final drain
    adaptive->flush_all(sim_ns + 1'000'000LL);
    adaptive->stop();

    // ── Report ────────────────────────────────────────────────────────────────
    std::ofstream csv("data/validation_results.csv");
    csv << "policy,p50_us,p99_us,p999_us,high_vol_p99_us,low_vol_p99_us\n";

    std::printf("\n%-28s %7s %8s %10s %12s %12s\n",
                "Policy", "p50(μs)", "p99(μs)", "p99.9(μs)",
                "HiVol-p99", "LoVol-p99");
    std::printf("%s\n", std::string(81,'-').c_str());

    for (auto& fp : fixed)
        report(fp.name, fp.samples, csv);

    std::printf("%s\n", std::string(81,'-').c_str());
    report("Adaptive (bandit+SGD)", adaptive->samples, csv);

    // Action distribution by regime
    std::printf("\nAdaptive action distribution:\n");
    for (Regime r : {Regime::HIGH_VOL, Regime::LOW_VOL}) {
        std::printf("  %s:\n", r == Regime::HIGH_VOL ? "HIGH_VOL" : "LOW_VOL");
        std::array<int,5> cnt{};
        int tot = 0;
        for (auto& s : adaptive->samples)
            if (s.regime == r && s.action >= 0) { cnt[s.action]++; ++tot; }
        const char* nm[] = {"IMMEDIATE(0μs)","FAST(50μs)","NORMAL(100μs)",
                             "SLOW(200μs)","LAZY(500μs)"};
        for (int a = 0; a < 5; ++a)
            if (cnt[a])
                std::printf("    %-18s %5.1f%%\n", nm[a], 100.f*cnt[a]/tot);
    }

    // Key comparisons
    auto hv_lats = [](std::vector<Sample>& v) {
        std::vector<int64_t> lats;
        for (auto& s : v)
            if (s.regime == Regime::HIGH_VOL) lats.push_back(s.latency_ns);
        return lats;
    };
    auto all_lats = [](std::vector<Sample>& v) {
        std::vector<int64_t> lats;
        for (auto& s : v) lats.push_back(s.latency_ns);
        return lats;
    };

    auto norm_hv  = hv_lats(fixed[2].samples);   // Fixed-NORMAL
    auto slow_hv  = hv_lats(fixed[3].samples);   // Fixed-SLOW (conservative HIGH_VOL baseline)
    auto adap_hv  = hv_lats(adaptive->samples);
    auto norm_all = all_lats(fixed[2].samples);
    auto adap_all = all_lats(adaptive->samples);

    float norm_p99     = pct(norm_hv, 99);
    float slow_p99     = pct(slow_hv, 99);
    float adap_p99     = pct(adap_hv, 99);
    float norm_p50_all = pct(norm_all, 50);
    float adap_p50_all = pct(adap_all, 50);

    float vs_normal = norm_p99 > 0 ? 100.f * (norm_p99 - adap_p99) / norm_p99 : 0.f;
    float vs_slow   = slow_p99 > 0 ? 100.f * (slow_p99 - adap_p99) / slow_p99 : 0.f;
    float p50_delta = norm_p50_all > 0 ? 100.f * (norm_p50_all - adap_p50_all) / norm_p50_all : 0.f;

    std::printf("\nHIGH_VOL p99 comparison (adaptive vs fixed baselines):\n");
    std::printf("  vs Fixed-NORMAL (100μs): %.0fμs → %.0fμs  (%.1f%% %s)\n",
                norm_p99, adap_p99,
                std::abs(vs_normal), vs_normal >= 0 ? "reduction" : "increase");
    std::printf("  vs Fixed-SLOW  (200μs): %.0fμs → %.0fμs  (%.1f%% %s)  ← conservative baseline\n",
                slow_p99, adap_p99,
                std::abs(vs_slow), vs_slow >= 0 ? "reduction" : "increase");
    std::printf("\nOverall p50 comparison (all regimes):\n");
    std::printf("  Fixed-NORMAL p50=%.0fμs  Adaptive p50=%.0fμs  (%.1f%% %s)\n",
                norm_p50_all, adap_p50_all,
                std::abs(p50_delta), p50_delta >= 0 ? "reduction" : "increase");

    std::printf("\nNote: p99 exploration tail reflects ε=%.0f%% random actions after cold-start.\n",
                100.f * 0.05f);
    std::printf("→ CSV written to data/validation_results.csv\n\n");
    return 0;
}
