#include "adaptive_batcher/common.hpp"
#include "adaptive_batcher/market_features.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace adaptive_batcher;

// ─── Market regimes ──────────────────────────────────────────────────────────

struct RegimeParams {
    std::string name;
    double spread_ticks_mean;   // mean bid-ask spread in ticks
    double spread_ticks_std;    // std dev of spread
    double msg_rate_per_sec;    // messages per second
    double volatility_bps;      // annualized volatility in bps
    double imbalance_mean;      // order flow imbalance mean
    double imbalance_std;       // std dev of imbalance
};

// Five predefined regimes
static const std::array<RegimeParams, 5> kRegimes = {{
    {"OPEN",     3.0, 1.5, 8000, 300, 0.0,  0.5},   // Opening auction: wide spread, high vol
    {"MORNING",  2.0, 0.8, 5000, 150, 0.1,  0.3},   // Active morning trading
    {"MIDDAY",   1.5, 0.5, 2000,  80, 0.0,  0.2},   // Quiet midday
    {"AFTERNOON",2.0, 0.7, 4000, 130, -0.1, 0.3},   // Afternoon pickup
    {"CLOSE",    4.0, 2.0, 9000, 350, 0.2,  0.6},   // Closing cross: wide spread, high vol
}};

// Default regime transition schedule (minutes since open)
static const std::array<std::pair<double, int>, 5> kSchedule = {{
    {0.0,   0},  // OPEN at t=0
    {30.0,  1},  // MORNING at t=30min
    {120.0, 2},  // MIDDAY at t=2h
    {270.0, 3},  // AFTERNOON at t=4.5h
    {360.0, 4},  // CLOSE at t=6h
}};

struct MarketEvent {
    int64_t  timestamp_ns;
    int64_t  bid_ticks;
    int64_t  ask_ticks;
    int64_t  bid_qty;
    int64_t  ask_qty;
    char     msg_type;  // 'A', 'X', 'E'
    int      regime_idx;
};

using EventCallback = std::function<void(const MarketEvent&)>;

static int get_regime(double elapsed_minutes) {
    int regime = 0;
    for (auto [t, r] : kSchedule) {
        if (elapsed_minutes >= t) regime = r;
    }
    return regime;
}

static void generate_events(int n_events, EventCallback callback,
                              int64_t start_ns = 34200LL * 1'000'000'000LL) {
    std::mt19937_64 rng(42);
    std::normal_distribution<double> norm(0.0, 1.0);
    std::uniform_real_distribution<double> unif(0.0, 1.0);

    int64_t ts    = start_ns;
    double  mid   = 100000.0;  // reference mid in ticks (= $10.0000 at 4 dp)

    for (int i = 0; i < n_events; ++i) {
        double elapsed_min = static_cast<double>(ts - start_ns) / 60e9;
        int    reg_idx     = get_regime(elapsed_min);
        const RegimeParams& reg = kRegimes[reg_idx];

        // Inter-event time: Poisson with rate = msg_rate_per_sec
        double mean_gap_ns = 1e9 / reg.msg_rate_per_sec;
        double gap_ns      = -mean_gap_ns * std::log(std::max(1e-10, unif(rng)));
        ts += static_cast<int64_t>(gap_ns);

        // Spread
        double spread = std::max(1.0, reg.spread_ticks_mean + norm(rng) * reg.spread_ticks_std);
        double bid    = mid - spread / 2.0;
        double ask    = mid + spread / 2.0;

        // Order flow imbalance
        double imb = reg.imbalance_mean + norm(rng) * reg.imbalance_std;
        imb        = std::max(-1.0, std::min(1.0, imb));

        int64_t bid_qty = static_cast<int64_t>(200.0 * (1.0 + imb));
        int64_t ask_qty = static_cast<int64_t>(200.0 * (1.0 - imb));
        bid_qty = std::max(1LL, bid_qty);
        ask_qty = std::max(1LL, ask_qty);

        // Message type distribution: 60% add, 30% cancel, 10% execute
        char msg_type;
        double roll = unif(rng);
        if      (roll < 0.60) msg_type = 'A';
        else if (roll < 0.90) msg_type = 'X';
        else                  msg_type = 'E';

        // Mid price walk (GBM)
        double dt       = gap_ns / 1e9;
        double vol_tick = reg.volatility_bps / 10000.0 * std::sqrt(dt / (252.0 * 6.5 * 3600.0));
        mid *= std::exp(vol_tick * norm(rng));

        MarketEvent ev;
        ev.timestamp_ns = ts;
        ev.bid_ticks    = static_cast<int64_t>(bid);
        ev.ask_ticks    = static_cast<int64_t>(ask);
        ev.bid_qty      = bid_qty;
        ev.ask_qty      = ask_qty;
        ev.msg_type     = msg_type;
        ev.regime_idx   = reg_idx;

        callback(ev);
    }
}

int main(int argc, char* argv[]) {
    int n = 1000;
    if (argc > 1) n = std::stoi(argv[1]);

    std::cout << "timestamp_ns,bid_ticks,ask_ticks,bid_qty,ask_qty,msg_type,regime\n";

    generate_events(n, [](const MarketEvent& ev) {
        std::cout << ev.timestamp_ns << ','
                  << ev.bid_ticks   << ','
                  << ev.ask_ticks   << ','
                  << ev.bid_qty     << ','
                  << ev.ask_qty     << ','
                  << ev.msg_type    << ','
                  << ev.regime_idx  << '\n';
    });

    return 0;
}
