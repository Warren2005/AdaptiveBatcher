#include "adaptive_batcher/adaptive_batcher.hpp"
#include "adaptive_batcher/common.hpp"

#include <vectorbook/itch_parser.h>
#include <vectorbook/order_book.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace adaptive_batcher;
using namespace vectorbook;

// ─── SimHandler — drives AdaptiveBatcher from ITCH messages ──────────────────

class SimHandler : public BookItchHandler<256> {
public:
    SimHandler(const std::string& symbol,
               int64_t reference_price_ticks,
               AdaptiveBatcher& ab)
        : BookItchHandler<256>(symbol, reference_price_ticks)
        , ab_(ab)
        , msg_count_(0)
    {}

    void on_add_order(const MsgAddOrder& msg) override {
        BookItchHandler<256>::on_add_order(msg);
        forward_state(msg.timestamp_ns, 'A');
    }

    void on_order_cancel(const MsgOrderCancel& msg) override {
        BookItchHandler<256>::on_order_cancel(msg);
        forward_state(msg.timestamp_ns, 'X');
    }

    void on_order_delete(const MsgOrderDelete& msg) override {
        BookItchHandler<256>::on_order_delete(msg);
        forward_state(msg.timestamp_ns, 'D');
    }

    void on_order_executed(const MsgOrderExecuted& msg) override {
        BookItchHandler<256>::on_order_executed(msg);
        forward_state(msg.timestamp_ns, 'E');
    }

    void on_order_replace(const MsgOrderReplace& msg) override {
        BookItchHandler<256>::on_order_replace(msg);
        forward_state(msg.timestamp_ns, 'U');
    }

    size_t message_count() const { return msg_count_; }

private:
    void forward_state(int64_t ts, char msg_type) {
        ++msg_count_;
        const auto& bk = this->book();

        auto bid_opt = bk.best_bid();
        auto ask_opt = bk.best_ask();

        BookState state{};
        state.timestamp_ns  = ts;
        state.msg_type      = msg_type;
        state.tick_size     = bk.tick_size();
        state.best_bid_ticks = bid_opt.value_or(0);
        state.best_ask_ticks = ask_opt.value_or(0);

        if (bid_opt) state.bid_quantity = bk.bid_quantity_at(*bid_opt);
        if (ask_opt) state.ask_quantity = bk.ask_quantity_at(*ask_opt);

        ab_.on_market_event(state);

        // Generate a synthetic request every 10 messages
        if (msg_count_ % 10 == 0) {
            Request req{};
            req.id           = msg_count_;
            req.timestamp_ns = ts;
            ab_.submit(req);
        }
    }

    AdaptiveBatcher& ab_;
    size_t           msg_count_;
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <itch_file> [symbol] [ref_price_ticks]\n";
        std::cerr << "  Default symbol: SPY, ref_price=4000000 (=$400.0000 at 4dp)\n";
        return 1;
    }

    std::string filename = argv[1];
    std::string symbol   = (argc > 2) ? argv[2] : "SPY";
    int64_t ref_price    = (argc > 3) ? std::stoll(argv[3]) : 4'000'000LL;

    // Read ITCH file
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open file: " << filename << "\n";
        return 1;
    }
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(file)),
                               std::istreambuf_iterator<char>());

    std::cout << "Loaded " << data.size() << " bytes from " << filename << "\n";
    std::cout << "Symbol: " << symbol << ", ref_price: " << ref_price << "\n";

    Config cfg;
    cfg.weights_path    = "";  // no persistence in sim
    cfg.epsilon_initial = 0.3f;
    cfg.epsilon_final   = 0.05f;

    AdaptiveBatcher ab(cfg);
    ab.start();

    SimHandler handler(symbol, ref_price, ab);
    size_t n_msgs = parse_feed(data.data(), data.size(), handler);

    std::cout << "Parsed " << n_msgs << " messages, "
              << handler.message_count() << " forwarded to batcher\n";
    std::cout << "Final p99 latency: " << ab.current_p99_us() << " μs\n";
    std::cout << "Fallback active: " << (ab.fallback_active() ? "yes" : "no") << "\n";

    ab.stop();
    return 0;
}
