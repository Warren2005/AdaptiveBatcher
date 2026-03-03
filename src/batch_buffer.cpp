#include "adaptive_batcher/batch_buffer.hpp"

#include <cstdint>
#include <cstring>

namespace adaptive_batcher {

BatchBufferArray::BatchBufferArray(FlushCallback on_flush)
    : on_flush_(std::move(on_flush))
{
    for (int a = 0; a < NUM_ACTIONS; ++a) {
        buffers_[a].window_duration_us = ACTION_WINDOW_US[a];
    }
}

uint64_t BatchBufferArray::push(BatchAction action, const Request& req,
                                 int64_t now_ns) {
    int a = static_cast<int>(action);
    BatchBuffer& buf = buffers_[a];

    // If buffer is empty, open a new window
    if (buf.empty()) {
        buf.window_open_ns = now_ns;
    }

    bool pushed = buf.push(req);
    if (!pushed) {
        // Buffer full — force flush (no training signal)
        flush(a, now_ns, /*force=*/true);
        buf.window_open_ns = now_ns;
        buf.push(req);  // push into freshly cleared buffer
        return UINT64_MAX;  // sentinel: no training signal
    }

    // Check capacity flush (after push, if we just hit the limit)
    if (buf.size() >= kBatchBufferCapacity) {
        flush(a, now_ns, /*force=*/true);
        return UINT64_MAX;  // force flush, no training signal
    }

    return buf.batch_id_counter;  // current batch ID (not yet assigned)
}

void BatchBufferArray::tick(int64_t now_ns) {
    for (int a = 0; a < NUM_ACTIONS; ++a) {
        BatchBuffer& buf = buffers_[a];
        if (buf.empty()) continue;

        uint32_t dur_us = buf.window_duration_us;
        if (dur_us == 0) {
            // IMMEDIATE action — flush on tick
            flush(a, now_ns);
            continue;
        }

        int64_t elapsed_ns = now_ns - buf.window_open_ns;
        if (elapsed_ns >= static_cast<int64_t>(dur_us) * 1000LL) {
            flush(a, now_ns);
        }
    }
}

void BatchBufferArray::flush(int action_idx, int64_t now_ns, bool force) {
    BatchBuffer& buf = buffers_[action_idx];
    if (buf.empty()) return;

    uint64_t batch_id = batch_id_seq_.fetch_add(1, std::memory_order_relaxed);

    // Drain all requests from the buffer
    uint32_t n = buf.size();
    // Temporary storage on the stack (up to 256)
    Request requests[kBatchBufferCapacity];
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t h = buf.head.load(std::memory_order_relaxed);
        requests[i] = buf.slots[h % kBatchBufferCapacity];
        buf.head.store(h + 1, std::memory_order_release);
    }

    buf.batch_id_counter = batch_id;

    FlushResult result{batch_id, n, now_ns, action_idx};

    if (on_flush_) {
        on_flush_(requests, n, result);
    }
}

void BatchBufferArray::force_flush_all(int64_t now_ns) {
    for (int a = 0; a < NUM_ACTIONS; ++a) {
        flush(a, now_ns, /*force=*/true);
    }
}

uint32_t BatchBufferArray::total_depth() const noexcept {
    uint32_t total = 0;
    for (const auto& buf : buffers_) {
        total += buf.size();
    }
    return total;
}

} // namespace adaptive_batcher
