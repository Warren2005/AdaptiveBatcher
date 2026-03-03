#pragma once

#include "common.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>

namespace adaptive_batcher {

// ─── BatchBuffer ─────────────────────────────────────────────────────────────
// Per-action ring buffer holding up to 256 pending requests.

static constexpr size_t kBatchBufferCapacity = 256;

struct BatchBuffer {
    std::array<Request, kBatchBufferCapacity> slots;

    alignas(64) std::atomic<uint32_t> head{0};
    alignas(64) std::atomic<uint32_t> tail{0};

    int64_t  window_open_ns    = 0;
    uint32_t window_duration_us = 0;
    uint64_t batch_id_counter  = 0;

    // Number of items currently in the buffer.
    uint32_t size() const noexcept {
        return tail.load(std::memory_order_relaxed) -
               head.load(std::memory_order_relaxed);
    }

    bool empty() const noexcept { return size() == 0; }

    // Push a request. Returns false if full.
    bool push(const Request& req) noexcept {
        const uint32_t t = tail.load(std::memory_order_relaxed);
        const uint32_t h = head.load(std::memory_order_acquire);
        if (t - h >= kBatchBufferCapacity) return false;
        slots[t % kBatchBufferCapacity] = req;
        tail.store(t + 1, std::memory_order_release);
        return true;
    }
};

// ─── FlushResult — returned on each batch flush ───────────────────────────────

struct FlushResult {
    uint64_t batch_id;
    uint32_t batch_size;
    int64_t  flush_time_ns;
    int      action_idx;
};

// ─── BatchBufferArray ────────────────────────────────────────────────────────
// Owns all 5 per-action buffers. Hot path: push(). Timer: tick().

class BatchBufferArray {
public:
    // Callback type: called on each flush with the drained requests.
    using FlushCallback = std::function<void(
        const Request* requests, uint32_t count, const FlushResult& result)>;

    explicit BatchBufferArray(FlushCallback on_flush = nullptr);

    // Hot path: push a request to the buffer for the given action.
    // Returns the batch_id for experience tracking.
    // NOTE: force-flush (capacity overflow) does NOT produce a training signal;
    //       the returned batch_id will be marked invalid (UINT64_MAX).
    uint64_t push(BatchAction action, const Request& req, int64_t now_ns);

    // Called periodically (every ~10μs) to check timer-based flushes.
    void tick(int64_t now_ns);

    // Drain all buffers immediately (shutdown path).
    void force_flush_all(int64_t now_ns);

    // Total pending request count across all buffers (for queue_depth feature).
    uint32_t total_depth() const noexcept;

    // Next global batch ID (monotonically increasing).
    uint64_t next_batch_id() noexcept { return batch_id_seq_.fetch_add(1, std::memory_order_relaxed); }

private:
    // Flush one buffer. Drains all requests, calls on_flush_, advances batch_id.
    void flush(int action_idx, int64_t now_ns, bool force = false);

    std::array<BatchBuffer, NUM_ACTIONS> buffers_;
    FlushCallback on_flush_;
    alignas(64) std::atomic<uint64_t> batch_id_seq_{1};
};

} // namespace adaptive_batcher
