#pragma once

#include "common.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace adaptive_batcher {

// ─── Experience — one bandit observation ─────────────────────────────────────

struct Experience {
    MarketFeatures context;       // feature vector at decision time
    BatchAction    action;
    uint64_t       batch_id;      // matched to batch for reward lookup
    int64_t        timestamp_ns;
    float          reward;        // filled by training thread after dispatch
    bool           reward_ready;
    uint8_t        _pad[2];       // alignment
};

// ─── SpscQueue — single-producer single-consumer lock-free ring buffer ────────
// Capacity must be a power of 2.

template<size_t Capacity>
class SpscQueue {
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "SpscQueue Capacity must be a power of 2");
    static constexpr size_t kMask = Capacity - 1;

public:
    SpscQueue() : head_(0), tail_(0) {}

    // Push one item. Returns false if queue is full (item not enqueued).
    // Must only be called from the producer thread.
    bool push(const Experience& item) noexcept {
        const uint32_t t = tail_.load(std::memory_order_relaxed);
        const uint32_t h = head_.load(std::memory_order_acquire);
        if (static_cast<uint32_t>(t - h) >= Capacity) return false;  // full
        slots_[t & kMask] = item;
        tail_.store(t + 1, std::memory_order_release);
        return true;
    }

    // Pop one item. Returns nullopt if queue is empty.
    // Must only be called from the consumer thread.
    std::optional<Experience> pop() noexcept {
        const uint32_t h = head_.load(std::memory_order_relaxed);
        const uint32_t t = tail_.load(std::memory_order_acquire);
        if (h == t) return std::nullopt;  // empty
        Experience item = slots_[h & kMask];
        head_.store(h + 1, std::memory_order_release);
        return item;
    }

    // Approximate size (may be stale across threads).
    size_t size_approx() const noexcept {
        return tail_.load(std::memory_order_relaxed) -
               head_.load(std::memory_order_relaxed);
    }

    bool empty() const noexcept {
        return head_.load(std::memory_order_relaxed) ==
               tail_.load(std::memory_order_relaxed);
    }

    bool is_full() const noexcept {
        return (tail_.load(std::memory_order_relaxed) -
                head_.load(std::memory_order_relaxed)) >= Capacity;
    }

private:
    alignas(64) std::atomic<uint32_t> head_;
    alignas(64) std::atomic<uint32_t> tail_;
    Experience slots_[Capacity];
};

// Default queue size used throughout the project
using ExperienceQueue = SpscQueue<4096>;

// ─── LatencyRecord — result posted from dispatch callback to training thread ──

struct LatencyRecord {
    uint64_t batch_id;
    int64_t  latency_ns;
    uint32_t batch_size;
};

template<size_t Capacity>
class LatencyQueue {
    static_assert((Capacity & (Capacity - 1)) == 0,
                  "LatencyQueue Capacity must be a power of 2");
    static constexpr size_t kMask = Capacity - 1;

public:
    LatencyQueue() : head_(0), tail_(0) {}

    bool push(const LatencyRecord& item) noexcept {
        const uint32_t t = tail_.load(std::memory_order_relaxed);
        const uint32_t h = head_.load(std::memory_order_acquire);
        if (static_cast<uint32_t>(t - h) >= Capacity) return false;
        slots_[t & kMask] = item;
        tail_.store(t + 1, std::memory_order_release);
        return true;
    }

    std::optional<LatencyRecord> pop() noexcept {
        const uint32_t h = head_.load(std::memory_order_relaxed);
        const uint32_t t = tail_.load(std::memory_order_acquire);
        if (h == t) return std::nullopt;
        LatencyRecord item = slots_[h & kMask];
        head_.store(h + 1, std::memory_order_release);
        return item;
    }

    bool empty() const noexcept {
        return head_.load(std::memory_order_relaxed) ==
               tail_.load(std::memory_order_relaxed);
    }

private:
    alignas(64) std::atomic<uint32_t> head_;
    alignas(64) std::atomic<uint32_t> tail_;
    LatencyRecord slots_[Capacity];
};

using LatencyFeedback = LatencyQueue<4096>;

} // namespace adaptive_batcher
