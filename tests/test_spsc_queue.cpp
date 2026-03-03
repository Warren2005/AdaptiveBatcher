#include "adaptive_batcher/experience_queue.hpp"

#include <gtest/gtest.h>
#include <thread>
#include <vector>

using namespace adaptive_batcher;

static Experience make_exp(uint64_t batch_id) {
    Experience e{};
    e.batch_id    = batch_id;
    e.action      = BatchAction::NORMAL;
    e.timestamp_ns = static_cast<int64_t>(batch_id * 1000);
    e.reward      = 0.0f;
    e.reward_ready = false;
    for (int i = 0; i < 10; ++i) e.context.v[i] = static_cast<float>(i);
    return e;
}

TEST(SpscQueue, PushPopSingle) {
    SpscQueue<16> q;
    EXPECT_TRUE(q.empty());

    auto exp = make_exp(42);
    EXPECT_TRUE(q.push(exp));
    EXPECT_FALSE(q.empty());

    auto got = q.pop();
    ASSERT_TRUE(got.has_value());
    EXPECT_EQ(got->batch_id, 42u);
    EXPECT_TRUE(q.empty());
}

TEST(SpscQueue, PopEmpty) {
    SpscQueue<16> q;
    EXPECT_FALSE(q.pop().has_value());
}

TEST(SpscQueue, FillAndDrain) {
    // Capacity = 4096; push 4095 (leave one slot; is_full triggers at Capacity)
    SpscQueue<4096> q;

    for (uint64_t i = 0; i < 4095; ++i) {
        ASSERT_TRUE(q.push(make_exp(i))) << "failed at i=" << i;
    }

    // 4096th push should succeed (ring has 4096 slots, 4095 filled)
    EXPECT_TRUE(q.push(make_exp(4095)));

    // Next push must fail (full)
    EXPECT_FALSE(q.push(make_exp(9999)));

    // Drain all
    for (uint64_t i = 0; i < 4096; ++i) {
        auto got = q.pop();
        ASSERT_TRUE(got.has_value()) << "empty at i=" << i;
        EXPECT_EQ(got->batch_id, i);
    }

    EXPECT_TRUE(q.empty());
    EXPECT_FALSE(q.pop().has_value());
}

TEST(SpscQueue, WrapAround) {
    SpscQueue<8> q;  // small queue to test wrap

    // Fill, drain, refill
    for (int round = 0; round < 5; ++round) {
        for (uint64_t i = 0; i < 7; ++i) {
            ASSERT_TRUE(q.push(make_exp(round * 100 + i)));
        }
        for (uint64_t i = 0; i < 7; ++i) {
            auto got = q.pop();
            ASSERT_TRUE(got.has_value());
            EXPECT_EQ(got->batch_id, static_cast<uint64_t>(round * 100 + i));
        }
        EXPECT_TRUE(q.empty());
    }
}

TEST(SpscQueue, ProducerConsumerThreaded) {
    SpscQueue<4096> q;
    constexpr int kItems = 10000;

    std::vector<uint64_t> received;
    received.reserve(kItems);

    std::thread producer([&] {
        for (int i = 0; i < kItems; ++i) {
            while (!q.push(make_exp(static_cast<uint64_t>(i)))) {
                std::this_thread::yield();
            }
        }
    });

    std::thread consumer([&] {
        int count = 0;
        while (count < kItems) {
            auto got = q.pop();
            if (got) {
                received.push_back(got->batch_id);
                ++count;
            } else {
                std::this_thread::yield();
            }
        }
    });

    producer.join();
    consumer.join();

    ASSERT_EQ(static_cast<int>(received.size()), kItems);
    for (int i = 0; i < kItems; ++i) {
        EXPECT_EQ(received[i], static_cast<uint64_t>(i));
    }
}

TEST(LatencyQueue, BasicPushPop) {
    LatencyFeedback q;
    LatencyRecord r{42, 1000, 10};
    EXPECT_TRUE(q.push(r));

    auto got = q.pop();
    ASSERT_TRUE(got.has_value());
    EXPECT_EQ(got->batch_id, 42u);
    EXPECT_EQ(got->latency_ns, 1000);
    EXPECT_EQ(got->batch_size, 10u);
}
