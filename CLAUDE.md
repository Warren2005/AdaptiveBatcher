# AdaptiveBatcher — CLAUDE.md

## Project Overview
AdaptiveBatcher is the second layer in a three-project quant stack, sitting atop VectorBook
(`../VectorBook`). It uses a contextual bandit (linear weights, online SGD) to dynamically
choose one of five batching windows (0–500μs) based on 10 live market microstructure features.

**Goal:** ≥30% p99 latency reduction vs fixed-window baselines, sub-100ns policy inference.

## Architecture
```
VectorBook (ITCH replay, OrderBook, EpochRcu)
    ↓
AdaptiveBatcher
  ├── FeatureExtractor   — 10 market features, O(1) per event
  ├── PolicyEngine       — ε-greedy bandit, AVX2 dot product, <100ns inference
  ├── BatchBufferArray   — 5 per-action ring buffers (256 slots each)
  ├── TrainingThread     — async SGD, RCU weight publication
  └── AdaptiveBatcher    — top-level orchestrator
```

## Key Constants
- `NUM_ACTIONS = 5`, windows: 0, 50, 100, 200, 500 μs
- `NUM_FEATURES = 10`, padded to 16 floats (AVX2 alignment)
- Experience queue capacity: 4096 entries (SPSC ring buffer)
- Learning rate η = 0.01 (constant, non-stationary market)
- ε decays 0.5 → 0.05 over first 500 samples

## Build
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
ctest --test-dir build
cmake -B build-tsan -DENABLE_TSAN=ON && cmake --build build-tsan
```

## VectorBook Reuse
- `vectorbook/rcu.h` — `EpochRcu<BanditWeights>`, `RcuReadGuard<BanditWeights>`
- `vectorbook/itch_parser.h` — `BookItchHandler` for ITCH replay
- `vectorbook/order_book.h` — `OrderBook<256>` driven by ITCH replayer

## Hot Path Constraint
`submit()` assumes single producer thread (SPSC queue). Document this to callers.
AVX2 dot product: two 8-wide loads + horizontal add, zero-padded slots [10..15].

## Safety Constraints
- Exploration never into window > 2× current optimal
- Suppress exploration: time_since_market_open < 30min OR p99_last_10s > SLA
- Fallback reversion: if adaptive_p99 > 2× baseline for 60s, reset to uniform weights
