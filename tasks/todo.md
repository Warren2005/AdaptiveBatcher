# AdaptiveBatcher — TODO

## Phase 0 — Project Skeleton ✓
- [x] CLAUDE.md
- [x] CMakeLists.txt
- [x] Directory structure

## Phase 1 — Core Types and SPSC Queue
- [ ] include/adaptive_batcher/common.hpp
- [ ] include/adaptive_batcher/experience_queue.hpp
- [ ] tests/test_spsc_queue.cpp

## Phase 2 — Market Feature Extractor
- [ ] include/adaptive_batcher/market_features.hpp
- [ ] src/market_features.cpp
- [ ] tests/test_feature_extraction.cpp

## Phase 3 — Policy Engine
- [ ] include/adaptive_batcher/policy_engine.hpp
- [ ] src/policy_engine.cpp
- [ ] tests/test_policy_correctness.cpp

## Phase 4 — Batch Buffers
- [ ] include/adaptive_batcher/batch_buffer.hpp
- [ ] src/batch_buffer.cpp

## Phase 5 — Training Thread
- [ ] include/adaptive_batcher/training_thread.hpp
- [ ] src/training_thread.cpp
- [ ] tests/test_rcu_weights.cpp

## Phase 6 — Top-Level Orchestrator
- [ ] include/adaptive_batcher/adaptive_batcher.hpp
- [ ] src/adaptive_batcher.cpp

## Phase 7 — Simulation Layer
- [ ] simulation/regime_generator.cpp
- [ ] simulation/market_simulator.cpp
- [ ] simulation/latency_oracle.cpp

## Phase 8 — Benchmarks
- [ ] benchmarks/bench_feature_extract.cpp
- [ ] benchmarks/bench_policy_inference.cpp
- [ ] benchmarks/bench_spsc_queue.cpp
- [ ] benchmarks/bench_vs_fixed.cpp

## Phase 9 — Tests
- [ ] tests/test_safe_exploration.cpp
- [ ] Integration test: 10k synthetic requests

## Phase 10 — Analysis Scripts
- [ ] analysis/plot_latency_cdf.py
- [ ] analysis/plot_policy_decisions.py
- [ ] analysis/plot_regime_adaptation.py
- [ ] analysis/plot_reward_convergence.py
