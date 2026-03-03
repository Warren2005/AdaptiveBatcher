# AdaptiveBatcher — Lessons Learned

## VectorBook Integration
- VectorBook's EpochRcu uses thread-local ReaderState auto-registered on first read_acquire()
- OrderBook<Depth> template instantiated in order_book.cpp (extern template declarations in header)
- VectorBook CMake target: `vectorbook`, adds include paths via PUBLIC target_include_directories

## Build System
- add_subdirectory with EXCLUDE_FROM_ALL avoids building VectorBook tests/benchmarks from AB's build
- GTest v1.14.0 fetched via FetchContent; Google Benchmark fetched similarly

## AVX2 Dot Product Notes
- MarketFeatures uses float v[16] with [10..15] zeroed — two _mm256_loadu_ps loads cover all 16
- _mm256_mul_ps + horizontal add is faster than _mm256_dp_ps for full-vector dot products
- On Apple Silicon (arm64), -mavx2 unavailable; scalar fallback required

## SPSC Queue Design
- head_ and tail_ on separate cache lines prevents false sharing
- Capacity must be power-of-2 for modular index via bitmasking
- is_full() check: (tail_ - head_) >= Capacity

## Training Thread
- SGD update: w_a[i] += η * (reward - w_a·features) * features[i]
- Constant η=0.01 (no decay) because market is non-stationary
- Experience stale after 1s: discard without SGD update
