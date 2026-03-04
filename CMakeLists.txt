cmake_minimum_required(VERSION 3.20)
project(AdaptiveBatcher VERSION 0.1.0 LANGUAGES CXX)

# ─── C++ standard ─────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Build options ────────────────────────────────────────────────────────────
option(ENABLE_TSAN  "Build with ThreadSanitizer"  OFF)
option(ENABLE_ASAN  "Build with AddressSanitizer" OFF)
option(ENABLE_UBSAN "Build with UBSanitizer"      OFF)

# ─── Compiler flags ───────────────────────────────────────────────────────────
add_compile_options(
    -Wall
    -Wextra
    -Wpedantic
    -Wno-unused-parameter
)

if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    add_compile_options(-O3 -ffast-math)
endif()

# ─── Sanitizer flags ──────────────────────────────────────────────────────────
if(ENABLE_ASAN AND ENABLE_TSAN)
    message(FATAL_ERROR "ASAN and TSAN cannot be enabled simultaneously")
endif()

if(ENABLE_ASAN)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

if(ENABLE_TSAN)
    add_compile_options(-fsanitize=thread -fno-omit-frame-pointer)
    add_link_options(-fsanitize=thread)
endif()

if(ENABLE_UBSAN AND NOT ENABLE_ASAN)
    add_compile_options(-fsanitize=undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=undefined)
endif()

# ─── AVX2 detection ───────────────────────────────────────────────────────────
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
if(COMPILER_SUPPORTS_AVX2)
    add_compile_options(-mavx2)
    add_compile_definitions(ADAPTIVE_BATCHER_AVX2=1)
    message(STATUS "AVX2 enabled")
else()
    message(STATUS "AVX2 not supported; using scalar fallback")
endif()

# ─── VectorBook integration ───────────────────────────────────────────────────
# Build VectorBook sources directly to avoid target name conflicts (VectorBook
# defines an executable named "benchmark" that clashes with Google Benchmark).
set(VB_DIR "${CMAKE_SOURCE_DIR}/../VectorBook")
add_library(vectorbook STATIC
    "${VB_DIR}/src/order_book.cpp"
    "${VB_DIR}/src/itch_parser.cpp"
)
target_include_directories(vectorbook PUBLIC "${VB_DIR}/include")
target_compile_features(vectorbook PUBLIC cxx_std_20)
# Forward AVX2 flag to VectorBook
if(COMPILER_SUPPORTS_AVX2)
    target_compile_options(vectorbook PRIVATE -mavx2)
    target_compile_definitions(vectorbook PUBLIC VECTORBOOK_AVX2=1)
endif()

# ─── Google Test via FetchContent ─────────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# ─── Google Benchmark via FetchContent ────────────────────────────────────────
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# ─── AdaptiveBatcher library ─────────────────────────────────────────────────
add_library(adaptive_batcher STATIC
    src/market_features.cpp
    src/policy_engine.cpp
    src/batch_buffer.cpp
    src/training_thread.cpp
    src/adaptive_batcher.cpp
)

target_include_directories(adaptive_batcher PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(adaptive_batcher PUBLIC vectorbook)
target_compile_features(adaptive_batcher PUBLIC cxx_std_20)

# ─── Simulation executables ───────────────────────────────────────────────────
add_executable(regime_generator simulation/regime_generator.cpp)
target_link_libraries(regime_generator PRIVATE adaptive_batcher)

add_executable(market_simulator simulation/market_simulator.cpp)
target_link_libraries(market_simulator PRIVATE adaptive_batcher)

add_executable(validate_adaptive simulation/validate_adaptive.cpp)
target_link_libraries(validate_adaptive PRIVATE adaptive_batcher)
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    target_compile_options(validate_adaptive PRIVATE -O3 -ffast-math)
endif()

# ─── Test infrastructure ──────────────────────────────────────────────────────
enable_testing()
include(GoogleTest)

# Usage: add_ab_test(test_foo tests/test_foo.cpp)
function(add_ab_test target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE adaptive_batcher GTest::gtest_main)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    gtest_discover_tests(${target})
endfunction()

# Usage: add_ab_bench(bench_foo benchmarks/bench_foo.cpp)
function(add_ab_bench target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE adaptive_batcher benchmark::benchmark)
    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        target_compile_options(${target} PRIVATE -O3 -ffast-math)
    endif()
endfunction()

# ─── Tests ────────────────────────────────────────────────────────────────────
add_ab_test(test_spsc_queue          tests/test_spsc_queue.cpp)
add_ab_test(test_feature_extraction  tests/test_feature_extraction.cpp)
add_ab_test(test_policy_correctness  tests/test_policy_correctness.cpp)
add_ab_test(test_safe_exploration    tests/test_safe_exploration.cpp)
add_ab_test(test_rcu_weights         tests/test_rcu_weights.cpp)

# ─── Benchmarks ───────────────────────────────────────────────────────────────
add_ab_bench(bench_feature_extract   benchmarks/bench_feature_extract.cpp)
add_ab_bench(bench_policy_inference  benchmarks/bench_policy_inference.cpp)
add_ab_bench(bench_spsc_queue        benchmarks/bench_spsc_queue.cpp)
add_ab_bench(bench_vs_fixed          benchmarks/bench_vs_fixed.cpp)
