cmake_minimum_required(VERSION 3.28)
project(tinyinfer LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_library(tinyinfer
    src/tensor.cpp
    src/ops_cpu.cpp
    src/nn.cpp
    cuda/ops_cuda.cu
)
target_include_directories(tinyinfer PUBLIC include)

add_executable(tinyinfer_example
    examples/main.cpp
)
target_link_libraries(tinyinfer_example PRIVATE tinyinfer)

enable_testing()

find_package(GTest REQUIRED)

add_executable(tinyinfer_tests
    tests/tinyinfer_tests.cpp
)

target_link_libraries(tinyinfer_tests
    PRIVATE
    tinyinfer
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(tinyinfer_tests)