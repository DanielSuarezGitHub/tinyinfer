# TinyInfer

Tiny C++20 + CUDA inference runtime.

## Configure

Run from the project root:

```bash
cmake -S . -B build -G Ninja
```

Run this initially or after changing the CMake configuration.

## Build

```bash
cmake --build build
```

## Run

```bash
./build/tinyinfer_example
```

## Test

```bash
ctest --test-dir build --output-on-failure
```

## Build + Test

```bash
cmake --build build && ctest --test-dir build --output-on-failure
```

## Clean Build

If the build directory gets messed up or you want to configure everything from scratch:

```bash
rm -rf build
cmake -S . -B build -G Ninja
cmake --build build
```

## Toolchain Checks

```bash
nvidia-smi
nvcc --version
g++ --version
cmake --version
ninja --version
```

## Project Structure

```text
tinyinfer/
├── CMakeLists.txt
├── README.md
├── include/tinyinfer/
├── src/
├── cuda/
├── tests/
├── bench/
└── examples/
```

## Development Loop

```text
change code
    ↓
cmake --build build
    ↓
ctest --test-dir build --output-on-failure
    ↓
./build/tinyinfer_example
```
