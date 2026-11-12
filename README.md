# TinyInfer

A small C++/CUDA inference project with CPU and GPU tensors, matrix multiplication,
ReLU, Linear layers, and a two-layer MLP.

## Build and test

Requires CUDA, a C++20 compiler, CMake 3.28+, and GoogleTest.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

## Handwritten digits

The example loads a trained `64 -> 64 -> 10` MLP and predicts handwritten digits
from the UCI 8 x 8 digit dataset. Run from the project root:

```bash
./build/digits_example
```

The included model gets **95.94% test accuracy (1,724 / 1,797)**.
The weights and test data are included. To retrain, install NumPy and run:

```bash
OPENBLAS_NUM_THREADS=1 python3 examples/digits/train.py
```

The trainer uses ordinary minibatch gradient descent on the training split, then
prints accuracy on the separate test split. See [dataset credit](examples/digits/DATASET.md).

## Model files

`MLP::save(path)` writes the three layer sizes, then `W1`, `b1`, `W2`, and `b2` as
plain text. `MLP::load(path, device)` reads them into a model on CPU or CUDA.

`matmul` allocates a result; `matmulInto` writes into an existing output tensor.
