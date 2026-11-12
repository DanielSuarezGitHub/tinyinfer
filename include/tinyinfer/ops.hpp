#pragma once
#include "tensor.hpp"

namespace tinyinfer {

    Tensor relu(const Tensor& input);
    Tensor add(const Tensor& a, const Tensor& b);
    Tensor matmul(const Tensor& a, const Tensor& b);
    void matmulInto(const Tensor& a, const Tensor& b, Tensor& output);
    Tensor addBias(const Tensor& a, const Tensor& b);
}
