#pragma once
#include "tensor.hpp"

namespace tinyinfer {

    Tensor relu(const Tensor& input);
    Tensor add(const Tensor& a, const Tensor& b);
    Tensor mamtul(const Tensor& a, const Tensor& b);
}