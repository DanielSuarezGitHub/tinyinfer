#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "device.hpp"

namespace tinyinfer
{

struct TensorDeleter
{
    Device device;

    explicit TensorDeleter(Device device);
    void operator()(float* ptr);
};

class Tensor
{
public:
    Tensor(std::vector<std::size_t> shape, Device device);

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    Tensor to(Device device) const;
    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;

    Device device() const;
    std::size_t numel() const;
    const std::vector<std::size_t>& shape() const;

    float* data();
    const float* data() const;

private:
    std::vector<std::size_t> shape_;
    Device device_;
    std::size_t numel_;

    std::unique_ptr<float[], TensorDeleter> data_;
};

}