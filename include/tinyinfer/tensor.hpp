#pragma once
#include <vector>
#include <cstddef>
#include "device.hpp"
#include <memory>

namespace tinyinfer
{

    class Tensor
    {
    public:
        Tensor(std::vector<std::size_t> shape, Device device);

        Device device() const;
        size_t numel() const;
        const std::vector<std::size_t>& shape() const;
        float *data();
        const float *data() const;

    private:
        std::vector<std::size_t> shape_;
        Device device_;
        std::size_t numel_;
        std::unique_ptr<float[]> data_;
    };
}
