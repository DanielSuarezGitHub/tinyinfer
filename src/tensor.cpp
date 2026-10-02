#include "tinyinfer/tensor.hpp"

#include <functional>
#include <memory>
#include <numeric>
#include <utility>
#include <stdexcept>
#include <ranges>
namespace tinyinfer
{

    Tensor::Tensor(std::vector<std::size_t> shape, Device device)
        : shape_(std::move(shape)),
          device_(device),
          numel_(std::accumulate(
              shape_.begin(),
              shape_.end(),
              std::size_t{1},
              std::multiplies<>{})),
          data_{nullptr}
    {
        if (device_ == Device::CUDA) {
            throw std::runtime_error("CUDA device not supported yet");
        }

        if (std::ranges::find(shape_, 0) != std::ranges::end(shape_)) {
            throw std::runtime_error("0 dim not supported");
        }

        data_ = std::make_unique<float[]>(numel_);


    }

    Device Tensor::device() const
    {
        return this->device_;
    }

    std::size_t Tensor::numel() const
    {
        return this->numel_;
    }

    const std::vector<std::size_t> &Tensor::shape() const
    {
        return this->shape_;
    }

    float *Tensor::data()
    {
        return data_.get();
    }
    const float *Tensor::data() const
    {
        return data_.get();
    }

}