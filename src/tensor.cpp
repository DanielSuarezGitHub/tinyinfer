#include "tinyinfer/tensor.hpp"

#include <cuda_runtime.h>

#include <functional>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <utility>

namespace tinyinfer
{

    TensorDeleter::TensorDeleter(Device device)
        : device{device}
    {
    }

    void TensorDeleter::operator()(float *ptr)
    {
        if (device == Device::CUDA)
        {
            cudaFree(ptr);
        }
        else if (device == Device::CPU)
        {
            delete[] ptr;
        }
    }

    Tensor::Tensor(std::vector<std::size_t> shape, Device device)
        : shape_(std::move(shape)),
          device_(device),
          numel_(std::accumulate(
              shape_.begin(),
              shape_.end(),
              std::size_t{1},
              std::multiplies<>{})),
          data_{
              nullptr,
              TensorDeleter{device_}}
    {
        if (std::ranges::find(shape_, 0) != std::ranges::end(shape_))
        {
            throw std::runtime_error("0 dim not supported");
        }

        if (device_ == Device::CUDA)
        {
            float *raw{nullptr};

            cudaMalloc(
                reinterpret_cast<void **>(&raw),
                numel_ * sizeof(float));

            data_.reset(raw);
        }
        else
        {
            data_.reset(new float[numel_]);
        }
    }

    Device Tensor::device() const
    {
        return device_;
    }

    std::size_t Tensor::numel() const
    {
        return numel_;
    }

    const std::vector<std::size_t> &Tensor::shape() const
    {
        return shape_;
    }

    float *Tensor::data()
    {
        return data_.get();
    }

    const float *Tensor::data() const
    {
        return data_.get();
    }

    Tensor Tensor::to(Device device) const
    {
        Tensor result{shape_, device};

        if (device_ == Device::CPU)
        {
            if (device == Device::CPU)
            {
                std::copy(
                    data_.get(),
                    data_.get() + numel_,
                    result.data_.get());
            }
            else
            {
                cudaMemcpy(
                    result.data_.get(),
                    data_.get(),
                    numel_ * sizeof(float),
                    cudaMemcpyHostToDevice);
            }
        }
        else
        {
            if (device == Device::CPU)
            {
                cudaMemcpy(
                    result.data_.get(),
                    data_.get(),
                    numel_ * sizeof(float),
                    cudaMemcpyDeviceToHost);
            }
            else
            {
                cudaMemcpy(
                    result.data_.get(),
                    data_.get(),
                    numel_ * sizeof(float),
                    cudaMemcpyDeviceToDevice);
            }
        }

        return result;
    }
}