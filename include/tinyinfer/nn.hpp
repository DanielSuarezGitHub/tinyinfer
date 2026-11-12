#pragma once

#include "tinyinfer/tensor.hpp"
#include <string>

namespace tinyinfer
{

    class Linear
    {
    public:
        Linear(std::size_t inFeatures,
               std::size_t outFeatures,
               Device device);

        Tensor forward(const Tensor &input) const;

        Tensor &weights() { return weights_; }
        Tensor &bias() { return bias_; }
        const Tensor &weights() const { return weights_; }
        const Tensor &bias() const { return bias_; }

    private:
        Tensor weights_;
        Tensor bias_;
    };

    class MLP
    {
    public:
        MLP(std::size_t inputSize,
            std::size_t hiddenSize,
            std::size_t outputSize,
            Device device);

        Tensor forward(const Tensor &input) const;
        void save(const std::string &path) const;
        static MLP load(const std::string &path, Device device);

    private:
        Linear fc1_;
        Linear fc2_;
    };

}
