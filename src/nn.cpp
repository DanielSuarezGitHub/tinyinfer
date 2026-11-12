

#include "tinyinfer/nn.hpp"
#include "tinyinfer/ops.hpp"

#include <cmath>
#include <random>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace tinyinfer
{

    Linear::Linear(std::size_t inFeatures,
                   std::size_t outFeatures,
                   Device device)
        : weights_({inFeatures, outFeatures}, Device::CPU),
          bias_({outFeatures}, Device::CPU)
    {
        std::mt19937 gen(42);

        std::normal_distribution<float> dist(
            0.0f, std::sqrt(2.0f / inFeatures));

        for (std::size_t i = 0; i < weights_.numel(); ++i)
        {
            weights_.data()[i] = dist(gen);
        }

        for (std::size_t i = 0; i < bias_.numel(); ++i)
        {
            bias_.data()[i] = 0.0f;
        }

        if (device == Device::CUDA)
        {
            weights_ = weights_.to(Device::CUDA);
            bias_ = bias_.to(Device::CUDA);
        }
    }

    Tensor Linear::forward(const Tensor &input) const
    {
        Tensor output = matmul(input, weights_);
        return addBias(output, bias_);
    }

    MLP::MLP(std::size_t inputSize,
             std::size_t hiddenSize,
             std::size_t outputSize,
             Device device)
        : fc1_(inputSize, hiddenSize, device),
          fc2_(hiddenSize, outputSize, device)
    {
    }

    Tensor MLP::forward(const Tensor &input) const
    {
        Tensor hidden = relu(fc1_.forward(input));
        return fc2_.forward(hidden);
    }


    void MLP::save(const std::string &path) const
    {
        std::ofstream file(path);
        if (!file)
            throw std::runtime_error("could not open model file");

        file << fc1_.weights().shape()[0] << ' '
             << fc1_.weights().shape()[1] << ' '
             << fc2_.weights().shape()[1] << '\n';
        file << std::setprecision(9);

        for (const Tensor *tensor : {&fc1_.weights(), &fc1_.bias(), &fc2_.weights(), &fc2_.bias()})
        {
            Tensor values = tensor->to(Device::CPU);
            for (std::size_t i = 0; i < values.numel(); ++i)
                file << values.data()[i] << ' ';
            file << '\n';
        }
        if (!file)
            throw std::runtime_error("could not write model file");
    }

    MLP MLP::load(const std::string &path, Device device)
    {
        std::ifstream file(path);
        int inputSize, hiddenSize, outputSize;
        if (!(file >> inputSize >> hiddenSize >> outputSize) ||
            inputSize <= 0 || hiddenSize <= 0 || outputSize <= 0 ||
            inputSize > 4096 || hiddenSize > 4096 || outputSize > 4096)
            throw std::runtime_error("invalid model dimensions");

        MLP model(inputSize, hiddenSize, outputSize, Device::CPU);
        for (Tensor *tensor : {&model.fc1_.weights(), &model.fc1_.bias(),
                               &model.fc2_.weights(), &model.fc2_.bias()})
        {
            for (std::size_t i = 0; i < tensor->numel(); ++i)
                if (!(file >> tensor->data()[i]))
                    throw std::runtime_error("missing model weights");
            if (device == Device::CUDA)
                *tensor = tensor->to(Device::CUDA);
        }
        return model;
    }

}
