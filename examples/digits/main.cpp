#include "tinyinfer/nn.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace tinyinfer;

int main()
{
    try
    {
        std::ifstream file("examples/digits/test.csv");
        if (!file)
            throw std::runtime_error("could not open test data; run from the project root");

        std::vector<float> pixels;
        std::vector<int> labels;
        std::string line;
        while (std::getline(file, line))
        {
            std::replace(line.begin(), line.end(), ',', ' ');
            std::istringstream row(line);
            for (int i = 0; i < 64; ++i)
            {
                float pixel;
                if (!(row >> pixel) || pixel < 0 || pixel > 16)
                    throw std::runtime_error("invalid pixel");
                pixels.push_back(pixel / 16.0f);
            }
            int label;
            if (!(row >> label) || label < 0 || label > 9)
                throw std::runtime_error("invalid label");
            labels.push_back(label);
        }
        if (labels.empty())
            throw std::runtime_error("test data is empty");

        Tensor input({labels.size(), 64}, Device::CPU);
        std::copy(pixels.begin(), pixels.end(), input.data());
        auto model = MLP::load("examples/digits/model.txt", Device::CUDA);
        Tensor output = model.forward(input.to(Device::CUDA)).to(Device::CPU);
        if (output.shape() != std::vector<std::size_t>{labels.size(), 10})
            throw std::runtime_error("expected ten digit scores per example");

        int correct = 0;
        for (std::size_t i = 0; i < labels.size(); ++i)
        {
            const float *scores = output.data() + i * 10;
            int prediction = std::max_element(scores, scores + 10) - scores;
            if (prediction == labels[i])
                ++correct;
        }
        std::cout << "Accuracy: " << 100.0 * correct / labels.size() << "% ("
                  << correct << '/' << labels.size() << ")\n";
    }
    catch (const std::exception &error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
