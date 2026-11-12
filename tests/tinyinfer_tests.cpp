#include <gtest/gtest.h>
#include <fstream>
#include <cstdio>

#include "tinyinfer/tensor.hpp"
#include "tinyinfer/ops.hpp"
#include "tinyinfer/nn.hpp"

using tinyinfer::Device;
using tinyinfer::Tensor;
TEST(TensorTest, ComputesNumel2D)
{
    Tensor t{{2, 3}, Device::CPU};
    EXPECT_EQ(t.numel(), 6);
}

TEST(TensorTest, ComputesNumel3D)
{
    Tensor t{{2, 3, 4}, Device::CPU};
    EXPECT_EQ(t.numel(), 24);
}

TEST(TensorTest, ScalarHasOneElement)
{
    Tensor t{{}, Device::CPU};
    EXPECT_EQ(t.numel(), 1);
}

TEST(TensorTest, PreservesShape)
{
    Tensor t{{2, 3, 4}, Device::CPU};
    std::vector<std::size_t> expected{2, 3, 4};

    EXPECT_EQ(t.shape(), expected);
}

TEST(TensorTest, StoresDevice)
{
    Tensor t{{2, 3, 4}, Device::CPU};
    EXPECT_EQ(t.device(), Device::CPU);
}

TEST(TensorTest, DataIsReadableAndWritable)
{
    Tensor t{{2, 3}, Device::CPU};
    float *data = t.data();

    ASSERT_NE(data, nullptr);

    data[0] = 5.0f;
    data[1] = -3.5f;
    data[5] = 42.0f;

    EXPECT_FLOAT_EQ(data[0], 5.0f);
    EXPECT_FLOAT_EQ(data[1], -3.5f);
    EXPECT_FLOAT_EQ(data[5], 42.0f);
}

TEST(TensorTest, RejectsZeroDimension)
{
    EXPECT_THROW(Tensor({2, 0, 3}, Device::CPU), std::runtime_error);
}

TEST(TensorTest, ConstructsCudaTensor)
{
    Tensor t{{2, 3}, Device::CUDA};

    EXPECT_EQ(t.device(), Device::CUDA);
    EXPECT_EQ(t.numel(), 6);
    EXPECT_EQ(t.shape(), (std::vector<std::size_t>{2, 3}));
    EXPECT_NE(t.data(), nullptr);
}

TEST(TensorTest, MoveConstructsCpuTensor)
{
    Tensor source{{2, 3}, Device::CPU};
    float *original = source.data();

    Tensor dest{std::move(source)};

    EXPECT_EQ(dest.data(), original);
    EXPECT_EQ(dest.device(), Device::CPU);
    EXPECT_EQ(dest.numel(), 6);
    EXPECT_EQ(dest.shape(), (std::vector<std::size_t>{2, 3}));
}

TEST(TensorTest, MoveConstructsCudaTensor)
{
    Tensor source{{2, 3}, Device::CUDA};
    float *original = source.data();

    Tensor dest{std::move(source)};

    EXPECT_EQ(dest.data(), original);
    EXPECT_EQ(dest.device(), Device::CUDA);
    EXPECT_EQ(dest.numel(), 6);
    EXPECT_EQ(dest.shape(), (std::vector<std::size_t>{2, 3}));
}

TEST(TensorTest, MoveAssignsCpuTensor)
{
    Tensor source{{2, 3}, Device::CPU};
    Tensor dest{{4, 4}, Device::CPU};

    float *original = source.data();

    dest = std::move(source);

    EXPECT_EQ(dest.data(), original);
    EXPECT_EQ(dest.device(), Device::CPU);
    EXPECT_EQ(dest.numel(), 6);
    EXPECT_EQ(dest.shape(), (std::vector<std::size_t>{2, 3}));
}

TEST(TensorTest, MoveAssignsCudaTensor)
{
    Tensor source{{2, 3}, Device::CUDA};
    Tensor dest{{4, 4}, Device::CUDA};

    float *original = source.data();

    dest = std::move(source);

    EXPECT_EQ(dest.data(), original);
    EXPECT_EQ(dest.device(), Device::CUDA);
    EXPECT_EQ(dest.numel(), 6);
    EXPECT_EQ(dest.shape(), (std::vector<std::size_t>{2, 3}));
}

TEST(TensorTest, ProperlyCopiesMemory)
{
    Tensor source{{2, 3}, Device::CPU};

    float *data = source.data();

    for (int i = 0; i < 6; ++i)
    {
        data[i] = static_cast<float>(i + 1);
    }

    Tensor deviceTensor = source.to(Device::CUDA);
    Tensor onHost = deviceTensor.to(Device::CPU);

    for (int i = 0; i < 6; ++i)
    {
        EXPECT_EQ(onHost.data()[i], static_cast<float>(i + 1));
        EXPECT_EQ(source.data()[i], static_cast<float>(i + 1));
    }
}

TEST(TensorTest, reluExecutesProperly)
{
    Tensor source{{2,3}, Device::CPU};
     float *data = source.data();
     for (int i = 0; i < 6; ++i)
    {
        data[i] = static_cast<float>(i -3);
    }
    // [-3, -2, -1, 0 1, 2] -> [0,0,0,0,1,2]
    int nums[6] = {0,0,0,0,1,2};
    Tensor opCPU = tinyinfer::relu(source);
    Tensor onGPU = source.to(Device::CUDA);
    Tensor opGPU = tinyinfer::relu(onGPU);
    Tensor final = opGPU.to(Device::CPU);

    for (int i = 0; i < 6; i++) {
        EXPECT_EQ(opCPU.data()[i], nums[i]);
        EXPECT_EQ(final.data()[i], nums[i]);
    }


}


TEST(TensorTest, AddWorks)
{
    Tensor a{{2, 3}, Device::CPU};
    Tensor b{{2, 3}, Device::CPU};

    float aValues[] = {1, 2, 3, 4, 5, 6};
    float bValues[] = {6, 5, 4, 3, 2, 1};
    float expected[] = {7, 7, 7, 7, 7, 7};

    std::copy(std::begin(aValues), std::end(aValues), a.data());
    std::copy(std::begin(bValues), std::end(bValues), b.data());

    Tensor cpuResult = tinyinfer::add(a, b);

    Tensor aGPU = a.to(Device::CUDA);
    Tensor bGPU = b.to(Device::CUDA);

    Tensor gpuResult = tinyinfer::add(aGPU, bGPU);
    Tensor gpuResultCPU = gpuResult.to(Device::CPU);

    for (std::size_t i = 0; i < a.numel(); ++i)
    {
        EXPECT_FLOAT_EQ(cpuResult.data()[i], expected[i]);
        EXPECT_FLOAT_EQ(gpuResultCPU.data()[i], expected[i]);
    }
}



TEST(TensorTest, MatmulWorks)
{
    Tensor a{{2, 3}, Device::CPU};
    Tensor b{{3, 2}, Device::CPU};

    float aValues[] = {1, 2, 3, 4, 5, 6};
    float bValues[] = {7, 8, 9, 10, 11, 12};
    float expected[] = {58, 64, 139, 154};

    std::copy(std::begin(aValues), std::end(aValues), a.data());
    std::copy(std::begin(bValues), std::end(bValues), b.data());

    Tensor cpuResult = tinyinfer::matmul(a, b);

    Tensor aGPU = a.to(Device::CUDA);
    Tensor bGPU = b.to(Device::CUDA);

    Tensor gpuResult = tinyinfer::matmul(aGPU, bGPU);
    Tensor gpuResultCPU = gpuResult.to(Device::CPU);

    EXPECT_EQ(cpuResult.shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_EQ(gpuResult.shape(), (std::vector<std::size_t>{2, 2}));

    for (std::size_t i = 0; i < 4; ++i)
    {
        EXPECT_FLOAT_EQ(cpuResult.data()[i], expected[i]);
        EXPECT_FLOAT_EQ(gpuResultCPU.data()[i], expected[i]);
    }
}


TEST(TensorTest, AddBiasWorks)
{
    Tensor a{{2, 3}, Device::CPU};
    Tensor bias{{3}, Device::CPU};

    float values[] = {1, 2, 3, 4, 5, 6};
    float biases[] = {10, 20, 30};
    float expected[] = {11, 22, 33, 14, 25, 36};

    std::copy(std::begin(values), std::end(values), a.data());
    std::copy(std::begin(biases), std::end(biases), bias.data());

    Tensor cpuResult = tinyinfer::addBias(a, bias);

    Tensor aGPU = a.to(Device::CUDA);
    Tensor biasGPU = bias.to(Device::CUDA);

    Tensor gpuResult = tinyinfer::addBias(aGPU, biasGPU).to(Device::CPU);

    for (std::size_t i = 0; i < 6; ++i)
    {
        EXPECT_FLOAT_EQ(cpuResult.data()[i], expected[i]);
        EXPECT_FLOAT_EQ(gpuResult.data()[i], expected[i]);
    }
}

TEST(TensorTest, MLPWorks)
{
    tinyinfer::MLP cpuModel(4, 8, 2, Device::CPU);
    tinyinfer::MLP gpuModel(4, 8, 2, Device::CUDA);

    Tensor input{{3, 4}, Device::CPU};
    std::fill_n(input.data(), input.numel(), 1.0f);

    Tensor cpuOutput = cpuModel.forward(input);
    Tensor gpuOutput = gpuModel.forward(input.to(Device::CUDA))
                              .to(Device::CPU);

    EXPECT_EQ(cpuOutput.shape(), (std::vector<std::size_t>{3, 2}));

    for (std::size_t i = 0; i < cpuOutput.numel(); ++i)
    {
        EXPECT_NEAR(cpuOutput.data()[i], gpuOutput.data()[i], 1e-4f);
    }
}
TEST(TensorTest, MatmulIntoWorks)
{
    Tensor a({2, 2}, Device::CPU);
    Tensor b({2, 2}, Device::CPU);
    Tensor output({2, 2}, Device::CPU);
    std::fill_n(a.data(), 4, 2.0f);
    std::fill_n(b.data(), 4, 3.0f);
    float *allocation = output.data();
    tinyinfer::matmulInto(a, b, output);
    EXPECT_EQ(output.data(), allocation);
    for (int i = 0; i < 4; ++i)
        EXPECT_FLOAT_EQ(output.data()[i], 12.0f);
}

TEST(TensorTest, RejectsBadShapes)
{
    Tensor matrix({2, 3}, Device::CPU);
    Tensor vector({3}, Device::CPU);
    EXPECT_THROW(tinyinfer::matmul(matrix, matrix), std::runtime_error);
    EXPECT_THROW(tinyinfer::addBias(vector, vector), std::runtime_error);
}

TEST(TensorTest, ModelSaveLoadWorks)
{
    {
        std::ofstream file("test_model.txt");
        file << "1 1 1\n2\n1\n3\n4\n";
    }
    auto model = tinyinfer::MLP::load("test_model.txt", Device::CPU);
    Tensor input({2, 1}, Device::CPU);
    input.data()[0] = -2;
    input.data()[1] = 2;
    Tensor output = model.forward(input);
    EXPECT_FLOAT_EQ(output.data()[0], 4);
    EXPECT_FLOAT_EQ(output.data()[1], 19);
    model.save("test_model_saved.txt");
    auto loaded = tinyinfer::MLP::load("test_model_saved.txt", Device::CUDA);
    Tensor gpuOutput = loaded.forward(input.to(Device::CUDA)).to(Device::CPU);
    EXPECT_NEAR(gpuOutput.data()[0], 4, 1e-5f);
    EXPECT_NEAR(gpuOutput.data()[1], 19, 1e-5f);
    std::remove("test_model.txt");
    std::remove("test_model_saved.txt");
}
