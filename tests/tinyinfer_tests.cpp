#include <gtest/gtest.h>

#include "tinyinfer/tensor.hpp"

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
    float* data = t.data();

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
    float* original = source.data();

    Tensor dest{std::move(source)};

    EXPECT_EQ(dest.data(), original);
    EXPECT_EQ(dest.device(), Device::CPU);
    EXPECT_EQ(dest.numel(), 6);
    EXPECT_EQ(dest.shape(), (std::vector<std::size_t>{2, 3}));
}

TEST(TensorTest, MoveConstructsCudaTensor)
{
    Tensor source{{2, 3}, Device::CUDA};
    float* original = source.data();

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

    float* original = source.data();

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

    float* original = source.data();

    dest = std::move(source);

    EXPECT_EQ(dest.data(), original);
    EXPECT_EQ(dest.device(), Device::CUDA);
    EXPECT_EQ(dest.numel(), 6);
    EXPECT_EQ(dest.shape(), (std::vector<std::size_t>{2, 3}));
}