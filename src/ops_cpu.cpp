#include "tinyinfer/ops.hpp"

namespace tinyinfer
{

    Tensor reluCUDA(const Tensor &input);
    Tensor reluCPU(const Tensor &input);

    Tensor relu(const Tensor &input)
    {
        if (input.device() == Device::CPU)
        {
            return reluCPU(input);
        }
        else
        {
            return reluCUDA(input);
        }
    }

    Tensor reluCPU(const Tensor &input)
    {
        Tensor T{input.shape(), Device::CPU};
        const float *dataI = input.data();
        float *dataT = T.data();
        for (std::size_t i = 0; i < input.numel(); i++)
        {
            dataT[i] = dataI[i] > 0 ? dataI[i] : 0;
        }
        
        return T;
    }

}
