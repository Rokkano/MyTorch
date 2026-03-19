#include "tensor.hh"

bool testTensorCreation(std::vector<std::size_t> shape)
{
    std::vector<float> buffer = std::vector<float>(128);
    Tensor<float> tensor = Tensor<float>(shape, buffer);
    
    ASSERT(tensor.numel() == 128);

    for (std::size_t i = 0; i < shape.size(); i++)
        ASSERT(tensor.shape()[i] == shape[i]);

    return true;
}
PARAMETRIZE(testTensorCreation, std::vector<size_t>{128})
PARAMETRIZE(testTensorCreation, std::vector<size_t>{2, 64})
PARAMETRIZE(testTensorCreation, std::vector<size_t>{2, 32, 2})
PARAMETRIZE(testTensorCreation, std::vector<size_t>{2, 2, 16, 2})
PARAMETRIZE(testTensorCreation, std::vector<size_t>{2, 2, 8, 2, 2})
PARAMETRIZE(testTensorCreation, std::vector<size_t>{2, 2, 2, 4, 2, 2})
PARAMETRIZE(testTensorCreation, std::vector<size_t>{2, 2, 2, 2, 2, 2, 2})
PARAMETRIZE(testTensorCreation, std::vector<size_t>{1, 2, 2, 2, 2, 2, 2, 2})
PARAMETRIZE(testTensorCreation, std::vector<size_t>{1, 2, 2, 2, 2, 2, 2, 2, 1})