#include "tensor.hh"

ParameterizedTestParameters(tensor, tensor_creation)
{
    static std::vector<std::size_t, criterion::allocator<std::size_t>> params[] = {
        {128},
        {2, 64},
        {2, 32, 2},
        {2, 2, 16, 2},
        {2, 2, 8, 2, 2},
        {2, 2, 2, 4, 2, 2},
        {2, 2, 2, 2, 2, 2, 2},
    };

    return criterion_test_params(params);
}

ParameterizedTest(std::vector<std::size_t> *param, tensor, tensor_creation)
{
    std::vector<float> buffer = std::vector<float>(128);
    Tensor<float> tensor = Tensor<float>(*param, buffer);
}