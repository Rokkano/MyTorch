#include <criterion/criterion.h>
#include <criterion/parameterized.h>
#include <criterion/new/assert.h>

#include <iostream>
#include <vector>

#include "../../src/tensor/tensor.hh"

ParameterizedTestParameters(tensor, tensor_creation)
{
    static std::vector<size_t, criterion::allocator<size_t>> params[] = {
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

ParameterizedTest(std::vector<size_t> *param, tensor, tensor_creation)
{
    std::vector<float> buffer = std::vector<float>(128);
    Tensor<float> tensor = Tensor<float>(*param, buffer);
}

// cr_assert_throw(2 + 2, std::exception)
