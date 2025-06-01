#include <criterion/criterion.h>
#include <criterion/parameterized.h>
#include <criterion/new/assert.h>
#include <criterion/logging.h>

#include <iostream>
#include <vector>

#include "../../src/tensor/tensor.hh"

using criterion::logging::error;
using criterion::logging::info;
using criterion::logging::warn;

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

struct tensor_broadcast_params_c
{
    criterion::parameters<size_t> shape;
    criterion::parameters<size_t> target;
    criterion::parameters<size_t> expected;
};

struct tensor_broadcast_params
{
    std::vector<size_t> shape;
    std::vector<size_t> target;
    std::vector<size_t> expected;
};

ParameterizedTestParameters(tensor, tensor_broadcast)
{
    // shape, target, expected
    static struct tensor_broadcast_params_c params[] = {
        {{2, 1, 3}, {2, 3, 3}, {2, 3, 3}},
        {{2, 1, 1}, {2, 3, 3}, {2, 3, 3}},
    };

    return criterion_test_params(params);
}

ParameterizedTest(struct tensor_broadcast_params *param, tensor, tensor_broadcast)
{
    auto identity = [](int x)
    { return x; };
    Tensor<int> tensor = Tensor<int>::from_function(identity, param->shape);
    Tensor<int> target = Tensor<int>(param->target);

    Tensor<int> b_tensor = tensor.broadcast(target);

    for (size_t i = 0; i < b_tensor.shape().size(); i++)
        cr_assert(b_tensor.shape()[i] == param->expected[i]);
    info << b_tensor << std::endl;
}

// cr_assert_throw(2 + 2, std::exception)
