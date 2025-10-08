#include "tensor.hh"

ParameterizedTestParameters(tensor, tensor_broadcast)
{
    static tensor_broadcast_params_c params[] = {
        {{2, 1, 3}, {2, 3, 3}, {2, 3, 3}},
        {{2, 1, 1}, {2, 3, 3}, {2, 3, 3}},
    };

    return criterion_test_params(params);
}

ParameterizedTest(tensor_broadcast_params *param, tensor, tensor_broadcast)
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
