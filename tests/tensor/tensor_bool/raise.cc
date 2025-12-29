#include "../tensor.hh"
#include "../../utils.hh"

ParameterizedTestParameters(tensor, tensor_bool_throw_all_bool)
{
    static tensor_call_throw_params_c<bool> params[] = {
        // shape, values, throw_expected
        {{1}, {true}, false},
        {{3}, {true, false, false}, false},
        {{1, 3}, {true, false, false}, false},
        {{2, 2}, {true, false, true, false}, false},
    };

    return criterion_test_params(params);
}

ParameterizedTest(tensor_call_throw_params<bool> *param, tensor, tensor_bool_throw_all_bool)
{
    Tensor<bool> tensor = Tensor<bool>(param->shape, param->values);
    test_throw([tensor]()
               { tensor.all(); }, param->throw_expected);
}

ParameterizedTestParameters(tensor, tensor_bool_throw_all_int)
{
    static tensor_call_throw_params_c<int> params[] = {
        // shape, values, throw_expected
        {{1}, {0}, true},
        {{3}, {0, 1, 2}, true},
        {{1, 3}, {0, 1, 2}, true},
        {{2, 2}, {0, 1, 2, 3}, true},
    };

    return criterion_test_params(params);
}

ParameterizedTest(tensor_call_throw_params<int> *param, tensor, tensor_bool_throw_all_int)
{
    Tensor<int> tensor = Tensor<int>(param->shape, param->values);
    test_throw([tensor]()
               { tensor.all(); }, param->throw_expected);
}

ParameterizedTestParameters(tensor, tensor_bool_throw_any_bool)
{
    static tensor_call_throw_params_c<bool> params[] = {
        // shape, values, throw_expected
        {{1}, {true}, false},
        {{3}, {true, false, false}, false},
        {{1, 3}, {true, false, false}, false},
        {{2, 2}, {true, false, true, false}, false},
    };

    return criterion_test_params(params);
}

ParameterizedTest(tensor_call_throw_params<bool> *param, tensor, tensor_bool_throw_any_bool)
{
    Tensor<bool> tensor = Tensor<bool>(param->shape, param->values);
    test_throw([tensor]()
               { tensor.any(); }, param->throw_expected);
}

ParameterizedTestParameters(tensor, tensor_bool_throw_any_int)
{
    static tensor_call_throw_params_c<int> params[] = {
        // shape, values, throw_expected
        {{1}, {0}, true},
        {{3}, {0, 1, 2}, true},
        {{1, 3}, {0, 1, 2}, true},
        {{2, 2}, {0, 1, 2, 3}, true},
    };

    return criterion_test_params(params);
}

ParameterizedTest(tensor_call_throw_params<int> *param, tensor, tensor_bool_throw_any_int)
{
    Tensor<int> tensor = Tensor<int>(param->shape, param->values);
    test_throw([tensor]()
               { tensor.any(); }, param->throw_expected);
}

ParameterizedTestParameters(tensor, tensor_bool_throw_none_bool)
{
    static tensor_call_throw_params_c<bool> params[] = {
        // func, shape, values, throw_expected
        {{1}, {true}, false},
        {{3}, {true, false, false}, false},
        {{1, 3}, {true, false, false}, false},
        {{2, 2}, {true, false, true, false}, false},
    };

    return criterion_test_params(params);
}

ParameterizedTest(tensor_call_throw_params<bool> *param, tensor, tensor_bool_throw_none_bool)
{
    Tensor<bool> tensor = Tensor<bool>(param->shape, param->values);
    test_throw([tensor]()
               { tensor.none(); }, param->throw_expected);
}

ParameterizedTestParameters(tensor, tensor_bool_throw_none_int)
{
    static tensor_call_throw_params_c<int> params[] = {
        // shape, values, throw_expected
        {{1}, {0}, true},
        {{3}, {0, 1, 2}, true},
        {{1, 3}, {0, 1, 2}, true},
        {{2, 2}, {0, 1, 2, 3}, true},
    };

    return criterion_test_params(params);
}

ParameterizedTest(tensor_call_throw_params<int> *param, tensor, tensor_bool_throw_none_int)
{
    Tensor<int> tensor = Tensor<int>(param->shape, param->values);
    test_throw([tensor]()
               { tensor.none(); }, param->throw_expected);
}

ParameterizedTestParameters(tensor, tensor_bool_throw_cast_bool)
{
    static tensor_call_throw_params_c<bool> params[] = {
        // shape, values, throw_expected
        {{1}, {true}, false},
        {{3}, {true, false, false}, true},
        {{1, 3}, {true, false, false}, true},
        {{2, 2}, {true, false, true, false}, true},
    };

    return criterion_test_params(params);
}

ParameterizedTest(tensor_call_throw_params<bool> *param, tensor, tensor_bool_throw_cast_bool)
{
    Tensor<bool> tensor = Tensor<bool>(param->shape, param->values);
    test_throw([tensor]()
               { (bool)tensor; }, param->throw_expected);
}

ParameterizedTestParameters(tensor, tensor_bool_throw_cast_int)
{
    static tensor_call_throw_params_c<int> params[] = {
        // shape, values, throw_expected
        {{1}, {0}, true},
        {{3}, {0, 1, 2}, true},
        {{1, 3}, {0, 1, 2}, true},
        {{2, 2}, {0, 1, 2, 3}, true},
    };

    return criterion_test_params(params);
}

ParameterizedTest(tensor_call_throw_params<int> *param, tensor, tensor_bool_throw_cast_int)
{
    Tensor<int> tensor = Tensor<int>(param->shape, param->values);
    test_throw([tensor]()
               { (bool)tensor; }, param->throw_expected);
}
