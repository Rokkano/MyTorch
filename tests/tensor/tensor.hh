#pragma once

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

template <typename allocator>
struct tensor_shape_params_
{
    std::vector<size_t, allocator> shape;
    std::vector<size_t, allocator> expected;
};
using tensor_shape_params_c = tensor_shape_params_<criterion::allocator<std::size_t>>;
using tensor_shape_params = tensor_shape_params_<std::allocator<std::size_t>>;

template <typename allocator>
struct tensor_broadcast_params_
{
    std::vector<size_t, allocator> shape;
    std::vector<size_t, allocator> target;
    std::vector<size_t, allocator> expected;
};
using tensor_broadcast_params_c = tensor_broadcast_params_<criterion::allocator<std::size_t>>;
using tensor_broadcast_params = tensor_broadcast_params_<std::allocator<std::size_t>>;
