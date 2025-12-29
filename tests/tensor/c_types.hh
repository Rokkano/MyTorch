#pragma once

#include <criterion/criterion.h>
#include <iostream>
#include <vector>

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

template <typename T, typename size_t_allocator, typename T_allocator>
struct tensor_call_throw_params_
{
    std::vector<size_t, size_t_allocator> shape;
    std::vector<T, T_allocator> values;
    bool throw_expected;
};

template <typename T>
using tensor_call_throw_params_c = tensor_call_throw_params_<T, criterion::allocator<std::size_t>, criterion::allocator<T>>;
template <typename T>
using tensor_call_throw_params = tensor_call_throw_params_<T, std::allocator<std::size_t>, std::allocator<T>>;
