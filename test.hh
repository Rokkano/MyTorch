#pragma once

#include <cstddef>
#include <concepts>
#include <vector>

template <typename T, typename B>
concept IsBackend = requires(B::TStorage storage, std::size_t i) {
    typename B::TStorage;
    { storage[i] } -> std::same_as<T &>;
};

template <typename T, typename B>
requires IsBackend<T, B>
class Tensor
{
public:
    // Tensor<bool, B> test1() const;
    Tensor<int, B> test2() const;
    // Tensor<std::size_t, B> test1() const;
};

template <typename T>
class Backend
{
public:
    using TStorage = std::vector<T>;
};