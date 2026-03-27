#pragma once

#include "../../src/tensor/tensor.hh"
#include "../test.hh"

#include <iostream>
#include <vector>

inline auto lin = [](std::size_t x) { return 2.5f * x + -4.7f; };

#define VEC std::vector<std::size_t>

template <typename T>
bool _vectorEQ(std::vector<T> lhs, std::vector<T> rhs)
{
    if (lhs.size() != rhs.size())
        return false;
    for (std::size_t i = 0; i < lhs.size(); i++)
        if (lhs[i] != rhs[i])
            return false;
    return true;
}