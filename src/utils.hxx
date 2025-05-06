#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <numeric>

template <typename T>
std::string reccursiveStr(std::vector<std::size_t> shape, T *buffer)
{
    if (shape.empty())
        return std::to_string(*buffer);
    std::size_t step = std::reduce(shape.begin() + 1, shape.end(), 1, std::multiplies<int>());
    std::string str = "[";
    for (std::size_t i = 0; i < shape[0]; i++)
        str += reccursiveBuild(std::vector<std::size_t>(shape.begin() + 1, shape.end()), buffer + i * step) + (i != shape[0] - 1 ? "," : "");
    str = str + "]";
    return str;
}