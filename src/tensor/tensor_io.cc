#include <iostream>
#include <format>

#include "tensor.hh"

template <>
std::string Tensor<bool>::tensorDataToStr(const std::vector<std::size_t> &shape, const std::vector<bool> &buffer)
{
    if (shape.empty())
        return buffer[0] ? "true" : "false";
    std::size_t step = std::reduce(shape.begin() + 1, shape.end(), 1, std::multiplies<int>());
    std::string str = "[";
    for (std::size_t i = 0; i < shape[0]; i++)
    {
        std::vector<std::size_t> new_shape = std::vector<std::size_t>(shape.begin() + 1, shape.end());
        std::vector<bool> new_buffer = std::vector<bool>(buffer.begin() + i * step, buffer.end());
        str += tensorDataToStr(new_shape, new_buffer) + (i != shape[0] - 1 ? "," : "");
    }
    str = str + "]";
    return str;
}
