#include <iostream>
#include <format>

#include "tensor.hh"
#include "../utils.hxx"

template <typename T>
std::string Tensor<T>::tensorDataToStr(const std::vector<std::size_t> &shape, const std::vector<T> &buffer)
{
    if (shape.empty())
        return std::to_string(buffer[0]);
    std::size_t step = std::reduce(shape.begin() + 1, shape.end(), 1, std::multiplies<int>());
    std::string str = "[";
    for (std::size_t i = 0; i < shape[0]; i++)
    {
        std::vector<std::size_t> new_shape = std::vector<std::size_t>(shape.begin() + 1, shape.end());
        std::vector<T> new_buffer = std::vector<T>(buffer.begin() + i * step, buffer.end());
        str += tensorDataToStr(new_shape, new_buffer) + (i != shape[0] - 1 ? "," : "");
    }
    str = str + "]";
    return str;
}

template <typename T>
std::string Tensor<T>::tensorShapeToStr(const std::vector<std::size_t> &shape)
{
    std::string str = "(";
    for (std::size_t i = 0; i < shape.size(); i++)
        str += std::to_string(shape[i]) + (i != shape.size() - 1 ? "," : "");
    str += ")";
    return str;
}

template <typename T>
std::string Tensor<T>::tensorToStr(const std::vector<std::size_t> &shape, const std::vector<T> &buffer)
{
    std::string data_str = Tensor<T>::tensorDataToStr(shape, buffer);
    std::string shape_str = Tensor<T>::tensorShapeToStr(shape);
    return "tensor(shape=" + shape_str + "; data=(" + data_str + "))";
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const Tensor<T> &t)
{
    return os << Tensor<T>::tensorToStr(t.shape_, t.buffer_);
}