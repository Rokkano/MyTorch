#include <iostream>

#include "tensor.hh"
#include "../utils.hxx"

template <typename T>
std::string Tensor<T>::tensorDataToStr(std::vector<std::size_t> shape, T *buffer)
{
    if (shape.empty())
        return std::to_string(*buffer);
    std::size_t step = std::reduce(shape.begin() + 1, shape.end(), 1, std::multiplies<int>());
    std::string str = "[";
    for (std::size_t i = 0; i < shape[0]; i++)
        str += tensorDataToStr(std::vector<std::size_t>(shape.begin() + 1, shape.end()), buffer + i * step) + (i != shape[0] - 1 ? "," : "");
    str = str + "]";
    return str;
}

template <typename T>
std::string Tensor<T>::tensorShapeToStr(std::vector<std::size_t> shape)
{
    std::string str = "(";
    for (std::size_t i = 0; i < shape.size(); i++)
        str += std::to_string(shape[i]) + (i != shape.size() - 1 ? "," : "");
    str += ")";
    return str;
}

template <typename T>
std::string Tensor<T>::tensorToStr(std::vector<std::size_t> shape, T *buffer)
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