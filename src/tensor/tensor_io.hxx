#include "../utils.hxx"
#include "tensor.hh"

#include <format>
#include <iostream>
#include <matplot/matplot.h>
#include <tuple>
#include <type_traits>

template <typename T>
std::string Tensor<T>::tensorDataToStr(const std::vector<std::size_t> &shape, const std::vector<T> &buffer)
{
    std::stringstream ss;
    if (shape.empty())
    {
        ss << buffer[0];
        return ss.str();
    }
    std::size_t step = std::reduce(shape.begin() + 1, shape.end(), 1, std::multiplies<int>());
    ss << "[";
    for (std::size_t i = 0; i < shape[0]; i++)
    {
        std::vector<std::size_t> new_shape = std::vector<std::size_t>(shape.begin() + 1, shape.end());
        std::vector<T> new_buffer = std::vector<T>(buffer.begin() + i * step, buffer.end());
        ss << tensorDataToStr(new_shape, new_buffer) + (i != shape[0] - 1 ? "," : "");
    }
    ss << "]";
    return ss.str();
}

template <typename T>
std::string Tensor<T>::tensorShapeToStr(const std::vector<std::size_t> &shape)
{
    std::stringstream ss;
    ss << "(";
    for (std::size_t i = 0; i < shape.size(); i++)
        ss << shape[i] << (i != shape.size() - 1 ? "," : "");
    ss << ")";
    return ss.str();
}

template <typename T>
std::string Tensor<T>::tensorToStr(const std::vector<std::size_t> &shape, const std::vector<T> &buffer)
{
    std::string data_str = Tensor<T>::tensorDataToStr(shape, buffer);
    std::string shape_str = Tensor<T>::tensorShapeToStr(shape);
    return "tensor(shape=" + shape_str + "; data=(" + data_str + "); dtype=" + type_name<T>() + ")";
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const Tensor<T> &t)
{
    return os << Tensor<T>::tensorToStr(t.shape_, t.buffer_);
}

template <typename T>
void Tensor<T>::plot(std::string title, std::string xlabel, std::string ylabel) const
    requires std::is_arithmetic_v<T>
{
    if (this->shape().size() != 1)
        throw TensorInvalidShapeException(std::format(
            "Shape {} is invalid for plot : need a single dimentional tensor.", this->tensorShapeToStr(this->shape())));

    std::vector<double> x = std::vector<double>(this->shape()[0]);
    std::iota(x.begin(), x.end(), 1);
    std::vector<T> y = this->buffer_;

    matplot::plot(x, y);
    matplot::title(title);
    matplot::xlabel(xlabel);
    matplot::ylabel(ylabel);
    matplot::show();
}