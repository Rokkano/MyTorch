#include "src/cv/cv.hh"
#include "src/utils.hh"
#include "tensor.hh"

#include <format>
#include <iostream>
#include <tuple>
#include <type_traits>

template <typename T, typename B>
requires IsBackend<T, B>
std::string Tensor<T, B>::tensorDataToStr(const std::vector<std::size_t> &shape, const Tensor<T, B>::TStorage &data,
                                          std::size_t data_index)
{
    std::stringstream ss;
    std::size_t step = std::reduce(shape.begin() + 1, shape.end(), 1, std::multiplies<int>());

    if (shape.empty())
    {
        if constexpr (std::is_same_v<T, bool>)
            ss << (data[data_index] ? "true" : "false");
        else
            ss << data[data_index];
        return ss.str();
    }

    ss << "[";
    for (std::size_t i = 0; i < shape[0]; i++)
    {
        std::vector<std::size_t> new_shape = std::vector<std::size_t>(shape.begin() + 1, shape.end());
        std::size_t new_data_index = data_index + i * step;
        ss << tensorDataToStr(new_shape, data, new_data_index) + (i != shape[0] - 1 ? "," : "");
    }
    ss << "]";
    return ss.str();
}

template <typename T, typename B>
requires IsBackend<T, B>
std::string Tensor<T, B>::tensorShapeToStr(const std::vector<std::size_t> &shape)
{
    std::stringstream ss;
    ss << "(";
    for (std::size_t i = 0; i < shape.size(); i++)
        ss << shape[i] << (i != shape.size() - 1 ? "," : "");
    ss << ")";
    return ss.str();
}

template <typename T, typename B>
requires IsBackend<T, B>
std::string Tensor<T, B>::tensorToStr(const std::vector<std::size_t> &shape, const Tensor<T, B>::TStorage &data)
{
    std::string data_str = Tensor<T, B>::tensorDataToStr(shape, data);
    std::string shape_str = Tensor<T, B>::tensorShapeToStr(shape);
    return "tensor(shape=" + shape_str + "; data=(" + data_str + "); dtype=" + type_name<T>() + ")";
}

template <typename T, typename B>
requires IsBackend<T, B>
std::ostream &operator<<(std::ostream &os, const Tensor<T, B> &t)
{
    return os << Tensor<T, B>::tensorToStr(t.shape_, t.data_);
}

template <typename T, typename B>
requires IsBackend<T, B>
void Tensor<T, B>::plot(const std::string &linespec, OpenCVWindowOpts opts) const requires std::is_arithmetic_v<T>
{
    if (this->shape().size() != 1)
        throw TensorInvalidShapeException(std::format(
            "Shape {} is invalid for plot : need a single dimentional tensor.", this->tensorShapeToStr(this->shape())));

    std::vector<double> x = std::vector<double>(this->shape()[0]);
    std::iota(x.begin(), x.end(), 1);
    std::vector<T> y = this->data();

    CvPlot::Axes parent = CvPlot::makePlotAxes();
    parent.create<CvPlot::Series>(x, y, linespec);

    ::show(parent, opts);
}