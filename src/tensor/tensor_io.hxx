#pragma once

#include "src/cv/cv.hh"
#include "src/utils.hh"
#include "tensor.hh"

#include <format>
#include <functional>
#include <iostream>
#include <tuple>
#include <type_traits>

template <typename T, typename B>
requires IsBackend<T, B>
std::string Tensor<T, B>::toStr() const
{
    // Shape
    std::stringstream ssShape;
    ssShape << "(";
    for (std::size_t i = 0; i < this->shape_.size(); i++)
        ssShape << this->shape_[i] << (i != this->shape_.size() - 1 ? "," : "");
    ssShape << ")";

    // Buffer
    std::stringstream ssBuffer;
    ssBuffer << "(";
    using RecLambda = std::function<void(std::vector<std::size_t>, std::size_t)>;
    RecLambda rec = [&](std::vector<std::size_t> shape, std::size_t index) 
    { 
        if (!shape.empty())
        {
            std::size_t step = std::reduce(shape.begin() + 1, shape.end(), 1, std::multiplies<int>());
            std::vector<std::size_t> new_shape = std::vector<std::size_t>(shape.begin() + 1, shape.end());
            ssBuffer << "[";
            for (std::size_t i = 0; i < shape[0]; i++)
            {
                rec(new_shape, index + i * step);
                ssBuffer << (i != shape[0] - 1 ? "," : "");
            }
            ssBuffer << "]";
        }
        else
        { 
            if constexpr (std::is_same_v<T, bool>) 
                ssBuffer << ((*this)[index] ? "true" : "false");
            else
                ssBuffer << (*this)[index];
        }
    };
    rec(this->shape_, 0);
    ssBuffer << ")";

    return "tensor(shape=" + ssShape.str() + "; data=" + ssBuffer.str() + "; dtype=(" + type_name<T>() + "))";
}

template <typename T, typename B>
requires IsBackend<T, B>
std::ostream &operator<<(std::ostream &os, const Tensor<T, B> &t)
{
    return os << t.toStr();
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