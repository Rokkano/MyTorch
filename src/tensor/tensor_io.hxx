#pragma once

#include "src/cv/cv.hh"
#include "src/utils.hh"
#include "tensor.hh"

#include <format>
#include <functional>
#include <iostream>
#include <tuple>
#include <type_traits>


template <typename T, template <typename> typename B>
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