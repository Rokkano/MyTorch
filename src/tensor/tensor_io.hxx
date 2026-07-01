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
void Tensor<T, B>::plot(TensorSpan tensors, const std::string &linespec, OpenCVWindowOpts opts) requires std::is_arithmetic_v<T>
{
    if (tensors.size() < 1)
        throw TensorInvalidArgException("No tensors to plot.");

    CvPlot::Axes parent = CvPlot::makePlotAxes();

    std::size_t len = tensors[0].get().shape()[0];
    std::vector<double> x = std::vector<double>(len);
    std::iota(x.begin(), x.end(), 1);

    for(std::size_t i = 0; i < tensors.size(); i++)
    {
        if (tensors[i].get().shape().size() != 1)
            throw TensorInvalidShapeException(std::format(
                "Shape {} is invalid for plot : need a single dimentional tensor.", Tensor<T, B>::shapeToStr(tensors[i].get().shape())));
        if (tensors[i].get().shape()[0] != len)
            throw TensorInvalidShapeException(std::format(
                "Plot tensors must all have the same length."));

        std::vector<T> y = tensors[i].get().vector();
        parent.create<CvPlot::Series>(x, y, linespec);
    }

    ::show(parent, opts);
}