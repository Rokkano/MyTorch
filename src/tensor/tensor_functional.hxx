#pragma once

#include <optional>

#include "tensor.hh"

template <typename T>
Tensor<T> *Tensor<T>::affine(const Tensor<T> &rhs, std::optional<T> a, std::optional<T> b)
{
    Tensor<T> *tensor = new Tensor<T>(rhs.shape_);
    for (std::size_t i = 0; i < rhs.numel(); i++)
    {
        T value = rhs.buffer_[i];
        if (a.has_value())
            value = value * a.value();
        if (b.has_value())
            value = value + b.value();
        tensor->buffer_[i] = value;
    }
    return tensor;
}