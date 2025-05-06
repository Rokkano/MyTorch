#pragma once

#include "tensor.hh"

template <>
Tensor<bool>::operator bool() const
{
    if (this->numel() != 1)
        throw std::invalid_argument("Boolean cast of a Tensor is ambiguous. Use .all() or .any() depending on the behaviour you want.");
    else
        return this->buffer_[0];
}

template <typename T>
Tensor<bool> *operator==(const Tensor<T> &rhs, const Tensor<T> &lhs)
{
    if (!rhs.validateSameShape(lhs.shape_))
        throw std::invalid_argument("Shape " + rhs.tensorShapeToStr(rhs.shape_) + " and " + lhs.tensorShapeToStr(lhs.shape_) + " are invalid for comparison.");

    Tensor<bool> *tensor = new Tensor<bool>(rhs.shape_);
    for (std::size_t i = 0; i < tensor->numel(); i++)
        tensor->buffer_[i] = (bool)(rhs.buffer_[i] == lhs.buffer_[i]);
    return tensor;
}

template <typename T>
Tensor<bool> *operator<(const Tensor<T> &rhs, const Tensor<T> &lhs)
{
    if (!rhs.validateSameShape(lhs.shape_))
        throw std::invalid_argument("Shape " + rhs.tensorShapeToStr(rhs.shape_) + " and " + lhs.tensorShapeToStr(lhs.shape_) + " are invalid for comparison.");

    Tensor<bool> *tensor = new Tensor<bool>(rhs.shape_);
    for (std::size_t i = 0; i < tensor->numel(); i++)
        tensor->buffer_[i] = (bool)(rhs.buffer_[i] < lhs.buffer_[i]);
    return tensor;
}

template <typename T>
Tensor<bool> *operator<=(const Tensor<T> &rhs, const Tensor<T> &lhs)
{
    if (!rhs.validateSameShape(lhs.shape_))
        throw std::invalid_argument("Shape " + rhs.tensorShapeToStr(rhs.shape_) + " and " + lhs.tensorShapeToStr(lhs.shape_) + " are invalid for comparison.");

    Tensor<bool> *tensor = new Tensor<bool>(rhs.shape_);
    for (std::size_t i = 0; i < tensor->numel(); i++)
        tensor->buffer_[i] = (bool)(rhs.buffer_[i] <= lhs.buffer_[i]);
    return tensor;
}

template <typename T>
Tensor<bool> *operator>(const Tensor<T> &rhs, const Tensor<T> &lhs)
{
    if (!rhs.validateSameShape(lhs.shape_))
        throw std::invalid_argument("Shape " + rhs.tensorShapeToStr(rhs.shape_) + " and " + lhs.tensorShapeToStr(lhs.shape_) + " are invalid for comparison.");

    Tensor<bool> *tensor = new Tensor<bool>(rhs.shape_);
    for (std::size_t i = 0; i < tensor->numel(); i++)
        tensor->buffer_[i] = (bool)(rhs.buffer_[i] > lhs.buffer_[i]);
    return tensor;
}

template <typename T>
Tensor<bool> *operator>=(const Tensor<T> &rhs, const Tensor<T> &lhs)
{
    if (!rhs.validateSameShape(lhs.shape_))
        throw std::invalid_argument("Shape " + rhs.tensorShapeToStr(rhs.shape_) + " and " + lhs.tensorShapeToStr(lhs.shape_) + " are invalid for comparison.");

    Tensor<bool> *tensor = new Tensor<bool>(rhs.shape_);
    for (std::size_t i = 0; i < tensor->numel(); i++)
        tensor->buffer_[i] = (bool)(rhs.buffer_[i] >= lhs.buffer_[i]);
    return tensor;
}

template <>
bool Tensor<bool>::all() const
{
    for (std::size_t i = 0; i < this->numel(); i++)
        if (!this->buffer_[i])
            return false;
    return true;
}

template <>
bool Tensor<bool>::any() const
{
    for (std::size_t i = 0; i < this->numel(); i++)
        if (this->buffer_[i])
            return true;
    return false;
}

template <>
bool Tensor<bool>::none() const
{
    for (std::size_t i = 0; i < this->numel(); i++)
        if (this->buffer_[i])
            return false;
    return true;
}
