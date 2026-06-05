#pragma once

#include "tensor.hh"

#include <format>

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
bool Tensor<T, B>::validateSameShape(const std::vector<std::size_t> &shape) const
{
    if (this->shape_.size() != shape.size())
        return false;
    for (std::size_t i = 0; i < shape.size(); i++)
        if (this->shape_[i] != shape[i])
            return false;
    return true;
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> operator+(const Tensor<T, B> &lhs, const Tensor<T, B> &rhs)
{
    if (!lhs.validateSameShape(rhs.shape_))
        throw TensorInvalidShapeException(std::format("Shape {} and {} are invalid for addition.",
                                                      Tensor<T, B>::shapeToStr(lhs.shape_),
                                                      Tensor<T, B>::shapeToStr(rhs.shape_)));

    Tensor<T, B> tensor = Tensor<T, B>(lhs.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor[i] = lhs[i] + rhs[i];
    return tensor;
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> operator+(const Tensor<T, B> &lhs, const T &rhs)
{
    Tensor<T, B> tensor = Tensor<T, B>(lhs.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor[i] = lhs[i] + rhs;
    return tensor;
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> operator+(const T &lhs, const Tensor<T, B> &rhs)
{
    Tensor<T, B> tensor = Tensor<T, B>(rhs.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor[i] = lhs + rhs[i];
    return tensor;
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> operator+(const Tensor<T, B> &lhs)
{
    Tensor<T, B> tensor = Tensor<T, B>(lhs.shape_);
    for (std::size_t i = 0; i < lhs.numel(); i++)
        tensor[i] = lhs[i];
    return tensor;
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> operator-(const Tensor<T, B> &lhs, const Tensor<T, B> &rhs)
{
    if (!lhs.validateSameShape(rhs.shape_))
        throw TensorInvalidShapeException(std::format("Shape {} and {} are invalid for substraction.",
                                                      Tensor<T, B>::shapeToStr(lhs.shape_),
                                                      Tensor<T, B>::shapeToStr(rhs.shape_)));

    Tensor<T, B> tensor = Tensor<T, B>(lhs.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor[i] = lhs[i] - rhs[i];
    return tensor;
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> operator-(const Tensor<T, B> &lhs, const T &rhs)
{
    Tensor<T, B> tensor = Tensor<T, B>(lhs.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor[i] = lhs[i] - rhs;
    return tensor;
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> operator-(const T &lhs, const Tensor<T, B> &rhs)
{
    Tensor<T, B> tensor = Tensor<T, B>(rhs.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor[i] = lhs - rhs[i];
    return tensor;
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> operator-(const Tensor<T, B> &lhs)
{
    Tensor<T, B> tensor = Tensor<T, B>(lhs.shape_);
    for (std::size_t i = 0; i < lhs.numel(); i++)
        tensor[i] = -(lhs[i]);
    return tensor;
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> operator*(const Tensor<T, B> &lhs, const Tensor<T, B> &rhs)
{
    if (!lhs.validateSameShape(rhs.shape_))
        throw TensorInvalidShapeException(std::format("Shape {} and {} are invalid for multiplication.",
                                                      Tensor<T, B>::shapeToStr(lhs.shape_),
                                                      Tensor<T, B>::shapeToStr(rhs.shape_)));

    Tensor<T, B> tensor = Tensor<T, B>(lhs.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor[i] = lhs[i] * rhs[i];
    return tensor;
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> operator*(const Tensor<T, B> &lhs, const T &rhs)
{
    Tensor<T, B> tensor = Tensor<T, B>(lhs.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor[i] = lhs[i] * rhs;
    return tensor;
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> operator*(const T &lhs, const Tensor<T, B> &rhs)
{
    Tensor<T, B> tensor = Tensor<T, B>(rhs.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor[i] = lhs * rhs[i];
    return tensor;
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> operator/(const Tensor<T, B> &lhs, const Tensor<T, B> &rhs)
{
    if (!lhs.validateSameShape(rhs.shape_))
        throw TensorInvalidShapeException(std::format("Shape {} and {} are invalid for division.",
                                                      Tensor<T, B>::shapeToStr(lhs.shape_),
                                                      Tensor<T, B>::shapeToStr(rhs.shape_)));

    Tensor<T, B> tensor = Tensor<T, B>(lhs.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor[i] = lhs[i] / rhs[i];
    return tensor;
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> operator/(const Tensor<T, B> &lhs, const T &rhs)
{
    Tensor<T, B> tensor = Tensor<T, B>(lhs.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor[i] = lhs[i] / rhs;
    return tensor;
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> operator/(const T &lhs, const Tensor<T, B> &rhs)
{
    Tensor<T, B> tensor = Tensor<T, B>(rhs.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor[i] = lhs / rhs[i];
    return tensor;
}
