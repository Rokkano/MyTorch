#include "tensor.hh"

#include <format>

template <typename T>
bool Tensor<T>::validateSameShape(const std::vector<std::size_t> &shape) const
{
    if (this->shape_.size() != shape.size())
        return false;
    for (std::size_t i = 0; i < shape.size(); i++)
        if (this->shape_[i] != shape[i])
            return false;
    return true;
}

template <typename T>
Tensor<T> operator+(const Tensor<T> &lhs, const Tensor<T> &rhs)
{
    if (!lhs.validateSameShape(rhs.shape_))
        throw TensorInvalidShapeException(std::format("Shape {} and {} are invalid for addition.",
                                                      lhs.tensorShapeToStr(lhs.shape_),
                                                      rhs.tensorShapeToStr(rhs.shape_)));

    Tensor<T> tensor = Tensor<T>(lhs.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = lhs.buffer_[i] + rhs.buffer_[i];
    return tensor;
}

template <typename T>
Tensor<T> operator+(const Tensor<T> &lhs, const T &rhs)
{
    Tensor<T> tensor = Tensor<T>(lhs.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = lhs.buffer_[i] + rhs;
    return tensor;
}

template <typename T>
Tensor<T> operator+(const T &lhs, const Tensor<T> &rhs)
{
    Tensor<T> tensor = Tensor<T>(rhs.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = lhs + rhs.buffer_[i];
    return tensor;
}

template <typename T>
Tensor<T> operator+(const Tensor<T> &lhs)
{
    Tensor<T> tensor = Tensor<T>(lhs.shape_);
    for (std::size_t i = 0; i < lhs.numel(); i++)
        tensor.buffer_[i] = lhs.buffer_[i];
    return tensor;
}

template <typename T>
Tensor<T> operator-(const Tensor<T> &lhs, const Tensor<T> &rhs)
{
    if (!lhs.validateSameShape(rhs.shape_))
        throw TensorInvalidShapeException(std::format("Shape {} and {} are invalid for substraction.",
                                                      lhs.tensorShapeToStr(lhs.shape_),
                                                      rhs.tensorShapeToStr(rhs.shape_)));

    Tensor<T> tensor = Tensor<T>(lhs.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = lhs.buffer_[i] - rhs.buffer_[i];
    return tensor;
}

template <typename T>
Tensor<T> operator-(const Tensor<T> &lhs, const T &rhs)
{
    Tensor<T> tensor = Tensor<T>(lhs.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = lhs.buffer_[i] - rhs;
    return tensor;
}

template <typename T>
Tensor<T> operator-(const T &lhs, const Tensor<T> &rhs)
{
    Tensor<T> tensor = Tensor<T>(rhs.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = lhs - rhs.buffer_[i];
    return tensor;
}

template <typename T>
Tensor<T> operator-(const Tensor<T> &lhs)
{
    Tensor<T> tensor = Tensor<T>(lhs.shape_);
    for (std::size_t i = 0; i < lhs.numel(); i++)
        tensor.buffer_[i] = -(lhs.buffer_[i]);
    return tensor;
}

template <typename T>
Tensor<T> operator*(const Tensor<T> &lhs, const Tensor<T> &rhs)
{
    if (!lhs.validateSameShape(rhs.shape_))
        throw TensorInvalidShapeException(std::format("Shape {} and {} are invalid for multiplication.",
                                                      lhs.tensorShapeToStr(lhs.shape_),
                                                      rhs.tensorShapeToStr(rhs.shape_)));

    Tensor<T> tensor = Tensor<T>(lhs.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = lhs.buffer_[i] * rhs.buffer_[i];
    return tensor;
}

template <typename T>
Tensor<T> operator*(const Tensor<T> &lhs, const T &rhs)
{
    Tensor<T> tensor = Tensor<T>(lhs.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = lhs.buffer_[i] * rhs;
    return tensor;
}

template <typename T>
Tensor<T> operator*(const T &lhs, const Tensor<T> &rhs)
{
    Tensor<T> tensor = Tensor<T>(rhs.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = lhs * rhs.buffer_[i];
    return tensor;
}

template <typename T>
Tensor<T> operator/(const Tensor<T> &lhs, const Tensor<T> &rhs)
{
    if (!lhs.validateSameShape(rhs.shape_))
        throw TensorInvalidShapeException(std::format("Shape {} and {} are invalid for division.",
                                                      lhs.tensorShapeToStr(lhs.shape_),
                                                      rhs.tensorShapeToStr(rhs.shape_)));

    Tensor<T> tensor = Tensor<T>(lhs.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = lhs.buffer_[i] / rhs.buffer_[i];
    return tensor;
}

template <typename T>
Tensor<T> operator/(const Tensor<T> &lhs, const T &rhs)
{
    Tensor<T> tensor = Tensor<T>(lhs.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = lhs.buffer_[i] / rhs;
    return tensor;
}

template <typename T>
Tensor<T> operator/(const T &lhs, const Tensor<T> &rhs)
{
    Tensor<T> tensor = Tensor<T>(rhs.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = lhs / rhs.buffer_[i];
    return tensor;
}
