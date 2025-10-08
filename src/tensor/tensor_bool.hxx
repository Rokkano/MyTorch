#include <format>

#include "tensor.hh"
#include "../utils.hh"

template <typename T>
Tensor<bool> Tensor<T>::operator==(const Tensor<T> &other)
{
    if (!this->validateSameShape(other.shape_))
        throw TensorInvalidShapeException(std::format("Shape {} and {} are invalid for comparison.", this->tensorShapeToStr(this->shape_), other.tensorShapeToStr(other.shape_)));

    Tensor<bool> tensor = Tensor<bool>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = (bool)(this->buffer_[i] == other.buffer_[i]);
    return tensor;
}

template <typename T>
Tensor<bool> Tensor<T>::operator==(const T &other)
{
    Tensor<bool> tensor = Tensor<bool>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = (bool)(this->buffer_[i] == other);
    return tensor;
}

template <typename T>
Tensor<bool> Tensor<T>::operator<(const Tensor<T> &other)
{
    if (!this->validateSameShape(other.shape_))
        throw TensorInvalidShapeException(std::format("Shape {} and {} are invalid for comparison.", this->tensorShapeToStr(this->shape_), other.tensorShapeToStr(other.shape_)));

    Tensor<bool> tensor = Tensor<bool>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = (bool)(this->buffer_[i] < other.buffer_[i]);
    return tensor;
}

template <typename T>
Tensor<bool> Tensor<T>::operator<(const T &other)
{
    Tensor<bool> tensor = Tensor<bool>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = (bool)(this->buffer_[i] < other);
    return tensor;
}

template <typename T>
Tensor<bool> Tensor<T>::operator<=(const Tensor<T> &other)
{
    if (!this->validateSameShape(other.shape_))
        throw TensorInvalidShapeException(std::format("Shape {} and {} are invalid for comparison.", this->tensorShapeToStr(this->shape_), other.tensorShapeToStr(other.shape_)));

    Tensor<bool> tensor = Tensor<bool>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = (bool)(this->buffer_[i] <= other.buffer_[i]);
    return tensor;
}

template <typename T>
Tensor<bool> Tensor<T>::operator<=(const T &other)
{
    Tensor<bool> tensor = Tensor<bool>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = (bool)(this->buffer_[i] <= other);
    return tensor;
}

template <typename T>
Tensor<bool> Tensor<T>::operator>(const Tensor<T> &other)
{
    if (!this->validateSameShape(other.shape_))
        throw TensorInvalidShapeException(std::format("Shape {} and {} are invalid for comparison.", this->tensorShapeToStr(this->shape_), other.tensorShapeToStr(other.shape_)));

    Tensor<bool> tensor = Tensor<bool>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = (bool)(this->buffer_[i] > other.buffer_[i]);
    return tensor;
}

template <typename T>
Tensor<bool> Tensor<T>::operator>(const T &other)
{
    Tensor<bool> tensor = Tensor<bool>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = (bool)(this->buffer_[i] > other);
    return tensor;
}

template <typename T>
Tensor<bool> Tensor<T>::operator>=(const Tensor<T> &other)
{
    if (!this->validateSameShape(other.shape_))
        throw TensorInvalidShapeException(std::format("Shape {} and {} are invalid for comparison.", this->tensorShapeToStr(this->shape_), other.tensorShapeToStr(other.shape_)));

    Tensor<bool> tensor = Tensor<bool>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = (bool)(this->buffer_[i] >= other.buffer_[i]);
    return tensor;
}

template <typename T>
Tensor<bool> Tensor<T>::operator>=(const T &other)
{
    Tensor<bool> tensor = Tensor<bool>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = (bool)(this->buffer_[i] >= other);
    return tensor;
}

template <typename T>
Tensor<T>::operator bool() const
{
    throw TensorInvalidTypeException(std::format("bool() can only be used on Tensor<bool>. Got Tensor<{}>.", type_name<T>()));
}

template <typename T>
Tensor<bool> Tensor<T>::all() const
{
    throw TensorInvalidTypeException(std::format("all() can only be used on Tensor<bool>. Got Tensor<{}>.", type_name<T>()));
}

template <typename T>
Tensor<bool> Tensor<T>::any() const
{
    throw TensorInvalidTypeException(std::format("any() can only be used on Tensor<bool>. Got Tensor<{}>.", type_name<T>()));
}

template <typename T>
Tensor<bool> Tensor<T>::none() const
{
    throw TensorInvalidTypeException(std::format("none() can only be used on Tensor<bool>. Got Tensor<{}>.", type_name<T>()));
}

template <>
Tensor<bool>::operator bool() const;

template <>
Tensor<bool> Tensor<bool>::all() const;

template <>
Tensor<bool> Tensor<bool>::any() const;

template <>
Tensor<bool> Tensor<bool>::none() const;
