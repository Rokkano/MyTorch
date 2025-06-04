#include "tensor.hh"

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
Tensor<T> Tensor<T>::operator+(const Tensor<T> &other)
{
    if (!this->validateSameShape(other.shape_))
        throw TensorInvalidShapeException(std::format("Shape {} and {} are invalid for addition.", this->tensorShapeToStr(this->shape_), other.tensorShapeToStr(other.shape_)));

    Tensor<T> tensor = Tensor<T>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = this->buffer_[i] + other.buffer_[i];
    return tensor;
}
template <typename T>
Tensor<T> Tensor<T>::operator+(const T &other)
{
    Tensor<T> tensor = Tensor<T>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = this->buffer_[i] + other;
    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::operator+()
{
    Tensor<T> tensor = Tensor<T>(this->shape_);
    for (std::size_t i = 0; i < this->numel(); i++)
        tensor.buffer_[i] = this->buffer_[i];
    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::operator-(const Tensor<T> &other)
{
    if (!this->validateSameShape(other.shape_))
        throw TensorInvalidShapeException(std::format("Shape {} and {} are invalid for substraction.", this->tensorShapeToStr(this->shape_), other.tensorShapeToStr(other.shape_)));

    Tensor<T> tensor = Tensor<T>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = this->buffer_[i] - other.buffer_[i];
    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::operator-(const T &other)
{
    Tensor<T> tensor = Tensor<T>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = this->buffer_[i] - other;
    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::operator-()
{
    Tensor<T> tensor = Tensor<T>(this->shape_);
    for (std::size_t i = 0; i < this->numel(); i++)
        tensor.buffer_[i] = -this->buffer_[i];
    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T> &other)
{
    if (!this->validateSameShape(other.shape_))
        throw TensorInvalidShapeException(std::format("Shape {} and {} are invalid for multiplication.", this->tensorShapeToStr(this->shape_), other.tensorShapeToStr(other.shape_)));

    Tensor<T> tensor = Tensor<T>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = this->buffer_[i] * other.buffer_[i];
    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::operator*(const T &other)
{
    Tensor<T> tensor = Tensor<T>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = this->buffer_[i] * other;
    return tensor;
}
