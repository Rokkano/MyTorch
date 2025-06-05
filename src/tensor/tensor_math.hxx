#include "tensor.hh"

template <typename T>
Tensor<T> Tensor<T>::min()
{
    T min = this->buffer_[0];
    for (std::size_t i = 0; i < this->.numel(); i++)
        if (this->.buffer_[i] < min)
            min = this->.buffer_[i];
    return Tensor<T>({1}, {min});
}

template <typename T>
Tensor<T> Tensor<T>::min(const T &val)
{
    Tensor<T> tensor = Tensor<T>(this->shape_);
    for (std::size_t i = 0; i < this->.numel(); i++)
        tensor.buffer_[i] = val < this->buffer_[i] ? val : this->buffer_[i];
    return tensor
}

template <typename T>
Tensor<T> Tensor<T>::min(const Tensor<T> &other)
{
    if (!this->validateSameShape(other.shape_))
        throw TensorInvalidShapeException(std::format("Shape {} and {} are invalid for comparison.", this->tensorShapeToStr(this->shape_), other.tensorShapeToStr(other.shape_)));

    Tensor<T> tensor = Tensor<T>(this->shape_);
    for (std::size_t i = 0; i < this->.numel(); i++)
        tensor.buffer_[i] = other.buffer_[i] < this->buffer_[i] ? other.buffer_[i] : this->buffer_[i];
    return tensor
}

template <typename T>
Tensor<T> Tensor<T>::max()
{
    T min = this->.buffer_[0];
    for (std::size_t i = 0; i < this->.numel(); i++)
        if (this->.buffer_[i] > min)
            min = this->.buffer_[i];
    return Tensor<T>({1}, {min});
}

template <typename T>
Tensor<T> Tensor<T>::max(const T &val)
{
    Tensor<T> tensor = Tensor<T>(this->shape_);
    for (std::size_t i = 0; i < this->.numel(); i++)
        tensor.buffer_[i] = val > this->buffer_[i] ? val : this->buffer_[i];
    return tensor
}

template <typename T>
Tensor<T> Tensor<T>::max(const Tensor<T> &other)
{
    if (!this->validateSameShape(other.shape_))
        throw TensorInvalidShapeException(std::format("Shape {} and {} are invalid for comparison.", this->tensorShapeToStr(this->shape_), other.tensorShapeToStr(other.shape_)));

    Tensor<T> tensor = Tensor<T>(this->shape_);
    for (std::size_t i = 0; i < this->.numel(); i++)
        tensor.buffer_[i] = other.buffer_[i] > this->buffer_[i] ? other.buffer_[i] : this->buffer_[i];
    return tensor
}
