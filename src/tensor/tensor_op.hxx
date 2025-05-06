#include "tensor.hh"

template <typename T>
bool Tensor<T>::validateSameShape(std::vector<std::size_t> shape) const
{
    if (this->shape_.size() != shape.size())
        return false;
    for (std::size_t i = 0; i < shape.size(); i++)
        if (this->shape_[i] != shape[i])
            return false;
    return true;
}

template <typename T>
Tensor<T> *operator+(const Tensor<T> &rhs, const Tensor<T> &lhs)
{
    if (!rhs.validateSameShape(lhs.shape_))
        throw std::invalid_argument("Shape " + rhs.tensorShapeToStr(rhs.shape_) + " and " + lhs.tensorShapeToStr(lhs.shape_) + " are invalid for addition.");

    Tensor<T> *tensor = new Tensor<T>(rhs.shape_);

    for (std::size_t i = 0; i < tensor->numel(); i++)
        tensor->buffer_[i] = rhs.buffer_[i] + lhs.buffer_[i];

    return tensor;
}

template <typename T>
Tensor<T> *operator+(const Tensor<T> &rhs, const T &lhs)
{
    Tensor<T> *tensor = new Tensor<T>(rhs.shape_);

    for (std::size_t i = 0; i < tensor->numel(); i++)
        tensor->buffer_[i] = rhs.buffer_[i] + lhs;

    return tensor;
}

template <typename T>
Tensor<T> *operator+(const Tensor<T> &lhs)
{
    return lhs;
}

template <typename T>
Tensor<T> *operator-(const Tensor<T> &rhs, const Tensor<T> &lhs)
{
    if (!rhs.validateSameShape(lhs.shape_))
        throw std::invalid_argument("Shape " + rhs.tensorShapeToStr(rhs.shape_) + " and " + lhs.tensorShapeToStr(lhs.shape_) + " are invalid for addition.");

    Tensor<T> *tensor = new Tensor<T>(rhs.shape_);

    for (std::size_t i = 0; i < tensor->numel(); i++)
        tensor->buffer_[i] = rhs.buffer_[i] - lhs.buffer_[i];

    return tensor;
}

template <typename T>
Tensor<T> *operator-(const Tensor<T> &rhs, const T &lhs)
{
    Tensor<T> *tensor = new Tensor<T>(rhs.shape_);

    for (std::size_t i = 0; i < tensor->numel(); i++)
        tensor->buffer_[i] = rhs.buffer_[i] - lhs;

    return tensor;
}

template <typename T>
Tensor<T> *operator-(const Tensor<T> &lhs)
{
    for (std::size_t i = 0; i < lhs.numel(); i++)
        lhs.buffer_[i] = -lhs.buffer_[i];
    return lhs;
}

template <typename T>
Tensor<T> *operator*(const Tensor<T> &rhs, const Tensor<T> &lhs)
{
    if (!rhs.validateSameShape(lhs.shape_))
        throw std::invalid_argument("Shape " + rhs.tensorShapeToStr(rhs.shape_) + " and " + lhs.tensorShapeToStr(lhs.shape_) + " are invalid for addition.");

    Tensor<T> *tensor = new Tensor<T>(rhs.shape_);

    for (std::size_t i = 0; i < tensor->numel(); i++)
        tensor->buffer_[i] = rhs.buffer_[i] * lhs.buffer_[i];

    return tensor;
}

template <typename T>
Tensor<T> *operator*(const Tensor<T> &rhs, const T &lhs)
{
    Tensor<T> *tensor = new Tensor<T>(rhs.shape_);

    for (std::size_t i = 0; i < tensor->numel(); i++)
        tensor->buffer_[i] = rhs.buffer_[i] * lhs;

    return tensor;
}