#include <optional>
#include <cmath>
#include <format>
#include <typeinfo>

#include "tensor.hh"

template <typename T>
Tensor<T> *Tensor<T>::affine(const Tensor<T> &rhs, std::optional<T> a, std::optional<T> b)
    requires std::is_arithmetic_v<T>
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

template <typename T>
Tensor<T> *Tensor<T>::exp(const Tensor<T> &t)
    requires std::is_arithmetic_v<T>
{
    Tensor<T> *tensor = new Tensor<T>(t.shape_);
    for (std::size_t i = 0; i < tensor->numel(); i++)
        tensor->buffer_[i] = std::exp(t.buffer_[i]);
    return tensor;
}

template <typename T>
Tensor<T> *Tensor<T>::pow(const Tensor<T> &t, const double exponent)
    requires std::is_arithmetic_v<T>
{
    Tensor<T> *tensor = new Tensor<T>(t.shape_);
    for (std::size_t i = 0; i < tensor->numel(); i++)
        tensor->buffer_[i] = std::pow(t.buffer_[i], exponent);
    return tensor;
}

template <typename T>
Tensor<T> *Tensor<T>::sqrt(const Tensor<T> &t)
    requires std::is_arithmetic_v<T>
{
    Tensor<T> *tensor = new Tensor<T>(t.shape_);
    for (std::size_t i = 0; i < tensor->numel(); i++)
        tensor->buffer_[i] = std::sqrt(t.buffer_[i]);
    return tensor;
}

template <typename T>
Tensor<T> *Tensor<T>::dot(const Tensor<T> &lhs, const Tensor<T> &rhs)
    requires std::is_arithmetic_v<T>
{
    if (lhs.shape_.size() != 1)
        throw std::invalid_argument("Dot product only applies for 1-dimensional tensors : " + Tensor<T>::tensorShapeToStr(lhs.shape_) + ".");
    if (rhs.shape_.size() != 1)
        throw std::invalid_argument("Dot product only applies for 1-dimensional tensors : " + Tensor<T>::tensorShapeToStr(rhs.shape_) + ".");
    if (lhs.shape_[0] != rhs.shape_[0])
        throw std::invalid_argument("Lengths are incompatible for dot product : " + Tensor<T>::tensorShapeToStr(rhs.shape_) + " and " + Tensor<T>::tensorShapeToStr(lhs.shape_) + ".");

    Tensor<T> *tensor = new Tensor<T>({1});
    tensor->buffer_[0] = 0;
    for (std::size_t i = 0; i < lhs.shape_[0]; i++)
        tensor->buffer_[0] += lhs.buffer_[i] * rhs.buffer_[i];
    return tensor;
}

// template <typename T>
// Tensor<T> *Tensor<T>::matmul(const Tensor<T> &lhs, const Tensor<T> &rhs)
//     requires std::is_arithmetic_v<T>
// {
//     if (lhs.shape_.size() == 1 && rhs.shape_.size() == 1)
//         return Tensor<T>::dot(lhs, rhs)
// }