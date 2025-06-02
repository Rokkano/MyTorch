#include <optional>
#include <cmath>
#include <format>
#include <typeinfo>

#include "tensor.hh"

template <typename T>
Tensor<T> Tensor<T>::affine(const Tensor<T> &rhs, std::optional<T> a, std::optional<T> b)
    requires std::is_arithmetic_v<T>
{
    Tensor<T> tensor = Tensor<T>(rhs.shape_);
    for (std::size_t i = 0; i < rhs.numel(); i++)
    {
        T value = rhs.buffer_[i];
        if (a.has_value())
            value = value * a.value();
        if (b.has_value())
            value = value + b.value();
        tensor.buffer_[i] = value;
    }
    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::exp(const Tensor<T> &t)
    requires std::is_arithmetic_v<T>
{
    Tensor<T> tensor = Tensor<T>(t.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = std::exp(t.buffer_[i]);
    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::pow(const Tensor<T> &t, const double exponent)
    requires std::is_arithmetic_v<T>
{
    Tensor<T> tensor = Tensor<T>(t.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = std::pow(t.buffer_[i], exponent);
    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::sqrt(const Tensor<T> &t)
    requires std::is_arithmetic_v<T>
{
    Tensor<T> tensor = Tensor<T>(t.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = std::sqrt(t.buffer_[i]);
    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::dot(const Tensor<T> &lhs, const Tensor<T> &rhs)
    requires std::is_arithmetic_v<T>
{
    if (lhs.shape_.size() != 1)
        throw std::invalid_argument(std::format("Dot product only applies for 1-dimensional tensors : {}.", Tensor<T>::tensorShapeToStr(lhs.shape_)));
    if (rhs.shape_.size() != 1)
        throw std::invalid_argument(std::format("Dot product only applies for 1-dimensional tensors : {}.", Tensor<T>::tensorShapeToStr(rhs.shape_)));
    if (lhs.shape_[0] != rhs.shape_[0])
        throw std::invalid_argument(std::format("Lengths are incompatible for dot product : {} and {}.", Tensor<T>::tensorShapeToStr(rhs.shape_), Tensor<T>::tensorShapeToStr(lhs.shape_)));

    Tensor<T> tensor = Tensor<T>({1});
    tensor.buffer_[0] = 0;
    for (std::size_t i = 0; i < lhs.shape_[0]; i++)
        tensor.buffer_[0] += lhs.buffer_[i] * rhs.buffer_[i];
    return tensor;
}

// // MULTIPLIER X et X.T (pour faire ligne x ligne)
// template <typename T>
// Tensor<T> mm(const Tensor<T> &lhs, const Tensor<T> &rhs)
// {
//     // matrix multiplication (2x2 tensors)
//     return lhs
// }

// // MULTIPLIER X et X.T (pour faire ligne x ligne)
// template <typename T>
// Tensor<T> bmm(const Tensor<T> &lhs, const Tensor<T> &rhs)
// {
//     return lhs
// }

// template <typename T>
// Tensor<T> Tensor<T>::matmul(const Tensor<T> &lhs, const Tensor<T> &rhs)
//     requires std::is_arithmetic_v<T>
// {
//     if (lhs.shape_.size() == 1 && rhs.shape_.size() == 1)
//         return Tensor<T>::dot(lhs, rhs);
//     if (lhs.shape_.size() == 2 && rhs.shape_.size() == 2)
//         return matmatproduct(lhs, rhs);
//     if (lhs.shape_.size() == 1 && rhs.shape_.size() == 2)
//     {
//         rhs = rhs.unsqueeze();
//         tensor = matmatproduct(lhs, rhs);
//         tensor = tensor.squeeze();
//         return tensor;
//     }
// }