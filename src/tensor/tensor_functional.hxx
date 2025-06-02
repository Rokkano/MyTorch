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

template <typename T>
Tensor<T> Tensor<T>::mm(const Tensor<T> &lhs, const Tensor<T> &rhs)
    requires std::is_arithmetic_v<T>
{
    // matrix multiplication (2x2 tensors)
    if (lhs.shape_.size() != 2)
        throw std::invalid_argument(std::format("mm only applies for 2-dimensional tensors : {}.", Tensor<T>::tensorShapeToStr(lhs.shape_)));
    if (rhs.shape_.size() != 2)
        throw std::invalid_argument(std::format("mm only applies for 2-dimensional tensors : {}.", Tensor<T>::tensorShapeToStr(rhs.shape_)));
    if (lhs.shape_[1] != rhs.shape_[0])
        throw std::invalid_argument(std::format("Tensors are not compatible for matrix multiplication : {} and {}.", Tensor<T>::tensorShapeToStr(lhs.shape_), Tensor<T>::tensorShapeToStr(rhs.shape_)));

    Tensor<T> tensor = Tensor<T>({lhs.shape_[0], rhs.shape_[1]});
    for (std::size_t y = 0; y < lhs.shape_[0]; y++)
        for (std::size_t x = 0; x < rhs.shape_[1]; x++)
            for (std::size_t k = 0; k < lhs.shape_[1]; k++)
                tensor.buffer_[x + y * rhs.shape_[1]] += lhs.buffer_[k + y * lhs.shape_[1]] * rhs.buffer_[x + k * rhs.shape_[1]];
    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::omm(const Tensor<T> &lhs, const Tensor<T> &rhs)
    requires std::is_arithmetic_v<T>
{
    // optimized matrix multiplication (2x2 tensors) with rhs transpose for quicker data reading
    if (lhs.shape_.size() != 2)
        throw std::invalid_argument(std::format("mm only applies for 2-dimensional tensors : {}.", Tensor<T>::tensorShapeToStr(lhs.shape_)));
    if (rhs.shape_.size() != 2)
        throw std::invalid_argument(std::format("mm only applies for 2-dimensional tensors : {}.", Tensor<T>::tensorShapeToStr(rhs.shape_)));
    if (lhs.shape_[1] != rhs.shape_[0])
        throw std::invalid_argument(std::format("Tensors are not compatible for matrix multiplication : {} and {}.", Tensor<T>::tensorShapeToStr(lhs.shape_), Tensor<T>::tensorShapeToStr(rhs.shape_)));

    Tensor<T> tensor = Tensor<T>({lhs.shape_[0], rhs.shape_[1]});
    Tensor<T> rhs_t = rhs.transpose();
    for (std::size_t y = 0; y < lhs.shape_[0]; y++)
        for (std::size_t x = 0; x < rhs.shape_[1]; x++)
            for (std::size_t k = 0; k < lhs.shape_[1]; k++)
                tensor.buffer_[x + y * rhs.shape_[1]] += lhs.buffer_[k + y * lhs.shape_[1]] * rhs_t.buffer_[k + x * rhs_t.shape_[1]];

    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::mvm(const Tensor<T> &lhs, const Tensor<T> &rhs)
    requires std::is_arithmetic_v<T>
{
    // matrix vector multiplication
    if (lhs.shape_.size() != 2)
        throw std::invalid_argument(std::format("mvm only applies for 2-dimensional tensors for lhs : {}.", Tensor<T>::tensorShapeToStr(lhs.shape_)));
    if (rhs.shape_.size() != 1)
        throw std::invalid_argument(std::format("mvm only applies for 1-dimensional tensors for rhs : {}.", Tensor<T>::tensorShapeToStr(rhs.shape_)));
    if (lhs.shape_[1] != rhs.shape_[0])
        throw std::invalid_argument(std::format("Tensors are not compatible for matrix-vector multiplication : {} and {}.", Tensor<T>::tensorShapeToStr(lhs.shape_), Tensor<T>::tensorShapeToStr(rhs.shape_)));

    Tensor<T> tensor = Tensor<T>({lhs.shape_[0]});
    for (std::size_t y = 0; y < lhs.shape_[0]; y++)
        for (std::size_t x = 0; x < lhs.shape_[1]; x++)
            tensor.buffer_[y] += lhs.buffer_[x + y * lhs.shape_[1]] * rhs.buffer_[x];
    return tensor;
}

// template <typename T>
// Tensor<T> Tensor<T>::bmm(const Tensor<T> &lhs, const Tensor<T> &rhs)
// requires std::is_arithmetic_v<T>
// {
//     return lhs
// }

template <typename T>
Tensor<T> Tensor<T>::matmul(const Tensor<T> &lhs, const Tensor<T> &rhs)
    requires std::is_arithmetic_v<T>
{
    if (lhs.shape_.size() == 1 && rhs.shape_.size() == 1)
        return Tensor<T>::dot(lhs, rhs);
    if (lhs.shape_.size() == 2 && rhs.shape_.size() == 2)
        return Tensor<T>::mm(lhs, rhs);
    if (lhs.shape_.size() == 1 && rhs.shape_.size() == 2)
        return Tensor<T>::mm(lhs.unsqueeze(), rhs).squeeze();
    if (lhs.shape_.size() == 2 && rhs.shape_.size() == 1)
        return Tensor<T>::mvm(lhs, rhs);
    throw std::invalid_argument("");
}