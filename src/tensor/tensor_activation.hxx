#include "tensor.hh"

template <typename T, typename B>
Tensor<T, B> Tensor<T, B>::relu(const Tensor<T, B> &tensor) requires std::is_arithmetic_v<T>
{
    Tensor<T, B> res = Tensor<T, B>(tensor.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        res[i] = (tensor[i] > 0) ? tensor[i] : 0;
    return res;
}

template <typename T, typename B>
Tensor<T, B> Tensor<T, B>::drelu(const Tensor<T, B> &tensor) requires std::is_arithmetic_v<T>
{
    Tensor<T, B> res = Tensor<T, B>(tensor.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        res[i] = (tensor[i] > 0) ? 1 : 0;
    return res;
}

template <typename T, typename B>
Tensor<T, B> Tensor<T, B>::sigmoid(const Tensor<T, B> &tensor) requires std::is_arithmetic_v<T>
{
    return 1.f / (1.f + Tensor<T, B>::exp(-tensor));
}

template <typename T, typename B>
Tensor<T, B> Tensor<T, B>::dsigmoid(const Tensor<T, B> &tensor) requires std::is_arithmetic_v<T>
{
    return Tensor<T, B>::sigmoid(tensor) * (1.f - Tensor<T, B>::sigmoid(tensor));
}

template <typename T, typename B>
Tensor<T, B> Tensor<T, B>::sinh(const Tensor<T, B> &tensor) requires std::is_arithmetic_v<T>
{
    return (Tensor<T, B>::exp(tensor) - Tensor<T, B>::exp(-tensor)) / 2;
}

template <typename T, typename B>
Tensor<T, B> Tensor<T, B>::cosh(const Tensor<T, B> &tensor) requires std::is_arithmetic_v<T>
{
    return (Tensor<T, B>::exp(tensor) + Tensor<T, B>::exp(-tensor)) / 2;
}

template <typename T, typename B>
Tensor<T, B> Tensor<T, B>::tanh(const Tensor<T, B> &tensor) requires std::is_arithmetic_v<T>
{
    return Tensor<T, B>::sinh(tensor) / Tensor<T, B>::cosh(tensor);
}

template <typename T, typename B>
Tensor<T, B> Tensor<T, B>::dtanh(const Tensor<T, B> &tensor) requires std::is_arithmetic_v<T>
{
    return 1 - Tensor<T, B>::tanh(Tensor<T, B>::tanh(tensor));
}

template <typename T, typename B>
Tensor<T, B> Tensor<T, B>::softmax(const Tensor<T, B> &tensor) requires std::is_arithmetic_v<T>
{
    // Numeric stability
    Tensor<T, B> exp = Tensor<T, B>::exp(tensor - tensor.max().item());
    return exp / exp.sum().item();
}
