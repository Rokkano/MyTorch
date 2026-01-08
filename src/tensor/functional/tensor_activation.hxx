#include "../tensor.hh"

template <typename T>
Tensor<T> Tensor<T>::relu(const Tensor<T> &tensor)
    requires std::is_arithmetic_v<T>
{
    Tensor<T> res = Tensor<T>(tensor.shape_);
    for(std::size_t i = 0; i < tensor.numel(); i++)
        res[i] = (tensor[i] > 0) ? tensor[i] : 0;
    return res;
}

template <typename T>
Tensor<T> Tensor<T>::drelu(const Tensor<T> &tensor)
    requires std::is_arithmetic_v<T>
{
    Tensor<T> res = Tensor<T>(tensor.shape_);
    for(std::size_t i = 0; i < tensor.numel(); i++)
        res[i] = (tensor[i] > 0) ? 1 : 0;
    return res;
}

template <typename T>
Tensor<T> Tensor<T>::sigmoid(const Tensor<T> &tensor)
    requires std::is_arithmetic_v<T>
{
    return 1 / (1 + Tensor<T>::exp(-tensor));
}

template <typename T>
Tensor<T> Tensor<T>::dsigmoid(const Tensor<T> &tensor)
    requires std::is_arithmetic_v<T>
{
    return Tensor<T>::sigmoid(tensor) * (1 - Tensor<T>::sigmoid(tensor));
}

template <typename T>
Tensor<T> Tensor<T>::sinh(const Tensor<T> &tensor)
    requires std::is_arithmetic_v<T>
{
    return (Tensor<T>::exp(tensor) - Tensor<T>::exp(-tensor)) / 2;
}

template <typename T>
Tensor<T> Tensor<T>::cosh(const Tensor<T> &tensor)
    requires std::is_arithmetic_v<T>
{
    return (Tensor<T>::exp(tensor) + Tensor<T>::exp(-tensor)) / 2;
}

template <typename T>
Tensor<T> Tensor<T>::tanh(const Tensor<T> &tensor)
    requires std::is_arithmetic_v<T>
{
    return Tensor<T>::sinh(tensor) / Tensor<T>::cosh(tensor);
}

template <typename T>
Tensor<T> Tensor<T>::dtanh(const Tensor<T> &tensor)
    requires std::is_arithmetic_v<T>
{
    return 1 - Tensor<T>::tanh(Tensor<T>::tanh(tensor));
}
