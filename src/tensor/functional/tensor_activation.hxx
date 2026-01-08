#include "../tensor.hh"

template <typename T>
Tensor<T> Tensor<T>::relu(const Tensor<T> &tensor)
    requires std::is_arithmetic_v<T>
{
    Tensor<T> full0 = Tensor<T>(tensor.shape_);
    full0.fill(0);
    return tensor.min(full0);
}

template <typename T>
Tensor<T> Tensor<T>::sigmoid(const Tensor<T> &tensor)
    requires std::is_arithmetic_v<T>
{
    return 1 / (1 + Tensor<T>::exp(-tensor));
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
