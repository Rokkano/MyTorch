#include "tensor.hh"

#include <functional>
#include <numeric>
#include <format>

template <typename T>
template <typename U>
Tensor<U> Tensor<T>::to_type()
{
    Tensor<U> tensor = Tensor<U>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = static_cast<U>(this->buffer_[i]);
    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::from_function(std::function<size_t(T)> lambda, const std::vector<std::size_t> &shape)
{
    Tensor<T> tensor = Tensor<T>(shape);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = lambda(i);
    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::from_vector(const std::vector<T> &buffer, const std::vector<std::size_t> &shape)
{
    if (std::accumulate(shape.begin(), shape.end(), 0) != buffer.size())
        throw std::invalid_argument(std::format("Lengths are incompatible : {} and {}.", std::accumulate(shape.begin(), shape.end(), 0), buffer.size()));

    Tensor<T> tensor = Tensor<T>(shape);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = buffer[i];
    return tensor;
}
