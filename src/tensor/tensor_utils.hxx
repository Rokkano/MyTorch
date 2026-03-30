#include "tensor.hh"

#include <format>
#include <functional>
#include <numeric>

template <typename T>
template <typename U>
Tensor<U> Tensor<T>::to_type()
{
    Tensor<U> tensor = Tensor<U>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
    {
        try
        {
            tensor.buffer_[i] = static_cast<U>(this->buffer_[i]);
        }
        catch (const std::exception &e)
        {
            throw CastException(e.what());
        }
    }
    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::from_function(std::function<T(std::size_t)> lambda, const std::vector<std::size_t> &shape)
{
    Tensor<T> tensor = Tensor<T>(shape);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = lambda(i);
    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::from_vector(const std::vector<T> &buffer, const std::vector<std::size_t> &shape)
{
    std::size_t num_e = std::size_t(std::accumulate(shape.begin(), shape.end(), 1.0, std::multiplies<double>()));
    if (num_e != buffer.size())
        throw TensorInvalidShapeException(
            std::format("Buffer length and shape are incompatible : {} and {}.", num_e, buffer.size()));

    Tensor<T> tensor = Tensor<T>(shape);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = buffer[i];
    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::one_hot(std::size_t index, const std::vector<std::size_t> &shape)
{
    auto oh = [index](std::size_t x) { return (x == index) ? 1 : 0; };
    return Tensor<T>::from_function(oh, shape);
}