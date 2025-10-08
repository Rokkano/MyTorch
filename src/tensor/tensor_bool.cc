#include <format>

#include "tensor.hh"

template <>
Tensor<bool>::operator bool() const
{
    if (this->numel() != 1)
        throw CastException("Boolean cast of a non single element Tensor<bool> is ambiguous. Use .all() or .any() depending on the behaviour you want.");
    else
        return this->buffer_[0];
}

template <>
Tensor<bool> Tensor<bool>::all() const
{
    for (std::size_t i = 0; i < this->numel(); i++)
        if (!this->buffer_[i])
            return Tensor<bool>({1}, {false});
    return Tensor<bool>({1}, {true});
}

template <>
Tensor<bool> Tensor<bool>::any() const
{
    for (std::size_t i = 0; i < this->numel(); i++)
        if (this->buffer_[i])
            return Tensor<bool>({1}, {true});
    return Tensor<bool>({1}, {false});
}

template <>
Tensor<bool> Tensor<bool>::none() const
{
    for (std::size_t i = 0; i < this->numel(); i++)
        if (this->buffer_[i])
            return Tensor<bool>({1}, {false});
    return Tensor<bool>({1}, {true});
}