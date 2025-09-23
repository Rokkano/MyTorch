#include <format>

#include "tensor.hh"

template <>
Tensor<bool>::operator bool() const
{
    if (this->numel() != 1)
        throw CastException("Boolean cast of a Tensor is ambiguous. Use .all() or .any() depending on the behaviour you want.");
    else
        return this->buffer_[0];
}

template <>
bool Tensor<bool>::all() const
{
    for (std::size_t i = 0; i < this->numel(); i++)
        if (!this->buffer_[i])
            return false;
    return true;
}

template <>
bool Tensor<bool>::any() const
{
    for (std::size_t i = 0; i < this->numel(); i++)
        if (this->buffer_[i])
            return true;
    return false;
}

template <>
bool Tensor<bool>::none() const
{
    for (std::size_t i = 0; i < this->numel(); i++)
        if (this->buffer_[i])
            return false;
    return true;
}