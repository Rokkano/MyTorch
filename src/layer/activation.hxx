#pragma once

#include "layer.hh"

#include <optional>

template <typename T, typename B>
class Sigmoid : public Layer<T, B>
{
    std::optional<Tensor<T, B>> inp_;

public:
    Tensor<T, B> forward(Tensor<T, B> tensor)
    {
        if (this->training_)
            this->inp_ = tensor;
        return Tensor<T, B>::sigmoid(tensor);
    }

    Tensor<T, B> backward(Tensor<T, B> gradient)
    {
        if (!this->inp_.has_value())
            throw;
        return gradient * Tensor<T, B>::dsigmoid(this->inp_.value());
    }
};

template <typename T, typename B>
class Tanh : public Layer<T, B>
{
    std::optional<Tensor<T, B>> inp_;

public:
    Tensor<T, B> forward(Tensor<T, B> tensor)
    {
        if (this->training_)
            this->inp_ = tensor;
        return Tensor<T, B>::tanh(tensor);
    }

    Tensor<T, B> backward(Tensor<T, B> gradient)
    {
        if (!this->inp_.has_value())
            throw;
        return gradient * Tensor<T, B>::dtanh(this->inp_.value());
    }
};

template <typename T, typename B>
class ReLu : public Layer<T, B>
{
    std::optional<Tensor<T, B>> inp_;

public:
    Tensor<T, B> forward(Tensor<T, B> tensor)
    {
        if (this->training_)
            this->inp_ = tensor;
        return Tensor<T, B>::relu(tensor);
    }

    Tensor<T, B> backward(Tensor<T, B> gradient)
    {
        if (!this->inp_.has_value())
            throw;
        return gradient * Tensor<T, B>::drelu(this->inp_.value());
    }
};

template <typename T, typename B>
class Softmax : public Layer<T, B>
{
    std::optional<Tensor<T, B>> inp_;

public:
    Tensor<T, B> forward(Tensor<T, B> tensor)
    {
        if (this->training_)
            this->inp_ = tensor;
        return Tensor<T, B>::softmax(tensor);
    }

    Tensor<T, B> backward(Tensor<T, B> gradient)
    {
        if (!this->inp_.has_value())
            throw;
        return Tensor<T, B>::softmax(this->inp_.value()) - gradient;
    }
};
