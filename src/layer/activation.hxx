#pragma once

#include "layer.hh"

#include <optional>

template <typename T>
class Sigmoid : public Layer<T>
{
    std::optional<Tensor<T>> inp_;

public:
    Tensor<T> forward(Tensor<T> tensor)
    {
        if (this->training_)
            this->inp_ = tensor;
        return Tensor<T>::sigmoid(tensor);
    }

    Tensor<T> backward(Tensor<T> gradient)
    {
        if (!this->inp_.has_value())
            throw;
        return gradient * Tensor<T>::dsigmoid(this->inp_.value());
    }
};

template <typename T>
class Tanh : public Layer<T>
{
    std::optional<Tensor<T>> inp_;

public:
    Tensor<T> forward(Tensor<T> tensor)
    {
        if (this->training_)
            this->inp_ = tensor;
        return Tensor<T>::tanh(tensor);
    }

    Tensor<T> backward(Tensor<T> gradient)
    {
        if (!this->inp_.has_value())
            throw;
        return gradient * Tensor<T>::dtanh(this->inp_.value());
    }
};

template <typename T>
class ReLu : public Layer<T>
{
    std::optional<Tensor<T>> inp_;

public:
    Tensor<T> forward(Tensor<T> tensor)
    {
        if (this->training_)
            this->inp_ = tensor;
        return Tensor<T>::relu(tensor);
    }

    Tensor<T> backward(Tensor<T> gradient)
    {
        if (!this->inp_.has_value())
            throw;
        return gradient * Tensor<T>::drelu(this->inp_.value());
    }
};

template <typename T>
class Softmax : public Layer<T>
{
    std::optional<Tensor<T>> inp_;

public:
    Tensor<T> forward(Tensor<T> tensor)
    {
        if (this->training_)
            this->inp_ = tensor;
        return Tensor<T>::softmax(tensor);
    }

    Tensor<T> backward(Tensor<T> gradient)
    {
        if (!this->inp_.has_value())
            throw;
        return Tensor<T>::softmax(this->inp_.value()) - gradient;
    }
};
