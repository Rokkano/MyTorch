// #pragma once

// #include "layer.hh"

#pragma once

#include <optional>
#include "layer.hh"

// template <typename T>
// class Sigmoid: public Layer<T>
// {
// public:
//     Tensor<T> forward(Tensor<T> tensor)
//     {
//         return Tensor<T>::sigmoid(tensor);
//     }

//     Tensor<T> backward(Tensor<T> tensor)
//     {
//         return Tensor<T>::dsigmoid(tensor);
//     }
// };

template <typename T>
class Sigmoid: public Layer<T>
{
    std::optional<Tensor<T>> inp_;
public:
    Tensor<T> forward(Tensor<T> tensor)
    {
        if (this->training)
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

// template <typename T>
// class Tanh: public Layer<T>
// {
// public:
//     Tensor<T> forward(Tensor<T> tensor)
//     {
//         return Tensor<T>::tanh(tensor);
//     }

//     Tensor<T> backward(Tensor<T> tensor)
//     {
//         return Tensor<T>::dtanh(tensor);
//     }
// };

template <typename T>
class Tanh: public Layer<T>
{
    std::optional<Tensor<T>> inp_;
public:
    Tensor<T> forward(Tensor<T> tensor)
    {
        if (this->training)
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

// template <typename T>
// class ReLu: public Layer<T>
// {
// public:
//     Tensor<T> forward(Tensor<T> tensor)
//     {
//         return Tensor<T>::relu(tensor);
//     }

//     Tensor<T> backward(Tensor<T> tensor)
//     {
//         return Tensor<T>::drelu(tensor);
//     }
// };

template <typename T>
class ReLu: public Layer<T>
{
    std::optional<Tensor<T>> inp_;
public:
    Tensor<T> forward(Tensor<T> tensor)
    {
        if (this->training)
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

