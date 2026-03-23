#pragma once

#include "layer.hh"

template <typename T>
class MeanSquaredErrorLoss: public Layer<T>
{
public:
    Tensor<T> forward(Tensor<T> pred, Tensor<T> gt)
    {
        Tensor<T> sum = Tensor<T>::pow(pred - gt, 2).sum();
        // 0.5 is for an easier gradient compute
        // See https://aew61.github.io/blog/artificial_neural_networks/1_background/1.c_loss_functions_and_derivatives.html
        return 1.f / (2.f * (float)pred.numel()) * sum;
    }

    Tensor<T> backward(Tensor<T> pred, Tensor<T> gt)
    {
        return (pred - gt) / (float)pred.numel();
    }
};

template <typename T>
using L2Loss = MeanSquaredErrorLoss<T>;

template <typename T>
class CrossEntropyLoss: public Layer<T>
{
public:
    Tensor<T> forward(Tensor<T> pred, Tensor<T> gt)
    {
        return - (gt * Tensor<T>::log(pred) + (1.f - gt) * Tensor<T>::log(1.f - pred));
    }

    Tensor<T> backward(Tensor<T> pred, Tensor<T> gt)
    {
        return (pred - gt) / (pred * (1.f - pred));
    }
};

template <typename T>
class BinaryCrossEntropyLoss: public Layer<T>
{
public:
    Tensor<T> forward(Tensor<T> pred, Tensor<T> gt)
    {
        return - (gt * Tensor<T>::log(pred) + (1.f - gt) * Tensor<T>::log(1.f - pred)).mean();
    }

    Tensor<T> backward(Tensor<T> pred, Tensor<T> gt)
    {
        return - (gt / pred - (1.f - gt)/(1.f - pred));
    }
};