#pragma once

#include "../tensor/tensor.hh"

template <typename T>
class Layer
{
public:
    bool training = false;
    
public:
    Tensor<T> forward(Tensor<T>...);
    Tensor<T> backward(Tensor<T>...);
};

#include "activation.hxx"
#include "linear.hxx"
#include "loss.hxx"