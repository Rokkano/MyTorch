#pragma once

#include "src/tensor/tensor_fwd.hh"

template <typename T, typename B>
class Layer
{
protected:
    bool training_ = false;

public:
    virtual Tensor<T, B> forward(Tensor<T, B> &...);
    virtual Tensor<T, B> backward(Tensor<T, B> &...);

    void training(bool);
};

#include "activation.hxx"
#include "layer.hxx"
#include "linear.hxx"
#include "loss.hxx"