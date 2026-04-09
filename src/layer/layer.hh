#pragma once

#include "../mt/imt.hh"
#include "../tensor/tensor.hh"

template <typename T>
class Layer : public IMTSerialize
{
protected:
    bool training_ = false;

public:
    virtual std::vector<std::byte> serialize();
    virtual std::size_t deserialize(std::vector<std::byte>);

    Tensor<T> forward(Tensor<T>...);
    Tensor<T> backward(Tensor<T>...);

    void training(bool);
};

#include "layer.hxx"
#include "activation.hxx"
#include "linear.hxx"
#include "loss.hxx"