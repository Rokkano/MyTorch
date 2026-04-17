#pragma once

#include "src/mt/imt.hh"
#include "src/tensor/tensor_fwd.hh"

template <typename T>
class Layer : public IMTSerialize
{
protected:
    bool training_ = false;

public:
    std::vector<std::byte> serialize() override;
    std::size_t deserialize(std::vector<std::byte> &bytes) override;

    Tensor<T> forward(Tensor<T> &...);
    Tensor<T> backward(Tensor<T> &...);

    void training(bool);
};

#include "activation.hxx"
#include "layer.hxx"
#include "linear.hxx"
#include "loss.hxx"