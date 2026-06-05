#pragma once

#include "src/tensor/tensor_fwd.hh"
#include <span>

template <typename T, template <typename> typename B>
class Layer
{
protected:
    bool training_ = false;

public:
    virtual ~Layer() {};
    
    virtual Tensor<T, B> forward(std::span<Tensor<T, B>> args) = 0;
    virtual Tensor<T, B> backward(std::span<Tensor<T, B>> args) = 0;

    void training(bool);
    bool training() const;
};