#pragma once

#include "src/tensor/tensor_fwd.hh"
#include "src/tensor/backend/backend_fwd.hh"
#include <span>

template <typename T, template <typename> typename B = DefaultBackend>
class Layer
{
protected:
    bool training_ = false;

public:
    virtual ~Layer() = default;
    
    virtual Tensor<T, B> forward(Tensor<T, B>::TensorSpan args) = 0;
    virtual Tensor<T, B> backward(Tensor<T, B>::TensorSpan args) = 0;

    void training(bool);
    bool training() const;
};