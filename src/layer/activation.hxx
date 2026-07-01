#pragma once

#include "layer.hh"
#include "src/exception/layer.hh"
#include "src/tensor/backend/backend_fwd.hh"
#include "src/tensor/tensor.hh"
#include "src/tensor/tensor_activation.hxx"
#include "src/tensor/tensor_op.hxx"

#include <optional>

template <typename T, template <typename> typename B = DefaultBackend>
class Sigmoid : public Layer<T, B>
{
    std::optional<Tensor<T, B>> inp_;

public:
    Tensor<T, B> forward(Tensor<T, B>::TensorSpan args) override
    {
        if (args.size() != 1)
            throw TensorInvalidArgException("Forward argument does not match this layer.");
        const Tensor<T, B> &tensor = args[0].get();

        if (this->training_)
            this->inp_ = tensor;
        return Tensor<T, B>::sigmoid(tensor);
    }

    Tensor<T, B> backward(Tensor<T, B>::TensorSpan args) override
    {
        if (args.size() != 1)
            throw TensorInvalidArgException("Backward argument does not match this layer.");
        const Tensor<T, B> &gradient = args[0].get();

        if (!this->inp_.has_value())
            throw MissingInputForBackwardException("Missing inp for backward pass of Sigmoid layer.");
        return gradient * Tensor<T, B>::dsigmoid(this->inp_.value());
    }
};

template <typename T, template <typename> typename B = DefaultBackend>
class Tanh : public Layer<T, B>
{
    std::optional<Tensor<T, B>> inp_;

public:
    Tensor<T, B> forward(Tensor<T, B>::TensorSpan args) override
    {
        if (args.size() != 1)
            throw TensorInvalidArgException("Forward argument does not match this layer.");
        const Tensor<T, B> &tensor = args[0].get();

        if (this->training_)
            this->inp_ = tensor;
        return Tensor<T, B>::tanh(tensor);
    }

    Tensor<T, B> backward(Tensor<T, B>::TensorSpan args) override
    {
        if (args.size() != 1)
            throw TensorInvalidArgException("Backward argument does not match this layer.");
        const Tensor<T, B> &gradient = args[0].get();

        if (!this->inp_.has_value())
            throw MissingInputForBackwardException("Missing inp for backward pass of TanH layer.");
        return gradient * Tensor<T, B>::dtanh(this->inp_.value());
    }
};

template <typename T, template <typename> typename B = DefaultBackend>
class ReLu : public Layer<T, B>
{
    std::optional<Tensor<T, B>> inp_;

public:
    Tensor<T, B> forward(Tensor<T, B>::TensorSpan args) override
    {
        if (args.size() != 1)
            throw TensorInvalidArgException("Forward argument does not match this layer.");
        const Tensor<T, B> &tensor = args[0].get();

        if (this->training_)
            this->inp_ = tensor;
        return Tensor<T, B>::relu(tensor);
    }

    Tensor<T, B> backward(Tensor<T, B>::TensorSpan args) override
    {
        if (args.size() != 1)
            throw TensorInvalidArgException("Backward argument does not match this layer.");
        const Tensor<T, B> &gradient = args[0].get();

        if (!this->inp_.has_value())
            throw MissingInputForBackwardException("Missing inp for backward pass of ReLU layer.");
        return gradient * Tensor<T, B>::drelu(this->inp_.value());
    }
};

template <typename T, template <typename> typename B = DefaultBackend>
class Softmax : public Layer<T, B>
{
    std::optional<Tensor<T, B>> inp_;

public:
    Tensor<T, B> forward(Tensor<T, B>::TensorSpan args) override
    {
        if (args.size() != 1)
            throw TensorInvalidArgException("Forward argument does not match this layer.");
        const Tensor<T, B> &tensor = args[0].get();

        if (this->training_)
            this->inp_ = tensor;
        return Tensor<T, B>::softmax(tensor);
    }

    Tensor<T, B> backward(Tensor<T, B>::TensorSpan args) override
    {
        if (args.size() != 1)
            throw TensorInvalidArgException("Backward argument does not match this layer.");
        const Tensor<T, B> &gradient = args[0].get();

        if (!this->inp_.has_value())
            throw MissingInputForBackwardException("Missing inp for backward pass of Softmax layer.");
        return Tensor<T, B>::softmax(this->inp_.value()) - gradient;
    }
};
