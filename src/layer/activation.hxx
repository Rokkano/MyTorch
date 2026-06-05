#pragma once

#include "layer.hh"
#include "src/tensor/tensor.hh"
#include "src/tensor/tensor_activation.hxx"
#include "src/tensor/tensor_op.hxx"

#include "src/exception/layer.hh"

#include <optional>

template <typename T, template <typename> typename B>
class Sigmoid : public Layer<T, B>
{
    std::optional<Tensor<T, B>> inp_;

public:
    Tensor<T, B> forward(Tensor<T, B> tensor)
    {
        if (this->training_)
            this->inp_ = tensor;
        return Tensor<T, B>::sigmoid(tensor);
    }

    Tensor<T, B> backward(Tensor<T, B> gradient)
    {
        if (!this->inp_.has_value())
            throw MissingInputForBackwardException("Missing inp for backward pass of Sigmoid layer.");
        return gradient * Tensor<T, B>::dsigmoid(this->inp_.value());
    }

    Tensor<T, B> forward(std::span<Tensor<T, B>> args) override 
    {
        return this->forward(args[0]);
    };
        Tensor<T, B> backward(std::span<Tensor<T, B>> args) override
    {
        return this->backward(args[0]);
    };
};

template <typename T, template <typename> typename B>
class Tanh : public Layer<T, B>
{
    std::optional<Tensor<T, B>> inp_;

public:
    Tensor<T, B> forward(Tensor<T, B> tensor)
    {
        if (this->training_)
            this->inp_ = tensor;
        return Tensor<T, B>::tanh(tensor);
    }

    Tensor<T, B> backward(Tensor<T, B> gradient)
    {
        if (!this->inp_.has_value())
            throw MissingInputForBackwardException("Missing inp for backward pass of TanH layer.");
        return gradient * Tensor<T, B>::dtanh(this->inp_.value());
    }
    Tensor<T, B> forward(std::span<Tensor<T, B>> args) override 
    {
        return this->forward(args[0]);
    };
        Tensor<T, B> backward(std::span<Tensor<T, B>> args) override
    {
        return this->backward(args[0]);
    };
};

template <typename T, template <typename> typename B>
class ReLu : public Layer<T, B>
{
    std::optional<Tensor<T, B>> inp_;

public:
    Tensor<T, B> forward(Tensor<T, B> tensor)
    {
        if (this->training_)
            this->inp_ = tensor;
        return Tensor<T, B>::relu(tensor);
    }

    Tensor<T, B> backward(Tensor<T, B> gradient)
    {
        if (!this->inp_.has_value())
            throw MissingInputForBackwardException("Missing inp for backward pass of ReLU layer.");
        return gradient * Tensor<T, B>::drelu(this->inp_.value());
    }
    Tensor<T, B> forward(std::span<Tensor<T, B>> args) override 
    {
        return this->forward(args[0]);
    };
        Tensor<T, B> backward(std::span<Tensor<T, B>> args) override
    {
        return this->backward(args[0]);
    };
};

template <typename T, template <typename> typename B>
class Softmax : public Layer<T, B>
{
    std::optional<Tensor<T, B>> inp_;

public:
    Tensor<T, B> forward(Tensor<T, B> tensor)
    {
        if (this->training_)
            this->inp_ = tensor;
        return Tensor<T, B>::softmax(tensor);
    }

    Tensor<T, B> backward(Tensor<T, B> gradient)
    {
        if (!this->inp_.has_value())
            throw MissingInputForBackwardException("Missing inp for backward pass of Softmax layer.");
        return Tensor<T, B>::softmax(this->inp_.value()) - gradient;
    }
    Tensor<T, B> forward(std::span<Tensor<T, B>> args) override 
    {
        return this->forward(args[0]);
    };
        Tensor<T, B> backward(std::span<Tensor<T, B>> args) override
    {
        return this->backward(args[0]);
    };
};
