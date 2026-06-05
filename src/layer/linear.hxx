#pragma once

#include "layer.hh"
#include "layer.hxx"
#include "src/mt/imt.hh"
#include "weights.hh"

#include "src/tensor/tensor_backend.hxx"
#include "src/tensor/tensor_op.hxx"
#include "src/exception/layer.hh"

#include <optional>

template <typename T, template <typename> typename B>
class Linear : public Layer<T, B>, public IMTSerialize
{
public:
    Tensor<T, B> weights_;
    Tensor<T, B> bias_;

    std::optional<Tensor<T, B>> inp_;
    float learning_rate_;

public:
    Linear();
    Linear(std::size_t in, std::size_t out, enum Initialization initialization = ZEROS, float learning_rate = 0.001);
    
    Tensor<T, B> forward(std::span<Tensor<T, B>> args) override 
    {
        return this->forward(args[0]);
    };
    Tensor<T, B> forward(Tensor<T, B> &tensor);
    
    Tensor<T, B> backward(std::span<Tensor<T, B>> args) override
    {
        return this->backward(args[0]);
    };
    Tensor<T, B> backward(Tensor<T, B> &gradient);

    // Serialization
    std::vector<std::byte> serialize() override;
    std::size_t deserialize(std::vector<std::byte> &bytes) override;
};

template <typename T, template <typename> typename B>
Linear<T, B>::Linear()
{
    this->weights_ = Tensor<T, B>();
    this->bias_ = Tensor<T, B>();
    this->learning_rate_ = 0;
    this->training_ = false;
}

template <typename T, template <typename> typename B>
Linear<T, B>::Linear(std::size_t in, std::size_t out, enum Initialization initialization, float learning_rate)
{
    Tensor<T, B> weights = Tensor<T, B>({in, out});
    initialize_weights(weights, initialization);

    Tensor<T, B> bias = Tensor<T, B>({1, out});
    initialize_weights(bias, ZEROS);

    this->weights_ = weights;
    this->bias_ = bias;
    this->learning_rate_ = learning_rate;
}

template <typename T, template <typename> typename B>
Tensor<T, B> Linear<T, B>::forward(Tensor<T, B> &tensor)
{
    if (this->training_)
        this->inp_ = tensor;
    return Tensor<T, B>::matmul(tensor, this->weights_) + this->bias_;
}

template <typename T, template <typename> typename B>
Tensor<T, B> Linear<T, B>::backward(Tensor<T, B> &gradient)
{
    Tensor<T, B> res = Tensor<T, B>::matmul(gradient, this->weights_.transpose());
    if (this->training_)
    {
        if (!this->inp_.has_value())
            throw MissingInputForBackwardException("Missing inp for backward pass of Linear layer.");

        Tensor<T, B> dW = Tensor<T, B>::matmul(gradient.transpose(), this->inp_.value());
        Tensor<T, B> db = gradient;

        this->weights_ = this->weights_ - this->learning_rate_ * dW.transpose();
        this->bias_ = this->bias_ - this->learning_rate_ * db;
    }
    return res;
}
