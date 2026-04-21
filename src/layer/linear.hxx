#pragma once

#include "layer.hh"
#include "src/mt/imt.hh"
#include "weights.hh"

#include <optional>

template <typename T, typename B>
class Linear : public Layer<T, B>, public IMTSerialize<Linear<T, B>>
{
public:
    Tensor<T, B> weights_;
    Tensor<T, B> bias_;

    std::optional<Tensor<T, B>> inp_;
    float learning_rate_;

public:
    Linear();
    Linear(std::size_t in, std::size_t out, enum Initialization initialization = ZEROS, float learning_rate = 0.001);
    Tensor<T, B> forward(Tensor<T, B> &tensor);
    Tensor<T, B> backward(Tensor<T, B> &gradient);

    // Serialization
    std::vector<std::byte> serialize() override;
    std::size_t deserialize(std::vector<std::byte> &bytes) override;
};

template <typename T, typename B>
Linear<T, B>::Linear()
{
    this->weights_ = Tensor<T, B>();
    this->bias_ = Tensor<T, B>();
    this->learning_rate_ = 0;
    this->training_ = false;
}

template <typename T, typename B>
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

template <typename T, typename B>
Tensor<T, B> Linear<T, B>::forward(Tensor<T, B> &tensor)
{
    if (this->training_)
        this->inp_ = tensor;
    return Tensor<T, B>::matmul(tensor, this->weights_) + this->bias_;
}

template <typename T, typename B>
Tensor<T, B> Linear<T, B>::backward(Tensor<T, B> &gradient)
{
    Tensor<T, B> res = Tensor<T, B>::matmul(gradient, this->weights_.transpose());
    if (this->training_)
    {
        if (!this->inp_.has_value())
            throw;

        Tensor<T, B> dW = Tensor<T, B>::matmul(gradient.transpose(), this->inp_.value());
        Tensor<T, B> db = gradient;

        this->weights_ = this->weights_ - this->learning_rate_ * dW.transpose();
        this->bias_ = this->bias_ - this->learning_rate_ * db;
    }
    return res;
}
