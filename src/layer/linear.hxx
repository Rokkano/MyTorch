#pragma once

#include "layer.hh"
#include "weights.hh"

#include <optional>

template <typename T>
class Linear : public Layer<T>
{
public:
    Tensor<T> weights_;
    Tensor<T> bias_;

    std::optional<Tensor<T>> inp_;
    float learning_rate_;

public:
    Linear(std::size_t in, std::size_t out, enum Initialization initialization = ZEROS, float learning_rate = 0.001)
    {
        Tensor<T> weights = Tensor<T>({in, out});
        initialize_weights(weights, initialization);

        Tensor<T> bias = Tensor<T>({1, out});
        initialize_weights(bias, ZEROS);

        this->weights_ = weights;
        this->bias_ = bias;
        this->learning_rate_ = learning_rate;
    }

    Tensor<T> forward(Tensor<T> tensor)
    {
        if (this->training)
            this->inp_ = tensor;
        return Tensor<T>::matmul(tensor, this->weights_) + this->bias_;
    }

    Tensor<T> backward(Tensor<T> gradient)
    {
        Tensor<T> res = Tensor<T>::matmul(gradient, this->weights_.transpose());
        if (this->training)
        {
            if (!this->inp_.has_value())
                throw;

            Tensor<T> dW = Tensor<T>::matmul(gradient.transpose(), this->inp_.value());
            Tensor<T> db = gradient;

            this->weights_ = this->weights_ - this->learning_rate_ * dW.transpose();
            this->bias_ = this->bias_ - this->learning_rate_ * db;
        }
        return res;
    }
};
