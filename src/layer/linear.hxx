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
    Linear();
    Linear(std::size_t in, std::size_t out, enum Initialization initialization = ZEROS, float learning_rate = 0.001);
    Tensor<T> forward(Tensor<T> tensor);
    Tensor<T> backward(Tensor<T> gradient);

    // Serialization
    virtual std::vector<std::byte> serialize();
    virtual std::size_t deserialize(std::vector<std::byte>);
};

template <typename T>
Linear<T>::Linear()
{
    this->weights_ = Tensor<T>();
    this->bias_ = Tensor<T>();
    this->learning_rate_ = 0;
    this->training_ = false;
}

template <typename T>
Linear<T>::Linear(std::size_t in, std::size_t out, enum Initialization initialization, float learning_rate)
{
    Tensor<T> weights = Tensor<T>({in, out});
    initialize_weights(weights, initialization);

    Tensor<T> bias = Tensor<T>({1, out});
    initialize_weights(bias, ZEROS);

    this->weights_ = weights;
    this->bias_ = bias;
    this->learning_rate_ = learning_rate;
}

template <typename T>
Tensor<T> Linear<T>::forward(Tensor<T> tensor)
{
    if (this->training_)
        this->inp_ = tensor;
    return Tensor<T>::matmul(tensor, this->weights_) + this->bias_;
}

template <typename T>
Tensor<T> Linear<T>::backward(Tensor<T> gradient)
{
    Tensor<T> res = Tensor<T>::matmul(gradient, this->weights_.transpose());
    if (this->training_)
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

template <typename T>
std::vector<std::byte> Linear<T>::serialize()
{
    std::vector<std::byte> buffer;

    auto write = [&]<typename U>(const U& val) {
        const auto* ptr = reinterpret_cast<const std::byte*>(&val);
        buffer.insert(buffer.end(), ptr, ptr + sizeof(U));
    };

    std::vector<std::byte> weightBuffer = this->weights_.serialize();
    buffer.insert(buffer.end(), weightBuffer.begin(), weightBuffer.end());

    std::vector<std::byte> biasBuffer = this->bias_.serialize();
    buffer.insert(buffer.end(), biasBuffer.begin(), biasBuffer.end());

    write.template operator()<float>(this->learning_rate_);

    return buffer;
}

template <typename T>
std::size_t Linear<T>::deserialize(std::vector<std::byte> buffer)
{
    std::size_t offset = 0;
    auto read = [&]<typename U>(U& val) {
        std::memcpy(&val, buffer.data() + offset, sizeof(U));
        offset += sizeof(U);
    };

    offset += this->weights_.deserialize(std::vector<std::byte>(buffer.begin() + offset, buffer.end()));
    offset += this->bias_.deserialize(std::vector<std::byte>(buffer.begin() + offset, buffer.end()));
    
    float learning_rate;
    read.template operator()<float>(learning_rate);

    return offset;
}
