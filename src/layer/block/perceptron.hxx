#pragma once

#include "src/layer/layer.hh"

#include <optional>

template <typename T, typename B>
class MultiLayerPerceptron : public Layer<T, B>
{
private:
    std::vector<Linear<T, B>> layers_ = std::vector<Linear<T, B>>();
    std::vector<ReLu<T, B>> activations_ = std::vector<ReLu<T, B>>();

public:
    std::optional<Tensor<T, B>> inp_;
    float learning_rate_ = 0;

public:
    MultiLayerPerceptron();
    MultiLayerPerceptron(std::size_t in, std::size_t out, std::size_t hid, std::size_t hidNum = 1,
                         enum Initialization hidInitialization = Initialization::ZEROS,
                         enum Initialization lastInitialization = Initialization::ZEROS, float learning_rate = 0.001);

    Tensor<T, B> forward(Tensor<T, B> &tensor);
    Tensor<T, B> backward(Tensor<T, B> &gradient);

    void training(bool flag);

    // Serialization
    virtual std::vector<std::byte> serialize();
    virtual std::size_t deserialize(std::vector<std::byte> &bytes);
};

template <typename T, typename B>
MultiLayerPerceptron<T, B>::MultiLayerPerceptron()
{
}

template <typename T, typename B>
MultiLayerPerceptron<T, B>::MultiLayerPerceptron(std::size_t in, std::size_t out, std::size_t hid, std::size_t hidNum,
                                                 enum Initialization hidInitialization,
                                                 enum Initialization lastInitialization, float learning_rate)
{
    this->layers_.insert(this->layers_.end(), Linear<T, B>(in, hid, hidInitialization, learning_rate));
    this->activations_.insert(this->activations_.end(), ReLu<T, B>());

    for (std::size_t index = 0; index < hidNum; index++)
    {
        this->layers_.insert(this->layers_.end(), Linear<T, B>(hid, hid, hidInitialization, learning_rate));
        this->activations_.insert(this->activations_.end(), ReLu<T, B>());
    }

    this->layers_.insert(this->layers_.end(), Linear<T, B>(hid, out, lastInitialization, learning_rate));
}

template <typename T, typename B>
Tensor<T, B> MultiLayerPerceptron<T, B>::forward(Tensor<T, B> &tensor)
{
    if (this->training_)
        this->inp_ = tensor;
    Tensor<T, B> output = tensor;
    for (std::size_t index = 0; index < this->activations_.size(); index++)
    {
        output = this->layers_[index].forward(output);
        output = this->activations_[index].forward(output);
    }
    return this->layers_[this->layers_.size() - 1].forward(output);
}

template <typename T, typename B>
Tensor<T, B> MultiLayerPerceptron<T, B>::backward(Tensor<T, B> &gradient)
{
    Tensor<T, B> output = this->layers_[this->layers_.size() - 1].backward(gradient);
    for (std::size_t index = 0; index < this->activations_.size(); index++)
    {
        output = this->activations_[this->activations_.size() - index - 1].backward(output);
        output = this->layers_[this->activations_.size() - index - 1].backward(output);
    }
    return output;
}

template <typename T, typename B>
void MultiLayerPerceptron<T, B>::training(bool flag)
{
    for (std::size_t index = 0; index < this->layers_.size(); index++)
        this->layers_[index].training(flag);
    for (std::size_t index = 0; index < this->activations_.size(); index++)
        this->activations_[index].training(flag);
}

// HEADER :
//  hid_len   : uint64_t

// BODY :
//  buffer      : array<Linear<T>>

template <typename T, typename B>
std::vector<std::byte> MultiLayerPerceptron<T, B>::serialize()
{
    std::vector<std::byte> buffer;

    auto write = [&]<typename U>(const U &val)
    {
        const auto *ptr = reinterpret_cast<const std::byte *>(&val);
        buffer.insert(buffer.end(), ptr, ptr + sizeof(U));
    };

    write.template operator()<uint64_t>(static_cast<uint64_t>(this->layers_.size()));

    std::vector<std::byte> layerBuffer;
    for (std::size_t index = 0; index < this->layers_.size(); index++)
    {
        layerBuffer = this->layers_[index].serialize();
        buffer.insert(buffer.end(), layerBuffer.begin(), layerBuffer.end());
        layerBuffer.clear();
    }

    return buffer;
}

template <typename T, typename B>
std::size_t MultiLayerPerceptron<T, B>::deserialize(std::vector<std::byte> &bytes)
{
    std::size_t offset = 0;
    auto read = [&]<typename U>(U &val)
    {
        std::memcpy(&val, bytes.data() + offset, sizeof(U));
        offset += sizeof(U);
    };

    uint64_t hidden_len_uint;
    read.template operator()<uint64_t>(hidden_len_uint);
    std::size_t hidden_len = static_cast<std::size_t>(hidden_len_uint);

    for (std::size_t index = 0; index < hidden_len; index++)
    {
        this->layers_.insert(this->layers_.end(), Linear<T, B>());
        std::vector<std::byte> layer_bytes = std::vector<std::byte>(bytes.begin() + offset, bytes.end());
        offset += this->layers_[index].deserialize(layer_bytes);

        if (index != hidden_len - 1)
            this->activations_.insert(this->activations_.end(), ReLu<T, B>());
    }

    return offset;
}
