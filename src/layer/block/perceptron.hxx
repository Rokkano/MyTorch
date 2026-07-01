#pragma once

#include "src/layer/activation.hxx"
#include "src/layer/layer.hh"
#include "src/layer/linear.hxx"
#include "src/tensor/backend/backend_fwd.hh"

#include <cstring>
#include <optional>

template <typename T, template <typename> typename B = DefaultBackend>
class MultiLayerPerceptron : public Layer<T, B>
{
private:
    std::vector<std::unique_ptr<Layer<T, B>>> layers_ = std::vector<std::unique_ptr<Layer<T, B>>>();
    float learning_rate_ = 0;

public:
    MultiLayerPerceptron();
    MultiLayerPerceptron(std::size_t in, std::size_t out, std::size_t hid, std::size_t hidNum = 1,
                         enum Initialization hidInitialization = Initialization::ZEROS,
                         enum Initialization lastInitialization = Initialization::ZEROS, float learning_rate = 0.001);

    Tensor<T, B> forward(Tensor<T, B>::TensorSpan args) override;
    Tensor<T, B> backward(Tensor<T, B>::TensorSpan args) override;

    void training(bool flag);

    // Serialization
    virtual std::vector<std::byte> serialize();
    virtual std::size_t deserialize(std::vector<std::byte> &bytes);
};

template <typename T, template <typename> typename B>
MultiLayerPerceptron<T, B>::MultiLayerPerceptron()
{
}

template <typename T, template <typename> typename B>
MultiLayerPerceptron<T, B>::MultiLayerPerceptron(std::size_t in, std::size_t out, std::size_t hid, std::size_t hidNum,
                                                 enum Initialization hidInitialization,
                                                 enum Initialization lastInitialization, float learning_rate)
{
    // in layer
    this->layers_.push_back(std::make_unique<Linear<T, B>>(in, hid, hidInitialization, learning_rate));
    this->layers_.push_back(std::make_unique<ReLu<T, B>>());

    // hid layers (potentially 0)
    for (std::size_t index = 0; index < hidNum; index++)
    {
        this->layers_.push_back(std::make_unique<Linear<T, B>>(hid, hid, hidInitialization, learning_rate));
        this->layers_.push_back(std::make_unique<ReLu<T, B>>());
    }

    // out layer (without activation)
    this->layers_.push_back(std::make_unique<Linear<T, B>>(hid, out, lastInitialization, learning_rate));
}

template <typename T, template <typename> typename B>
Tensor<T, B> MultiLayerPerceptron<T, B>::forward(Tensor<T, B>::TensorSpan args)
{
    if (args.size() != 1)
        throw TensorInvalidArgException("Forward argument does not match this layer.");
    const Tensor<T, B> &tensor = args[0].get();

    Tensor<T, B> output = tensor;
    for (std::size_t index = 0; index < this->layers_.size(); index++)
        output = this->layers_[index]->forward({output});
    return output;
}

template <typename T, template <typename> typename B>
Tensor<T, B> MultiLayerPerceptron<T, B>::backward(Tensor<T, B>::TensorSpan args)
{
    if (args.size() != 1)
        throw TensorInvalidArgException("Backward argument does not match this layer.");
    const Tensor<T, B> &gradient = args[0].get();

    Tensor<T, B> output = gradient;
    for (int index = this->layers_.size() - 1; index >= 0; index--)
        output = this->layers_[index]->backward({output});
    return output;
}

template <typename T, template <typename> typename B>
void MultiLayerPerceptron<T, B>::training(bool flag)
{
    for (std::size_t index = 0; index < this->layers_.size(); index++)
        this->layers_[index]->training(flag);
}

// HEADER :
//  hid_len   : uint64_t

// BODY :
//  buffer      : array<Linear<T>>

template <typename T, template <typename> typename B>
std::vector<std::byte> MultiLayerPerceptron<T, B>::serialize()
{
    std::vector<std::byte> buffer;

    // auto write = [&]<typename U>(const U &val)
    // {
    //     const auto *ptr = reinterpret_cast<const std::byte *>(&val);
    //     buffer.insert(buffer.end(), ptr, ptr + sizeof(U));
    // };

    // write.template operator()<uint64_t>(static_cast<uint64_t>(this->layers_.size()));

    // std::vector<std::byte> layerBuffer;
    // for (std::size_t index = 0; index < this->layers_.size(); index++)
    // {
    //     layerBuffer = this->layers_[index].serialize();
    //     buffer.insert(buffer.end(), layerBuffer.begin(), layerBuffer.end());
    //     layerBuffer.clear();
    // }

    return buffer;
}

template <typename T, template <typename> typename B>
std::size_t MultiLayerPerceptron<T, B>::deserialize(std::vector<std::byte> &bytes)
{
    std::size_t offset = 0;
    (void)bytes;
    // auto read = [&]<typename U>(U &val)
    // {
    //     std::memcpy(&val, bytes.data() + offset, sizeof(U));
    //     offset += sizeof(U);
    // };

    // uint64_t hidden_len_uint;
    // read.template operator()<uint64_t>(hidden_len_uint);
    // std::size_t hidden_len = static_cast<std::size_t>(hidden_len_uint);

    // for (std::size_t index = 0; index < hidden_len; index++)
    // {
    //     this->layers_.insert(this->layers_.end(), Linear<T, B>());
    //     std::vector<std::byte> layer_bytes = std::vector<std::byte>(bytes.begin() + offset, bytes.end());
    //     offset += this->layers_[index].deserialize(layer_bytes);

    //     if (index != hidden_len - 1)
    //         this->activations_.insert(this->activations_.end(), ReLu<T, B>());
    // }

    return offset;
}
