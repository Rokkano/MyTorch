#include "layer.hh"

template <typename T>
std::vector<std::byte> Linear<T>::serialize()
{
    std::vector<std::byte> buffer;

    auto write = [&]<typename U>(const U &val)
    {
        const auto *ptr = reinterpret_cast<const std::byte *>(&val);
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
std::size_t Linear<T>::deserialize(std::vector<std::byte> &bytes)
{
    std::size_t offset = 0;
    auto read = [&]<typename U>(U &val)
    {
        std::memcpy(&val, bytes.data() + offset, sizeof(U));
        offset += sizeof(U);
    };

    std::vector<std::byte> weights_bytes = std::vector<std::byte>(bytes.begin() + offset, bytes.end());
    offset += this->weights_.deserialize(weights_bytes);

    std::vector<std::byte> bias_bytes = std::vector<std::byte>(bytes.begin() + offset, bytes.end());
    offset += this->bias_.deserialize(bias_bytes);

    float learning_rate;
    read.template operator()<float>(learning_rate);

    return offset;
}