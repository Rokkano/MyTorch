#include "tensor.hh"
#include "../utils.hh"

#include <cstring>
#include <cstdint>

// HEADER :
//  shape_len   : uint64_t
//  dtype_len  : uint64_t
//  buffer_len  : uint64_t

// BODY :
//  shape       : array<std::size_t>
//  dtype       : array<char>
//  buffer      : array<T>

template <typename T>
std::vector<std::byte> Tensor<T>::serialize()
{
    // std::size_t are stored as uint64_t since the byte length
    // of std::size_t is platform-dependent.
    std::vector<std::byte> buffer;

    auto write = [&]<typename U>(const U& val) {
        const auto* ptr = reinterpret_cast<const std::byte*>(&val);
        buffer.insert(buffer.end(), ptr, ptr + sizeof(U));
    };

    write.template operator()<uint64_t>(static_cast<uint64_t>(this->shape().size()));
    
    std::string dtype = type_name<T>();
    write.template operator()<uint64_t>(static_cast<uint64_t>(dtype.size()));

    write.template operator()<uint64_t>(static_cast<uint64_t>(this->buffer_.size()));

    for (std::size_t s : this->shape())
        write.template operator()<uint64_t>(static_cast<uint64_t>(s));

    for (char c : dtype)
        write.template operator()<char>(c);

    for (std::size_t i = 0; i < this->numel(); i++)
        write.template operator()<T>(this->buffer_[i]);

    return buffer;
}

template <typename T>
std::size_t Tensor<T>::deserialize(std::vector<std::byte> bytes)
{
    if (this->buffer_.size() != 0)
        throw Exception("Tensor<T>::deserialize can only be called on empty tensor.");

    std::size_t offset = 0;
    auto read = [&]<typename U>(U& val) {
        std::memcpy(&val, bytes.data() + offset, sizeof(U));
        offset += sizeof(U);
    };

    uint64_t shape_len_uint;
    read.template operator()<uint64_t>(shape_len_uint);
    std::size_t shape_len = static_cast<std::size_t>(shape_len_uint);
    
    uint64_t dtype_len_uint;
    read.template operator()<uint64_t>(dtype_len_uint);
    std::size_t dtype_len = static_cast<std::size_t>(dtype_len_uint);

    uint64_t buffer_len_uint;
    read.template operator()<uint64_t>(buffer_len_uint);
    std::size_t buffer_len = static_cast<std::size_t>(buffer_len_uint);

    
    // read shape
    std::vector<std::size_t> shape;
    uint64_t shape_tmp_uint;
    for (std::size_t _ = 0; _ < shape_len; _++)
    {
        read.template operator()<uint64_t>(shape_tmp_uint);
        shape.insert(shape.end(), static_cast<std::size_t>(shape_tmp_uint));
    }

    // read dtype
    std::string dtype;
    char dtype_tmp;
    for (std::size_t _ = 0; _ < dtype_len; _++)
    {
        read.template operator()<char>(dtype_tmp);
        dtype.insert(dtype.end(), dtype_tmp);
    }

    // Validation
    if (dtype != type_name<T>())
        throw Exception(std::format("Type mismatch : tensor was serialized with type {}, got {}.", dtype, type_name<T>()));

    // read buffer
    std::vector<T> buffer;
    T buffer_tmp;
    for (std::size_t _ = 0; _ < buffer_len; _++)
    {
        read.template operator()<T>(buffer_tmp);
        buffer.insert(buffer.end(), buffer_tmp);
    }

        
    this->buffer_ = buffer;
    this->shape_ = shape;
    return offset;
}

template <typename T>
Tensor<T> Tensor<T>::from_bytes(std::vector<std::byte> &bytes)
{
    Tensor<T> tensor = Tensor<T>({0});
    tensor.deserialize(bytes);
    return tensor;
}
