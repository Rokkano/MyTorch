#pragma once

#include <vector>


// template<typename T>
// concept CMTSerialize = requires(T t, std::vector<std::byte>& bytes)
// {
//     { t.serialize() }   -> std::same_as<std::vector<std::byte>>;
//     { t.deserialize(bytes) } -> std::same_as<std::size_t>;

//     // static
//     { T::from_bytes(bytes) } -> std::same_as<T>;
// };

// template <typename T>
// requires CMTSerialize<T>
// class IMTSerialize{};


// template <class E>
class IMTSerialize
{
public:
    virtual ~IMTSerialize() {};
    virtual std::vector<std::byte> serialize() = 0;
    virtual std::size_t deserialize(std::vector<std::byte> &bytes) = 0;

    // virtual E from_bytes(std::vector<std::byte> &bytes) = 0;
};