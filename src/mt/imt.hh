#pragma once

#include <vector>

template <class E>
class IMTSerialize
{
public:
    virtual ~IMTSerialize() {};
    virtual std::vector<std::byte> serialize() = 0;
    virtual std::size_t deserialize(std::vector<std::byte> &bytes) = 0;

    virtual E from_bytes(std::vector<std::byte> &bytes) = 0;
};