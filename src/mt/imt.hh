#pragma once

#include <vector>

class IMTSerialize
{
    public:
        virtual ~IMTSerialize() {};
        virtual std::vector<std::byte> serialize() = 0;
        virtual std::size_t deserialize(std::vector<std::byte>) = 0;
};