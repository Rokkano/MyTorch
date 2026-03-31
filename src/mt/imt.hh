#pragma once

#include <vector>

class IMTSerialize
{
    public:
        virtual ~IMTSerialize() {};
        virtual std::vector<std::byte> serialize() = 0;
        virtual void deserialize(std::vector<std::byte>) = 0;
};