#pragma once

#include <vector>
#include <sys/types.h>

#include "../utils.hh"

class IMTSerialize
{
    public:
        virtual ~IMTSerialize() {};
        virtual std::vector<std::byte> serialize() = 0;
        virtual void deserialize(std::vector<std::byte>) = 0;
};

inline uint MTFILE_VERSION_MAJOR = 1;
inline uint MTFILE_VERSION_MINOR = 0;

template <typename T>
class MTFile
{
private:
    std::string path;

public:
    MTFile(std::string &);
    T read();
    void write(T);
};

template <typename T>
inline void MTFile<T>::write(T)
{
}
