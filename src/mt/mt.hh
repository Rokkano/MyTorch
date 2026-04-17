#pragma once

#include "imt.hh"
#include "src/tensor/tensor.hh"
#include "src/utils.hh"

#include <optional>
#include <sys/types.h>
#include <vector>

inline char MTFILE_VERSION_MAJOR = 1;
inline char MTFILE_VERSION_MINOR = 0;

enum MTFILE_TYPE
{
    NONE = 0,
    TENSOR = 1,
    LAYER = 2,
    MODEL = 3,
};

template <typename T>
requires(is_base_of_template<IMTSerialize, T>::value)
class MTFile
{
public:
    static T read(std::string);
    static void write(std::string, T);

    static std::size_t sizeOfFile(std::string);
    static std::size_t sizeOfObject(T);

private:
    static std::vector<std::byte> readFile(std::string);
    static void writeFile(std::string, std::vector<std::byte>);
};

#include "mt.hxx"