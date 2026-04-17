#include "mt.hh"

#include <cstddef>
#include <cstring>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

//  Header :
//  mt version : 2 char -> 1.0
//  mt type : char
//  object

template <typename T>
requires(is_base_of_template<IMTSerialize, T>::value)
std::vector<std::byte> MTFile<T>::readFile(std::string path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file)
        throw std::runtime_error("Cannot open file : " + path);

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<std::byte> buffer(size);

    if (!file.read(reinterpret_cast<char *>(buffer.data()), size))
        throw std::runtime_error("Cannot read file : " + path);

    return buffer;
}

template <typename T>
requires(is_base_of_template<IMTSerialize, T>::value)
void MTFile<T>::writeFile(std::string path, std::vector<std::byte> buffer)
{
    std::ofstream file(path, std::ios::binary);

    if (!file)
        throw std::runtime_error("Impossible d'ouvrir le fichier : " + path);

    if (!file.write(reinterpret_cast<const char *>(buffer.data()), buffer.size()))
        throw std::runtime_error("Erreur d'écriture du fichier : " + path);
}

template <typename T>
requires(is_base_of_template<IMTSerialize, T>::value)
T MTFile<T>::read(std::string path)
{
    std::vector<std::byte> buffer = MTFile<T>::readFile(path);

    std::size_t offset = 0;
    auto read = [&]<typename U>(U &val)
    {
        std::memcpy(&val, buffer.data() + offset, sizeof(U));
        offset += sizeof(U);
    };

    char mtFileVersionMajor;
    read.template operator()<char>(mtFileVersionMajor);

    char mtFileVersionMinor;
    read.template operator()<char>(mtFileVersionMinor);

    if (mtFileVersionMajor != MTFILE_VERSION_MAJOR || mtFileVersionMinor != MTFILE_VERSION_MINOR)
        throw std::invalid_argument(std::format("MTFILE_VERSION mismatch : expected {}.{}, got {}.{}",
                                                MTFILE_VERSION_MAJOR, MTFILE_VERSION_MINOR, mtFileVersionMajor,
                                                mtFileVersionMinor));

    char mtType;
    read.template operator()<char>(mtType);

    if (mtType == MTFILE_TYPE::TENSOR && !(is_base_of_template<Tensor, T>::value))
        throw Exception(std::format("Expected Tensor type, got {}.", type_name<T>()));

    std::vector<std::byte> subBuffer;
    std::span span = std::span(buffer).subspan(offset);
    subBuffer.assign(span.begin(), span.end());

    T object;
    object.deserialize(subBuffer);
    return object;
}

template <typename T>
requires(is_base_of_template<IMTSerialize, T>::value)
void MTFile<T>::write(std::string path, T object)
{
    std::vector<std::byte> buffer;

    auto write = [&]<typename U>(const U &val)
    {
        const auto *ptr = reinterpret_cast<const std::byte *>(&val);
        buffer.insert(buffer.end(), ptr, ptr + sizeof(U));
    };

    write.template operator()<char>(MTFILE_VERSION_MAJOR);
    write.template operator()<char>(MTFILE_VERSION_MINOR);

    if (is_base_of_template<Tensor, T>::value)
        write.template operator()<char>(MTFILE_TYPE::TENSOR);
    else
        write.template operator()<char>(MTFILE_TYPE::NONE);

    std::vector<std::byte> objectBuffer = object.serialize();
    buffer.insert(buffer.end(), objectBuffer.begin(), objectBuffer.end());

    MTFile<T>::writeFile(path, buffer);
}

template <typename T>
requires(is_base_of_template<IMTSerialize, T>::value)
std::size_t MTFile<T>::sizeOfFile(std::string path)
{
    std::vector<std::byte> buffer = MTFile<T>::readFile(path);
    return buffer.size();
}

template <typename T>
requires(is_base_of_template<IMTSerialize, T>::value)
std::size_t MTFile<T>::sizeOfObject(T object)
{
    std::vector<std::byte> buffer = object.serialize();
    return buffer.size();
}