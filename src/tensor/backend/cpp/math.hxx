#pragma once

#include "backend.hh"


template <typename T>
CppBackend<T>::TStorage CppBackend<T>::affine(const TStorage &storage, const TShape &shape, std::optional<T> a, std::optional<T> b) requires std::is_arithmetic_v<T>
{
    std::size_t numel = CppBackend<T>::size(storage);
    CppBackend<T>::TStorage newStorage = CppBackend<T>::allocate(numel);
    for (std::size_t i = 0; i < numel; i++)
    {
        T value = storage[i];
        if (a.has_value())
            value = value * a.value();
        if (b.has_value())
            value = value + b.value();
        newStorage[i] = value;
    }
    return newStorage;
}

template <typename T>
CppBackend<T>::TStorage CppBackend<T>::exp(const TStorage &storage, [[maybe_unused]] const TShape &shape) requires std::is_arithmetic_v<T>
{
    std::size_t numel = CppBackend<T>::size(storage);
    CppBackend<T>::TStorage newStorage = CppBackend<T>::allocate(numel);
    for (std::size_t i = 0; i < numel; i++)
        newStorage[i] = std::exp(storage[i]);
    return newStorage;
}

template <typename T>
CppBackend<T>::TStorage CppBackend<T>::log(const TStorage &storage, [[maybe_unused]] const TShape &shape) requires std::is_arithmetic_v<T>
{
    std::size_t numel = CppBackend<T>::size(storage);
    CppBackend<T>::TStorage newStorage = CppBackend<T>::allocate(numel);
    for (std::size_t i = 0; i < numel; i++)
        newStorage[i] = std::log(storage[i]);
    return newStorage;
}

template <typename T>
CppBackend<T>::TStorage CppBackend<T>::pow(const TStorage &storage, [[maybe_unused]] const TShape &shape, double exponent) requires std::is_arithmetic_v<T>
{
    std::size_t numel = CppBackend<T>::size(storage);
    CppBackend<T>::TStorage newStorage = CppBackend<T>::allocate(numel);
    for (std::size_t i = 0; i < numel; i++)
        newStorage[i] = std::pow(storage[i], exponent);
    return newStorage;
}
template <typename T>
CppBackend<T>::TStorage CppBackend<T>::sqrt(const TStorage &storage, [[maybe_unused]] const TShape &shape) requires std::is_arithmetic_v<T>
{
    std::size_t numel = CppBackend<T>::size(storage);
    CppBackend<T>::TStorage newStorage = CppBackend<T>::allocate(numel);
    for (std::size_t i = 0; i < numel; i++)
        newStorage[i] = std::sqrt(storage[i]);
    return newStorage;
}