#pragma once

#include <cstddef>
#include <concepts>
#include <vector>

template <typename T, template <typename> typename B>
concept IsBackend = requires(B<T>::TStorage storage, std::size_t i) {
    // Storage
    typename B<T>::TStorage;
    { storage } -> std::same_as<std::vector<T> &>;

    // Base generic functions
    { B<T>::allocate(i) } -> std::same_as<typename B<T>::TStorage>;
    { B<T>::deallocate(storage) } -> std::same_as<void>;
    { B<T>::size(storage) } -> std::same_as<std::size_t>;
    { B<T>::vector(storage) } -> std::same_as<std::vector<T>>;
};
