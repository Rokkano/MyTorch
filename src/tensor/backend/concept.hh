#include <concepts>
#include <type_traits>
#include <vector>

template <typename T, typename B>
concept IsBackend = requires(B::TStorage storage, std::size_t i) {
    typename B::TStorage;
    { storage[i] } -> std::same_as<T &>;

    { B::allocate(i) } -> std::same_as<typename B::TStorage>;
    { B::deallocate(storage) } -> std::same_as<void>;
    { B::size(storage) } -> std::same_as<std::size_t>;
    { B::vector(storage) } -> std::same_as<std::vector<T>>;
};
