#pragma once

#include "src/tensor/tensor_fwd.hh"

#include <optional>

template <typename T>
struct CppBackend
{
    using TStorage = std::vector<T>;
    using TShape = std::vector<std::size_t>;

    static TStorage allocate(std::size_t n) { return TStorage(n); };
    static void deallocate(TStorage &s) {};
    static std::size_t size(const TStorage &s) { return s.size(); };
    static std::vector<T> vector(const TStorage &s) { return std::vector<T>(s); };

    static TStorage affine(const TStorage &storage, const TShape &shape, std::optional<T> a, std::optional<T> b)
        requires std::is_arithmetic_v<T>;
    static TStorage exp(const TStorage &storage, const TShape &shape) requires std::is_arithmetic_v<T>;
    static TStorage log(const TStorage &storage, const TShape &shape) requires std::is_arithmetic_v<T>;
    static TStorage pow(const TStorage &storage, const TShape &shape, double exponent) requires std::is_arithmetic_v<T>;
    static TStorage sqrt(const TStorage &storage, const TShape &shape) requires std::is_arithmetic_v<T>;

    static TStorage dot(const TStorage &lhsStorage, const TShape &lhsShape, const TStorage &rhsStorage,
                        const TShape &rhsShape) requires std::is_arithmetic_v<T>;
    static TStorage mvm(const TStorage &lhsStorage, const TShape &lhsShape, const TStorage &rhsStorage,
                        const TShape &rhsShape) requires std::is_arithmetic_v<T>;
    static TStorage mm(const TStorage &lhsStorage, const TShape &lhsShape, const TStorage &rhsStorage,
                       const TShape &rhsShape) requires std::is_arithmetic_v<T>;
    // static TStorage omm(const TStorage &lhsStorage, const TShape &lhsShape, const TStorage &rhsStorage, const TShape
    // &rhsShape) requires std::is_arithmetic_v<T>;
    static TStorage bmm(const TStorage &lhsStorage, const TShape &lhsShape, const TStorage &rhsStorage,
                        const TShape &rhsShape) requires std::is_arithmetic_v<T>;
};