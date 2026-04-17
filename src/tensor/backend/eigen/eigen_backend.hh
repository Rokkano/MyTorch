#pragma once

#include "src/tensor/backend/backend.hh"

template <typename T>
struct EigenBackend
{
    using TStorage = std::vector<T>;

    static TStorage allocate(std::size_t n) { return TStorage(n); }

    static T *data_ptr(TStorage &s) { return s.data(); }

    static std::vector<T>::iterator begin(TStorage &s) { return s.begin(); }
    static std::vector<T>::iterator const_begin(TStorage &s) { return s.const_begin(); }
    static std::vector<T>::iterator end(TStorage &s) { return s.end(); }
    static std::vector<T>::iterator const_end(TStorage &s) { return s.const_end(); }

    static Tensor<T, EigenBackend<T>> affine(const Tensor<T, EigenBackend<T>> &tensor, std::optional<T> a,
                                             std::optional<T> b) requires std::is_arithmetic_v<T>;
    static Tensor<T, EigenBackend<T>> exp(const Tensor<T, EigenBackend<T>> &tensor) requires std::is_arithmetic_v<T>;
    static Tensor<T, EigenBackend<T>> log(const Tensor<T, EigenBackend<T>> &tensor) requires std::is_arithmetic_v<T>;
    static Tensor<T, EigenBackend<T>> pow(const Tensor<T, EigenBackend<T>> &tensor, double exponent)
        requires std::is_arithmetic_v<T>;
    static Tensor<T, EigenBackend<T>> sqrt(const Tensor<T, EigenBackend<T>> &tensor) requires std::is_arithmetic_v<T>;
    static Tensor<T, EigenBackend<T>> dot(const Tensor<T, EigenBackend<T>> &lhs, const Tensor<T, EigenBackend<T>> &rhs)
        requires std::is_arithmetic_v<T>;
    static Tensor<T, EigenBackend<T>> matmul(const Tensor<T, EigenBackend<T>> &lhs,
                                             const Tensor<T, EigenBackend<T>> &rhs) requires std::is_arithmetic_v<T>;
    static Tensor<T, EigenBackend<T>> mvm(const Tensor<T, EigenBackend<T>> &lhs, const Tensor<T, EigenBackend<T>> &rhs)
        requires std::is_arithmetic_v<T>;
    static Tensor<T, EigenBackend<T>> mm(const Tensor<T, EigenBackend<T>> &lhs, const Tensor<T, EigenBackend<T>> &rhs)
        requires std::is_arithmetic_v<T>;
    static Tensor<T, EigenBackend<T>> omm(const Tensor<T, EigenBackend<T>> &lhs, const Tensor<T, EigenBackend<T>> &rhs)
        requires std::is_arithmetic_v<T>;
    static Tensor<T, EigenBackend<T>> bmm(const Tensor<T, EigenBackend<T>> &lhs, const Tensor<T, EigenBackend<T>> &rhs)
        requires std::is_arithmetic_v<T>;
    static Tensor<T, EigenBackend<T>> identity(std::size_t n) requires std::is_arithmetic_v<T>;
};

#include "eigen_linear.hxx"