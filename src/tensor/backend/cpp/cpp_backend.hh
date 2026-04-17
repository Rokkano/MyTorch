#pragma once

#include "src/tensor/backend/backend.hh"

template <typename T>
struct CppBackend
{
    using TStorage = std::vector<T>;

    static TStorage allocate(std::size_t n) { return TStorage(n); }

    static std::size_t size(TStorage &s) { return s.size(); }

    static T *data_ptr(TStorage &s) { return s.data(); }

    static T &get(TStorage &s, std::size_t i) { return s[i]; }
    static const T &get(const TStorage &s, std::size_t i) { return s[i]; }

    static std::vector<T>::iterator begin(TStorage &s) { return s.begin(); }
    static std::vector<T>::iterator const_begin(TStorage &s) { return s.const_begin(); }
    static std::vector<T>::iterator end(TStorage &s) { return s.end(); }
    static std::vector<T>::iterator const_end(TStorage &s) { return s.const_end(); }

    static Tensor<T, CppBackend<T>> affine(const Tensor<T, CppBackend<T>> &tensor, std::optional<T> a,
                                           std::optional<T> b) requires std::is_arithmetic_v<T>;
    static Tensor<T, CppBackend<T>> exp(const Tensor<T, CppBackend<T>> &tensor) requires std::is_arithmetic_v<T>;
    static Tensor<T, CppBackend<T>> log(const Tensor<T, CppBackend<T>> &tensor) requires std::is_arithmetic_v<T>;
    static Tensor<T, CppBackend<T>> pow(const Tensor<T, CppBackend<T>> &tensor, double exponent)
        requires std::is_arithmetic_v<T>;
    static Tensor<T, CppBackend<T>> sqrt(const Tensor<T, CppBackend<T>> &tensor) requires std::is_arithmetic_v<T>;
    static Tensor<T, CppBackend<T>> dot(const Tensor<T, CppBackend<T>> &lhs, const Tensor<T, CppBackend<T>> &rhs)
        requires std::is_arithmetic_v<T>;
    static Tensor<T, CppBackend<T>> matmul(const Tensor<T, CppBackend<T>> &lhs, const Tensor<T, CppBackend<T>> &rhs)
        requires std::is_arithmetic_v<T>;
    static Tensor<T, CppBackend<T>> mvm(const Tensor<T, CppBackend<T>> &lhs, const Tensor<T, CppBackend<T>> &rhs)
        requires std::is_arithmetic_v<T>;
    static Tensor<T, CppBackend<T>> mm(const Tensor<T, CppBackend<T>> &lhs, const Tensor<T, CppBackend<T>> &rhs)
        requires std::is_arithmetic_v<T>;
    static Tensor<T, CppBackend<T>> omm(const Tensor<T, CppBackend<T>> &lhs, const Tensor<T, CppBackend<T>> &rhs)
        requires std::is_arithmetic_v<T>;
    static Tensor<T, CppBackend<T>> bmm(const Tensor<T, CppBackend<T>> &lhs, const Tensor<T, CppBackend<T>> &rhs)
        requires std::is_arithmetic_v<T>;
};

#include "cpp_linear.hxx"