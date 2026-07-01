#pragma once

#include "src/mt/imt.hh"
#include "src/tensor/backend/concept.hh"

#include <concepts>

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
class Tensor;

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> operator+(const Tensor<T, B> &lhs, const Tensor<T, B> &rhs);
template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> operator+(const Tensor<T, B> &lhs, const T &rhs);
template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> operator+(const T &lhs, const Tensor<T, B> &rhs);
template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> operator+(const Tensor<T, B> &lhs);
template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> operator-(const Tensor<T, B> &lhs, const Tensor<T, B> &rhs);
template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> operator-(const Tensor<T, B> &lhs, const T &rhs);
template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> operator-(const T &lhs, const Tensor<T, B> &rhs);
template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> operator-(const Tensor<T, B> &lhs);
template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> operator*(const Tensor<T, B> &lhs, const Tensor<T, B> &rhs);
template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> operator*(const Tensor<T, B> &lhs, const T &rhs);
template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> operator*(const T &lhs, const Tensor<T, B> &rhs);
template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> operator/(const Tensor<T, B> &lhs, const Tensor<T, B> &rhs);
template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> operator/(const Tensor<T, B> &lhs, const T &rhs);
template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> operator/(const T &lhs, const Tensor<T, B> &rhs);