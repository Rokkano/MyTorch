#pragma once

#include <iostream>
#include <vector>
#include <optional>
#include <functional>

#include "../exception/exception.hh"

template <typename T>
class Tensor
{
private:
    std::vector<std::size_t> shape_;
    std::vector<T> buffer_;

    template <typename>
    friend class Tensor;

    bool validateCoord(const std::vector<std::size_t> &) const;
    bool validateAbs(std::size_t) const;

public:
    Tensor(const std::vector<std::size_t> &);
    Tensor(const std::vector<std::size_t> &, const std::vector<T> &);
    ~Tensor();

    std::vector<std::size_t> shape() const;
    std::size_t numel() const;
    Tensor<T> flatten();
    Tensor<T> unsqueeze(std::size_t dim = 0);
    Tensor<T> squeeze(std::size_t dim = 0);
    Tensor<T> t(std::size_t dim0 = 0, std::size_t dim1 = 1) const;
    Tensor<T> transpose(std::size_t dim0 = 0, std::size_t dim1 = 1) const;
    void fill(const T &);
    Tensor<T> broadcast(const std::vector<std::size_t> &);
    Tensor<T> broadcast(Tensor<T> &);
    Tensor<T> batch_broadcast(const std::vector<std::size_t> &);
    Tensor<T> batch_broadcast(Tensor<T> &);

    std::size_t coordToAbs(const std::vector<std::size_t> &) const;
    std::vector<std::size_t> absToCoord(std::size_t) const;

    // ###### TENSOR OP ######
public:
    Tensor<T> operator+(const Tensor<T> &);
    Tensor<T> operator+(const T &);
    Tensor<T> operator+();
    Tensor<T> operator-(const Tensor<T> &);
    Tensor<T> operator-(const T &);
    Tensor<T> operator-();
    Tensor<T> operator*(const Tensor<T> &);
    Tensor<T> operator*(const T &);

private:
    bool validateSameShape(const std::vector<std::size_t> &) const;

    // ###### TENSOR BOOL ######
public:
    explicit operator bool() const;
    bool all() const;
    bool any() const;
    bool none() const;

    Tensor<bool> operator==(const Tensor<T> &);
    Tensor<bool> operator==(const T &);
    Tensor<bool> operator<(const Tensor<T> &);
    Tensor<bool> operator<(const T &);
    Tensor<bool> operator<=(const Tensor<T> &);
    Tensor<bool> operator<=(const T &);
    Tensor<bool> operator>(const Tensor<T> &);
    Tensor<bool> operator>(const T &);
    Tensor<bool> operator>=(const Tensor<T> &);
    Tensor<bool> operator>=(const T &);

    // ###### TENSOR OP FUNCTIONAL (arithmetic only) ######
public:
    static Tensor<T> affine(const Tensor<T> &, std::optional<T>, std::optional<T>)
        requires std::is_arithmetic_v<T>;
    static Tensor<T> exp(const Tensor<T> &)
        requires std::is_arithmetic_v<T>;
    static Tensor<T> pow(const Tensor<T> &t, const double)
        requires std::is_arithmetic_v<T>;
    static Tensor<T> sqrt(const Tensor<T> &)
        requires std::is_arithmetic_v<T>;
    static Tensor<T> dot(const Tensor<T> &, const Tensor<T> &)
        requires std::is_arithmetic_v<T>;
    static Tensor<T> matmul(const Tensor<T> &, const Tensor<T> &)
        requires std::is_arithmetic_v<T>;
    static Tensor<T> mvm(const Tensor<T> &, const Tensor<T> &)
        requires std::is_arithmetic_v<T>;
    static Tensor<T> mm(const Tensor<T> &, const Tensor<T> &)
        requires std::is_arithmetic_v<T>;
    static Tensor<T> omm(const Tensor<T> &, const Tensor<T> &)
        requires std::is_arithmetic_v<T>;
    static Tensor<T> bmm(const Tensor<T> &, const Tensor<T> &)
        requires std::is_arithmetic_v<T>;

    // ###### TENSOR IO ######
private:
    static std::string tensorDataToStr(const std::vector<std::size_t> &, const std::vector<T> &);
    static std::string tensorShapeToStr(const std::vector<std::size_t> &);
    static std::string tensorToStr(const std::vector<std::size_t> &, const std::vector<T> &);

public:
    template <typename U>
    friend std::ostream &operator<<(std::ostream &, const Tensor<U> &);

    // ###### TENSOR UTILS ######
public:
    template <typename U>
    Tensor<U> to_type();
    static Tensor<T> from_function(std::function<std::size_t(T)>, const std::vector<std::size_t> &);
    static Tensor<T> from_vector(const std::vector<T> &, const std::vector<std::size_t> &);
};

#include "tensor.hxx"
#include "tensor_op.hxx"
#include "tensor_functional.hxx"
#include "tensor_bool.hxx"
#include "tensor_io.hxx"
#include "tensor_utils.hxx"