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
    T item() const;
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
    Tensor<bool> all() const;
    Tensor<bool> any() const;
    Tensor<bool> none() const;

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

    // ###### TENSOR MATH ######
    Tensor<T> min()
        requires std::is_arithmetic_v<T>;
    Tensor<T> min(const T &)
        requires std::is_arithmetic_v<T>;
    Tensor<T> min(const Tensor<T> &)
        requires std::is_arithmetic_v<T>;
    Tensor<T> amin(const std::size_t)
        requires std::is_arithmetic_v<T>;
    Tensor<T> max()
        requires std::is_arithmetic_v<T>;
    Tensor<T> max(const T &)
        requires std::is_arithmetic_v<T>;
    Tensor<T> max(const Tensor<T> &)
        requires std::is_arithmetic_v<T>;
    Tensor<T> amax(const std::size_t)
        requires std::is_arithmetic_v<T>;
    Tensor<T> mean(int bessel_correction = 0)
        requires std::is_arithmetic_v<T>;
    Tensor<T> amean(const std::size_t, int bessel_correction = 0)
        requires std::is_arithmetic_v<T>;
    Tensor<T> var(int bessel_correction = 0)
        requires std::is_arithmetic_v<T>;
    Tensor<T> std(int bessel_correction = 0)
        requires std::is_arithmetic_v<T>;
    Tensor<std::size_t> argmin()
        requires std::is_arithmetic_v<T>;
    Tensor<std::size_t> argmin(const std::size_t)
        requires std::is_arithmetic_v<T>;
    Tensor<std::size_t> argmax()
        requires std::is_arithmetic_v<T>;
    Tensor<std::size_t> argmax(const std::size_t)
        requires std::is_arithmetic_v<T>;
    Tensor<T> sum()
        requires std::is_arithmetic_v<T>;
    Tensor<T> heaviside()
        requires std::is_arithmetic_v<T>;

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

    static Tensor<T> relu(const Tensor<T> &)
        requires std::is_arithmetic_v<T>;
    static Tensor<T> sigmoid(const Tensor<T> &)
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

#include "tensor_bool.hxx"
#include "tensor_io.hxx"
#include "tensor_math.hxx"
#include "tensor_op.hxx"
#include "tensor_utils.hxx"
#include "tensor.hxx"

// functionals
#include "functional/tensor_activation.hxx"
#include "functional/tensor_convolution.hxx"
#include "functional/tensor_linear.hxx"
#include "functional/tensor_loss.hxx"
#include "functional/tensor_pooling.hxx"
#include "functional/tensor_vision.hxx"