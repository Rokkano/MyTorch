#pragma once

#include "../mt/mt.hh"
#include "../exception/exception.hh"

#include <functional>
#include <iostream>
#include <optional>
#include <vector>

template <typename T>
class Tensor: public IMTSerialize
{
private:
    std::vector<std::size_t> shape_;
    std::vector<T> buffer_;

    template <typename>
    friend class Tensor;

    bool validateCoord(const std::vector<std::size_t> &) const;
    bool validateAbs(std::size_t) const;

public:
    Tensor();
    Tensor(const std::vector<std::size_t> &);
    Tensor(const std::vector<std::size_t> &, const std::vector<T> &);
    ~Tensor();

    std::vector<T>::iterator begin();
    std::vector<T>::iterator const_begin() const;
    std::vector<T>::iterator end();
    std::vector<T>::iterator const_end() const;

    T &operator[](std::size_t);

    std::vector<std::size_t> shape() const;
    std::size_t numel() const;
    T item() const;
    Tensor<T> flatten() const;
    Tensor<T> unsqueeze(std::size_t dim = 0) const;
    Tensor<T> squeeze(std::size_t dim = 0) const;
    Tensor<T> t(std::size_t dim0 = 0, std::size_t dim1 = 1) const;
    Tensor<T> transpose(std::size_t dim0 = 0, std::size_t dim1 = 1) const;
    void fill(const T &);
    Tensor<T> broadcast(const std::vector<std::size_t> &) const;
    Tensor<T> broadcast(Tensor<T> &) const;
    Tensor<T> batch_broadcast(const std::vector<std::size_t> &) const;
    Tensor<T> batch_broadcast(Tensor<T> &) const;

    std::size_t coordToAbs(const std::vector<std::size_t> &) const;
    std::vector<std::size_t> absToCoord(std::size_t) const;

    // ###### TENSOR OP ######
public:
    template <typename U>
    friend Tensor<U> operator+(const Tensor<U> &, const Tensor<U> &);
    template <typename U>
    friend Tensor<U> operator+(const Tensor<U> &, const U &);
    template <typename U>
    friend Tensor<U> operator+(const U &, const Tensor<U> &);
    template <typename U>
    friend Tensor<U> operator+(const Tensor<U> &);
    template <typename U>
    friend Tensor<U> operator-(const Tensor<U> &, const Tensor<U> &);
    template <typename U>
    friend Tensor<U> operator-(const Tensor<U> &, const U &);
    template <typename U>
    friend Tensor<U> operator-(const U &, const Tensor<U> &);
    template <typename U>
    friend Tensor<U> operator-(const Tensor<U> &);
    template <typename U>
    friend Tensor<U> operator*(const Tensor<U> &, const Tensor<U> &);
    template <typename U>
    friend Tensor<U> operator*(const Tensor<U> &, const U &);
    template <typename U>
    friend Tensor<U> operator*(const U &, const Tensor<U> &);
    template <typename U>
    friend Tensor<U> operator/(const Tensor<U> &, const Tensor<U> &);
    template <typename U>
    friend Tensor<U> operator/(const Tensor<U> &, const U &);
    template <typename U>
    friend Tensor<U> operator/(const U &, const Tensor<U> &);

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
    static Tensor<T> log(const Tensor<T> &)
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
    static Tensor<T> drelu(const Tensor<T> &)
        requires std::is_arithmetic_v<T>;
    static Tensor<T> sigmoid(const Tensor<T> &)
        requires std::is_arithmetic_v<T>;
    static Tensor<T> dsigmoid(const Tensor<T> &)
        requires std::is_arithmetic_v<T>;
    static Tensor<T> sinh(const Tensor<T> &)
        requires std::is_arithmetic_v<T>;
    static Tensor<T> cosh(const Tensor<T> &)
        requires std::is_arithmetic_v<T>;
    static Tensor<T> tanh(const Tensor<T> &)
        requires std::is_arithmetic_v<T>;
    static Tensor<T> dtanh(const Tensor<T> &)
        requires std::is_arithmetic_v<T>;

    // ###### TENSOR IO ######
public:
    static std::string tensorDataToStr(const std::vector<std::size_t> &, const std::vector<T> &);
    static std::string tensorShapeToStr(const std::vector<std::size_t> &);
    static std::string tensorToStr(const std::vector<std::size_t> &, const std::vector<T> &);

    void plot(std::string title = "", std::string xlabel = "", std::string ylabel = "") const
        requires std::is_arithmetic_v<T>;

public:
    template <typename U>
    friend std::ostream &operator<<(std::ostream &, const Tensor<U> &);

    // ###### TENSOR MT SERIALIZE ######
    virtual std::vector<std::byte> serialize();
    virtual void deserialize(std::vector<std::byte>);
    static Tensor<T> from_bytes(std::vector<std::byte> &);

    // ###### TENSOR UTILS ######
public:
    template <typename U>
    Tensor<U> to_type();
    static Tensor<T> from_function(std::function<T(std::size_t)>, const std::vector<std::size_t> &);
    static Tensor<T> from_vector(const std::vector<T> &, const std::vector<std::size_t> &);
    static Tensor<T> one_hot(std::size_t, const std::vector<std::size_t> &);
};

template <typename T>
struct is_tuple : std::false_type
{
};

template <typename... Args>
struct is_tuple<std::tuple<Args...>> : std::true_type
{
};

template <typename T>
concept Tuple = is_tuple<T>::value;

#include "tensor.hxx"
#include "tensor_bool.hxx"
#include "tensor_io.hxx"
#include "tensor_math.hxx"
#include "tensor_op.hxx"
#include "tensor_utils.hxx"
#include "tensor_serialize.hxx"

// functionals
#include "functional/tensor_activation.hxx"
#include "functional/tensor_convolution.hxx"
#include "functional/tensor_linear.hxx"
#include "functional/tensor_loss.hxx"
#include "functional/tensor_pooling.hxx"
#include "functional/tensor_vision.hxx"