#pragma once

#include "tensor_fwd.hh"

#include <functional>
#include <iostream>
#include <optional>
#include <vector>

struct OpenCVWindowOpts;

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
template <typename T, template <typename> typename B >
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

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
class Tensor : public IMTSerialize
{
    using TStorage = typename B<T>::TStorage;

protected:
    std::vector<std::size_t> shape_ = std::vector<std::size_t>();
    std::size_t numel_ = 0;
    std::vector<std::size_t> stride_ = std::vector<std::size_t>();

    TStorage data_ = TStorage();

    // Constructors
public:
    Tensor();
    Tensor(const std::vector<std::size_t> &shape);
    Tensor(const std::vector<std::size_t> &shape, const TStorage &data);
    ~Tensor();

    // Accessors
public:
    std::vector<T>::iterator begin();
    std::vector<T>::iterator const_begin() const;
    std::vector<T>::iterator end();
    std::vector<T>::iterator const_end() const;

    T &operator[](std::size_t);
    const T &operator[](std::size_t) const;

    TStorage &data();
    std::vector<std::size_t> &shape();

    std::size_t numel() const;
    std::vector<std::size_t> stride() const;

    T item() const;
    
    // Basic operations
    Tensor<T, B> &fill(const T value);
    Tensor<T, B> &flatten();
    Tensor<T, B> &unsqueeze(std::size_t dim = 0);
    Tensor<T, B> &squeeze(std::size_t dim = 0);
    Tensor<T, B> &broadcast(const std::vector<std::size_t> &shape);
    Tensor<T, B> &broadcast(Tensor<T, B> &tensor);
    Tensor<T, B> &batch_broadcast(const std::vector<std::size_t> &shape);
    Tensor<T, B> &batch_broadcast(Tensor<T, B> &tensor);

    Tensor<T, B> t(std::size_t dim0 = 0, std::size_t dim1 = 1);
    Tensor<T, B> transpose(std::size_t dim0 = 0, std::size_t dim1 = 1);

private:
    std::size_t coordToAbs(const std::vector<std::size_t> &coord) const;
    std::vector<std::size_t> absToCoord(std::size_t abs) const;

    // ###### TENSOR OP ######
    // See : https://stackoverflow.com/questions/4421706/what-are-the-basic-rules-and-idioms-for-operator-overloading/4421729#4421729
public:
    friend Tensor<T, B> operator+<T, B>(const Tensor<T, B> &lhs, const Tensor<T, B> &rhs);
    friend Tensor<T, B> operator+<T, B>(const Tensor<T, B> &lhs, const T &rhs);
    friend Tensor<T, B> operator+<T, B>(const T &lhs, const Tensor<T, B> &rhs);
    friend Tensor<T, B> operator+<T, B>(const Tensor<T, B> &lhs);
    friend Tensor<T, B> operator-<T, B>(const Tensor<T, B> &lhs, const Tensor<T, B> &rhs);
    friend Tensor<T, B> operator-<T, B>(const Tensor<T, B> &lhs, const T &rhs);
    friend Tensor<T, B> operator-<T, B>(const T &lhs, const Tensor<T, B> &rhs);
    friend Tensor<T, B> operator-<T, B>(const Tensor<T, B> &lhs);
    friend Tensor<T, B> operator*<T, B>(const Tensor<T, B> &lhs, const Tensor<T, B> &rhs);
    friend Tensor<T, B> operator*<T, B>(const Tensor<T, B> &lhs, const T &rhs);
    friend Tensor<T, B> operator*<T, B>(const T &lhs, const Tensor<T, B> &rhs);
    friend Tensor<T, B> operator/<T, B>(const Tensor<T, B> &lhs, const Tensor<T, B> &rhs);
    friend Tensor<T, B> operator/<T, B>(const Tensor<T, B> &lhs, const T &rhs);
    friend Tensor<T, B> operator/<T, B>(const T &lhs, const Tensor<T, B> &rhs);

private:
    bool validateSameShape(const std::vector<std::size_t> &shape) const;

    // ###### TENSOR COMPARISON/BOOL ######
public:
    explicit operator bool() const;
    Tensor<bool, B> all() const;
    Tensor<bool, B> any() const;
    Tensor<bool, B> none() const;

    Tensor<bool, B> operator==(const Tensor<T, B> &other);
    Tensor<bool, B> operator==(const T &other);
    Tensor<bool, B> operator<(const Tensor<T, B> &other);
    Tensor<bool, B> operator<(const T &other);
    Tensor<bool, B> operator<=(const Tensor<T, B> &other);
    Tensor<bool, B> operator<=(const T &other);
    Tensor<bool, B> operator>(const Tensor<T, B> &other);
    Tensor<bool, B> operator>(const T &other);
    Tensor<bool, B> operator>=(const Tensor<T, B> &other);
    Tensor<bool, B> operator>=(const T &other);

    // ###### TENSOR MATH ######
    Tensor<T, B> min() const requires std::is_arithmetic_v<T>;
    Tensor<T, B> min(const T &) const requires std::is_arithmetic_v<T>;
    Tensor<T, B> min(const Tensor<T, B> &) const requires std::is_arithmetic_v<T>;
    Tensor<T, B> amin(const std::size_t) const requires std::is_arithmetic_v<T>;
    Tensor<T, B> max() const requires std::is_arithmetic_v<T>;
    Tensor<T, B> max(const T &) const requires std::is_arithmetic_v<T>;
    Tensor<T, B> max(const Tensor<T, B> &) const requires std::is_arithmetic_v<T>;
    Tensor<T, B> amax(const std::size_t) const requires std::is_arithmetic_v<T>;
    Tensor<T, B> mean(int bessel_correction = 0) const requires std::is_arithmetic_v<T>;
    Tensor<T, B> amean(const std::size_t, int bessel_correction = 0) const requires std::is_arithmetic_v<T>;
    Tensor<T, B> var(int bessel_correction = 0) const requires std::is_arithmetic_v<T>;
    Tensor<T, B> std(int bessel_correction = 0) const requires std::is_arithmetic_v<T>;
    Tensor<std::size_t, B> argmin() const requires std::is_arithmetic_v<T>;
    Tensor<std::size_t, B> argmin(const std::size_t) const requires std::is_arithmetic_v<T>;
    Tensor<std::size_t, B> argmax() const requires std::is_arithmetic_v<T>;
    Tensor<std::size_t, B> argmax(const std::size_t) const requires std::is_arithmetic_v<T>;
    Tensor<T, B> sum() const requires std::is_arithmetic_v<T>;
    Tensor<T, B> heaviside() const requires std::is_arithmetic_v<T>;
    Tensor<T, B> round() const requires std::is_arithmetic_v<T>;
    Tensor<T, B> floor() const requires std::is_arithmetic_v<T>;
    Tensor<T, B> ceil() const requires std::is_arithmetic_v<T>;

    // ###### TENSOR OP FUNCTIONAL (arithmetic only) ######
public:
    static Tensor<T, B> affine(const Tensor<T, B> &tensor, std::optional<T>, std::optional<T>)
        requires std::is_arithmetic_v<T>;
    static Tensor<T, B> exp(const Tensor<T, B> &tensor) requires std::is_arithmetic_v<T>;
    static Tensor<T, B> log(const Tensor<T, B> &tensor) requires std::is_arithmetic_v<T>;
    static Tensor<T, B> pow(const Tensor<T, B> &tensor, double exponent) requires std::is_arithmetic_v<T>;
    static Tensor<T, B> sqrt(const Tensor<T, B> &tensor) requires std::is_arithmetic_v<T>;
    static Tensor<T, B> dot(const Tensor<T, B> &lhs, const Tensor<T, B> &rhs) requires std::is_arithmetic_v<T>;
    static Tensor<T, B> matmul(const Tensor<T, B> &lhs, const Tensor<T, B> &rhs) requires std::is_arithmetic_v<T>;
    static Tensor<T, B> mvm(const Tensor<T, B> &lhs, const Tensor<T, B> &rhs) requires std::is_arithmetic_v<T>;
    static Tensor<T, B> mm(const Tensor<T, B> &lhs, const Tensor<T, B> &rhs) requires std::is_arithmetic_v<T>;
    static Tensor<T, B> omm(const Tensor<T, B> &lhs, const Tensor<T, B> &rhs) requires std::is_arithmetic_v<T>;
    static Tensor<T, B> bmm(const Tensor<T, B> &lhs, const Tensor<T, B> &rhs) requires std::is_arithmetic_v<T>;
    static Tensor<T, B> identity(std::size_t n) requires std::is_arithmetic_v<T>;
    static Tensor<T, B> relu(const Tensor<T, B> &tensor) requires std::is_arithmetic_v<T>;
    static Tensor<T, B> drelu(const Tensor<T, B> &tensor) requires std::is_arithmetic_v<T>;
    static Tensor<T, B> sigmoid(const Tensor<T, B> &tensor) requires std::is_arithmetic_v<T>;
    static Tensor<T, B> dsigmoid(const Tensor<T, B> &tensor) requires std::is_arithmetic_v<T>;
    static Tensor<T, B> sinh(const Tensor<T, B> &tensor) requires std::is_arithmetic_v<T>;
    static Tensor<T, B> cosh(const Tensor<T, B> &tensor) requires std::is_arithmetic_v<T>;
    static Tensor<T, B> tanh(const Tensor<T, B> &tensor) requires std::is_arithmetic_v<T>;
    static Tensor<T, B> dtanh(const Tensor<T, B> &tensor) requires std::is_arithmetic_v<T>;
    static Tensor<T, B> softmax(const Tensor<T, B> &tensor) requires std::is_arithmetic_v<T>;

    // ###### TENSOR IO ######

public:
    std::string toStr() const;
    void plot(const std::string &linespec, OpenCVWindowOpts opts) const requires std::is_arithmetic_v<T>;

    template <typename T2, template <typename> typename B2>
    friend std::ostream &operator<<(std::ostream &, const Tensor<T2, B2> &);

    static std::string shapeToStr(const std::vector<std::size_t> &shape);
    static std::string dataToStr(const TStorage &storage, const std::vector<std::size_t> &shape, std::size_t truncate = 50);
    static std::string dataToStr(const TStorage &storage, std::size_t truncate = 50);


    // ###### TENSOR MT SERIALIZE ######
public:
    virtual std::vector<std::byte> serialize();
    virtual std::size_t deserialize(std::vector<std::byte> &bytes);
    static Tensor<T, B> from_bytes(std::vector<std::byte> &bytes);

    // ###### TENSOR UTILS ######
public:
    template <typename T2>
    Tensor<T2, B> to_type();

    static Tensor<T, B> from_function(std::function<T(std::size_t)> lambda, const std::vector<std::size_t> &shape);
    static Tensor<T, B> from_vector(const std::vector<T> &buffer, const std::vector<std::size_t> &shape);
    static Tensor<T, B> one_hot(std::size_t index, const std::vector<std::size_t> &shape);

private:
    bool validateCoord(const std::vector<std::size_t> &coord) const;
    bool validateAbs(std::size_t abs) const;
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