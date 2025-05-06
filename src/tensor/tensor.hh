#pragma once

#include <iostream>
#include <vector>
#include <optional>

template <typename T>
class Tensor
{
private:
    std::vector<std::size_t> shape_;
    T *buffer_;

    bool validateCoord(std::vector<std::size_t>) const;
    bool validateAbs(std::size_t abs) const;

public:
    Tensor(std::vector<std::size_t>);
    Tensor(std::vector<std::size_t>, T *);
    ~Tensor();

    std::size_t numel() const;
    Tensor<T> *flatten();
    void fill(T value);

    std::size_t coordToAbs(std::vector<std::size_t> coord) const;
    std::vector<std::size_t> absToCoord(std::size_t abs) const;

    Tensor<T> *operator[](size_t index)
    {
        std::vector<std::size_t> new_shape = std::vector<std::size_t>(this->shape_.begin() + 1, this->shape_.end());
        std::size_t step = 1;
        for (std::size_t j = 1; j < this->shape_.size(); j++)
            step *= this->shape_[j];
        T *new_buffer = &this->buffer_[index * step];
        return new Tensor<T>(new_shape, new_buffer);
    }

    // ###### TENSOR OP ######
public:
    template <typename U>
    friend Tensor<U> *operator+(const Tensor<U> &, const Tensor<U> &);
    template <typename U>
    friend Tensor<U> *operator+(const Tensor<U> &, const U &);
    template <typename U>
    friend Tensor<U> *operator+(const U &, const Tensor<U> &);
    template <typename U>
    friend Tensor<U> *operator+(const Tensor<U> &);
    template <typename U>
    friend Tensor<U> *operator-(const Tensor<U> &, const Tensor<U> &);
    template <typename U>
    friend Tensor<U> *operator-(const Tensor<U> &, const U &);
    template <typename U>
    friend Tensor<U> *operator-(const U &, const Tensor<U> &);
    template <typename U>
    friend Tensor<U> *operator-(const Tensor<U> &);
    template <typename U>
    friend Tensor<U> *operator*(const Tensor<U> &, const Tensor<U> &);
    template <typename U>
    friend Tensor<U> *operator*(const Tensor<U> &, const U &);
    template <typename U>
    friend Tensor<U> *operator*(const U &, const Tensor<U> &);

private:
    bool validateSameShape(std::vector<std::size_t>) const;

    // ###### TENSOR BOOL ######
public:
    explicit operator bool() const;
    bool all() const;
    bool any() const;
    bool none() const;

    template <typename U>
    friend Tensor<bool> *operator==(const Tensor<U> &, const Tensor<U> &);
    template <typename U>
    friend Tensor<bool> *operator<(const Tensor<U> &, const Tensor<U> &);
    template <typename U>
    friend Tensor<bool> *operator<=(const Tensor<U> &, const Tensor<U> &);
    template <typename U>
    friend Tensor<bool> *operator>(const Tensor<U> &, const Tensor<U> &);
    template <typename U>
    friend Tensor<bool> *operator>=(const Tensor<U> &, const Tensor<U> &);

    // ###### TENSOR OP FUNCTIONAL ######
public:
    static Tensor<T> *affine(const Tensor<T> &rhs, std::optional<T> a, std::optional<T> b);

    // ###### TENSOR IO ######
private:
    static std::string tensorDataToStr(std::vector<std::size_t>, T *);
    static std::string tensorShapeToStr(std::vector<std::size_t>);
    static std::string tensorToStr(std::vector<std::size_t>, T *);

public:
    template <typename U>
    friend std::ostream &operator<<(std::ostream &, const Tensor<U> &);
};

#include "tensor.hxx"
#include "tensor_op.hxx"
#include "tensor_functional.hxx"
#include "tensor_bool.hxx"
#include "tensor_io.hxx"