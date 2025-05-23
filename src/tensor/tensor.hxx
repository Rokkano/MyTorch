#include <iostream>

#include "tensor.hh"
#include "../utils.hxx"

template <typename T>
bool Tensor<T>::validateCoord(const std::vector<std::size_t> &coord) const
{
    if (coord.size() < 1 || coord.size() != this->shape_.size())
        return false;

    for (std::size_t i = 0; i < coord.size(); i++)
        if (coord[i] < 0 || coord[i] >= this->shape_[i])
            return false;

    return true;
}

template <typename T>
bool Tensor<T>::validateAbs(std::size_t abs) const
{
    return abs < this->numel();
}

template <typename T>
std::size_t Tensor<T>::coordToAbs(const std::vector<std::size_t> &coord) const
{
    // [x] with [x_m] -> x
    // [y,x] with [y_m,x_m] -> y * x_m + x
    // [z,y,x] with [z_m,y_m,x_m] -> z * y_m * x_m + y * x_m + x
    if (!this->validateCoord(coord))
    {
        throw std::invalid_argument("Coordinates " + this->tensorShapeToStr(coord) + " are invalid for tensor of shape" + this->tensorShapeToStr(this->shape_) + ".");
    }
    std::size_t abs = 0;
    for (std::size_t i = 0; i < coord.size(); i++)
    {
        std::size_t step = 1;
        for (std::size_t j = i + 1; j < coord.size(); j++)
            step *= this->shape_[j];
        abs += coord[i] * step;
    }
    return abs;
}

template <typename T>
std::vector<std::size_t> Tensor<T>::absToCoord(std::size_t abs) const
{
    // [x] with [x_m] -> x
    // [y,x] with [y_m,x_m] -> y * x_m + x
    // [z,y,x] with [z_m,y_m,x_m] -> z * y_m * x_m + y * x_m + x
    if (!this->validateAbs(abs))
        throw std::invalid_argument("Absolute " + std::to_string(abs) + " is invalid for tensor of shape" + this->tensorShapeToStr(this->shape_) + " (" + std::to_string(this->numel()) + "elements).");
    std::vector<std::size_t> coord = std::vector<std::size_t>();
    for (std::size_t i = 0; i < this->shape_.size(); i++)
    {
        std::size_t step = 1;
        for (std::size_t j = i + 1; j < this->shape_.size(); j++)
            step *= this->shape_[j];

        std::size_t div = (std::size_t)abs / step;
        std::size_t rem = abs % step;

        coord.push_back(div);
        abs = rem;
    }
    return coord;
}

template <typename T>
Tensor<T>::Tensor(const std::vector<std::size_t> &shape) : shape_(shape)
{
    std::size_t num_e = this->numel();
    this->buffer_ = std::vector<T>(num_e);
}

template <typename T>
Tensor<T>::Tensor(const std::vector<std::size_t> &shape, const std::vector<T> &buffer) : shape_(shape), buffer_(buffer) {}

template <typename T>
Tensor<T>::~Tensor()
{
}

template <typename T>
void Tensor<T>::fill(const T &value)
{
    for (std::size_t i = 0; i < this->numel(); i++)
        this->buffer_[i] = value;
}

template <typename T>
std::size_t Tensor<T>::numel() const
{
    if (this->shape_.empty())
    {
        return 0;
    }
    std::size_t num_e = 1;
    for (const std::size_t &e : this->shape_)
        num_e *= e;
    return num_e;
}

template <typename T>
Tensor<T> *Tensor<T>::flatten()
{
    this->shape_ = std::vector({this->numel()});
    return this;
}
