#include <iostream>
#include <format>

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
        throw std::invalid_argument(std::format("Coordinates {} are invalid for tensor of shape {}.", this->tensorShapeToStr(coord), this->tensorShapeToStr(this->shape_)));

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
        throw std::invalid_argument(std::format("Absolute {} is invalid for tensor of shape {} ({} elements).", std::to_string(abs), this->tensorShapeToStr(this->shape_), std::to_string(this->numel())));
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
Tensor<T>::Tensor(const std::vector<std::size_t> &shape, const std::vector<T> &buffer) : shape_(shape), buffer_(buffer)
{
    std::size_t shape_size = std::reduce(shape.begin(), shape.end(), 1, std::multiplies<int>());
    std::size_t buffer_size = buffer.size();
    if (shape_size != buffer_size)
        throw std::invalid_argument(std::format("Shape and Buffer number of elements are incompatible : {} and {}.", shape_size, buffer_size));
}

template <typename T>
Tensor<T>::~Tensor()
{
}

template <typename T>
std::vector<size_t> Tensor<T>::shape() const
{
    return this->shape_;
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
Tensor<T> Tensor<T>::flatten()
{
    this->shape_ = std::vector({this->numel()});
    return &this;
}

template <typename T>
Tensor<T> Tensor<T>::unsqueeze(std::size_t dim)
{
    this->shape_.insert(this->shape_.begin() + dim, 1);
    return *this;
}

template <typename T>
Tensor<T> Tensor<T>::squeeze(std::size_t dim)
{
    if (this->shape_[dim] != 1)
        throw std::invalid_argument(std::format("Cannot squeeze at dim {} : non-1 dimension.", dim));
    this->shape_.erase(this->shape_.begin() + dim);
    return *this;
}

template <typename T>
Tensor<T> Tensor<T>::t(std::size_t dim0, std::size_t dim1)
{
    return this->transpose(dim0, dim1);
}

template <typename T>
Tensor<T> Tensor<T>::transpose(std::size_t dim0, std::size_t dim1)
{
    if (this->shape_.size() < 2)
        throw std::invalid_argument("Cannot transpose tensor with less than 2 dimensions");

    std::vector<std::size_t> new_shape = this->shape_;
    std::swap(new_shape[dim0], new_shape[dim1]);
    Tensor<T> tensor = Tensor<T>(new_shape);
    std::cout << tensor << std::endl;
    for (std::size_t i = 0; i < this->numel(); i++)
    {
        std::vector<std::size_t> target_coord = this->absToCoord(i);
        std::swap(target_coord[dim0], target_coord[dim1]);
        tensor.buffer_[tensor.coordToAbs(target_coord)] = this->buffer_[i];
    }
    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::broadcast(Tensor<T> &tensor)
{
    return this->broadcast(tensor.shape_);
}

template <typename T>
Tensor<T> Tensor<T>::broadcast(const std::vector<std::size_t> &shape)
{
    // unsqueeze both dimensions until same number of dim
    std::vector<size_t> tensor_shape = this->shape_;
    std::vector<size_t> target_shape = shape;
    while (tensor_shape.size() < target_shape.size())
        tensor_shape.insert(tensor_shape.begin(), 1);
    while (target_shape.size() < tensor_shape.size())
        target_shape.insert(target_shape.begin(), 1);

    std::vector<std::size_t> new_shape = std::vector<std::size_t>();
    for (std::size_t i = 0; i < tensor_shape.size(); i++)
    {
        if (tensor_shape[i] != target_shape[i] && tensor_shape[i] != 1 && target_shape[i] != 1)
            throw std::invalid_argument(std::format("Tensor is not broadcastable to this shape : {} to {} at index {}.", this->tensorShapeToStr(tensor_shape), this->tensorShapeToStr(target_shape), i));
        new_shape.insert(new_shape.end(), tensor_shape[i] >= target_shape[i] ? tensor_shape[i] : target_shape[i]);
    }

    Tensor<T> new_tensor = Tensor<T>(new_shape);
    for (std::size_t i = 0; i < new_tensor.numel(); i++)
    {
        std::vector<std::size_t> coords = new_tensor.absToCoord(i);
        for (std::size_t j = 0; j < coords.size(); j++)
            coords[j] = tensor_shape[j] != 1 ? coords[j] : 0;
        new_tensor.buffer_[i] = this->buffer_[this->coordToAbs(coords)];
    }
    return new_tensor;
}