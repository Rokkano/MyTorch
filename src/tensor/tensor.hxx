#include "src/exception/exception.hh"
#include "tensor.hh"

#include <format>
#include <iostream>
#include <numeric>

template <typename T, typename B>
requires IsBackend<T, B>
bool Tensor<T, B>::validateCoord(const std::vector<std::size_t> &coord) const
{
    if (coord.size() < 1 || coord.size() != this->shape_.size())
        return false;

    for (std::size_t i = 0; i < coord.size(); i++)
        if (coord[i] >= this->shape_[i])
            return false;

    return true;
}

template <typename T, typename B>
requires IsBackend<T, B>
bool Tensor<T, B>::validateAbs(std::size_t abs) const
{
    return abs < this->numel();
}

template <typename T, typename B>
requires IsBackend<T, B>
std::size_t Tensor<T, B>::coordToAbs(const std::vector<std::size_t> &coord) const
{
    // [x] with [x_m] -> x
    // [y,x] with [y_m,x_m] -> y * x_m + x
    // [z,y,x] with [z_m,y_m,x_m] -> z * y_m * x_m + y * x_m + x
    if (!this->validateCoord(coord))
        throw std::invalid_argument(std::format("Coordinates {} are invalid for tensor of shape {}.",
                                                this->tensorShapeToStr(coord), this->tensorShapeToStr(this->shape_)));

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

template <typename T, typename B>
requires IsBackend<T, B>
std::vector<std::size_t> Tensor<T, B>::absToCoord(std::size_t abs) const
{
    // [x] with [x_m] -> x
    // [y,x] with [y_m,x_m] -> y * x_m + x
    // [z,y,x] with [z_m,y_m,x_m] -> z * y_m * x_m + y * x_m + x
    if (!this->validateAbs(abs))
        throw std::invalid_argument(std::format("Absolute {} is invalid for tensor of shape {} ({} elements).",
                                                std::to_string(abs), this->tensorShapeToStr(this->shape_),
                                                std::to_string(this->numel())));
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

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B>::Tensor(const std::vector<std::size_t> &shape) : shape_(shape)
{
    std::size_t num_e = this->numel();
    this->data() = std::vector<T>(num_e);
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B>::Tensor(const std::vector<std::size_t> &shape, const Tensor<T, B>::TStorage &data)
{
    std::size_t shape_size = std::reduce(shape.begin(), shape.end(), 1, std::multiplies<int>());
    std::size_t buffer_size = data.size();
    if (shape_size != buffer_size)
        throw TensorInvalidShapeException(
            std::format("Shape and Buffer number of elements are incompatible : {} and {}.", shape_size, buffer_size));
    this->shape_ = shape;
    this->data_ = data;
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B>::~Tensor()
{
}

template <typename T, typename B>
requires IsBackend<T, B>
std::vector<T>::iterator Tensor<T, B>::begin()
{
    return this->data().begin();
}

template <typename T, typename B>
requires IsBackend<T, B>
std::vector<T>::iterator Tensor<T, B>::const_begin() const
{
    return this->data().begin();
}

template <typename T, typename B>
requires IsBackend<T, B>
std::vector<T>::iterator Tensor<T, B>::end()
{
    return this->data().end();
}

template <typename T, typename B>
requires IsBackend<T, B>
std::vector<T>::iterator Tensor<T, B>::const_end() const
{
    return this->data().end();
}

template <typename T, typename B>
requires IsBackend<T, B>
T &Tensor<T, B>::operator[](std::size_t pos)
{
    return this->data_[pos];
}

template <typename T, typename B>
requires IsBackend<T, B>
const T &Tensor<T, B>::operator[](std::size_t pos) const
{
    return this->data_[pos];
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B>::TStorage &Tensor<T, B>::data()
{
    return this->data_;
}

template <typename T, typename B>
requires IsBackend<T, B>
std::vector<size_t> &Tensor<T, B>::shape()
{
    return this->shape_;
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B> &Tensor<T, B>::fill(T value)
{
    for (std::size_t i = 0; i < this->numel(); i++)
        this->data()[i] = value;
    return *this;
}

template <typename T, typename B>
requires IsBackend<T, B>
std::size_t Tensor<T, B>::numel() const
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

template <typename T, typename B>
requires IsBackend<T, B>
T Tensor<T, B>::item() const
{
    if (this->numel() != 1)
        throw TensorInvalidShapeException(std::format("Tensor .item() only works on single-element tensor : {}",
                                                      this->tensorShapeToStr(this->shape_)));
    return this->data()[0];
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B> &Tensor<T, B>::flatten()
{
    this->shape_ = std::vector({this->numel()});
    return *this;
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B> &Tensor<T, B>::unsqueeze(std::size_t dim)
{
    this->shape_.insert(this->shape_.begin() + dim, 1);
    return *this;
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B> &Tensor<T, B>::squeeze(std::size_t dim)
{
    if (this->shape_[dim] != 1)
        throw TensorSqueezeException(std::format("Cannot squeeze at dim {} : non-1 dimension.", dim));
    this->shape_.erase(this->shape_.begin() + dim);
    return *this;
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B> &Tensor<T, B>::t(std::size_t dim0, std::size_t dim1)
{
    return this->transpose(dim0, dim1);
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B> &Tensor<T, B>::transpose(std::size_t dim0, std::size_t dim1)
{
    if (this->shape_.size() < 2)
        throw TensorTransposeException("Cannot transpose tensor with less than 2 dimensions");

    std::vector<std::size_t> new_shape = this->shape_;
    std::swap(new_shape[dim0], new_shape[dim1]);
    Tensor<T, B> tensor = Tensor<T, B>(new_shape);
    for (std::size_t i = 0; i < this->numel(); i++)
    {
        std::vector<std::size_t> target_coord = this->absToCoord(i);
        std::swap(target_coord[dim0], target_coord[dim1]);
        tensor[tensor.coordToAbs(target_coord)] = this->data()[i];
    }

    this->shape_ = tensor.shape_;
    this->data() = tensor.data();

    return *this;
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B> &Tensor<T, B>::broadcast(Tensor<T, B> &tensor)
{
    return this->broadcast(tensor.shape_);
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B> &Tensor<T, B>::broadcast(const std::vector<std::size_t> &shape)
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
            throw TensorBroadcastException(std::format("Tensor is not broadcastable to this shape : {} to "
                                                       "{} at index {}.",
                                                       this->tensorShapeToStr(tensor_shape),
                                                       this->tensorShapeToStr(target_shape), i));
        new_shape.insert(new_shape.end(), tensor_shape[i] >= target_shape[i] ? tensor_shape[i] : target_shape[i]);
    }

    Tensor<T, B> new_tensor = Tensor<T, B>(new_shape);
    for (std::size_t i = 0; i < new_tensor.numel(); i++)
    {
        std::vector<std::size_t> coords = new_tensor.absToCoord(i);
        for (std::size_t j = 0; j < coords.size(); j++)
            coords[j] = tensor_shape[j] != 1 ? coords[j] : 0;
        new_tensor[i] = this->data()[this->coordToAbs(coords)];
    }

    this->shape_ = new_tensor.shape_;
    this->data() = new_tensor.data();
    return *this;
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B> &Tensor<T, B>::batch_broadcast(Tensor<T, B> &tensor)
{
    return this->batch_broadcast(tensor.shape_);
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B> &Tensor<T, B>::batch_broadcast(const std::vector<std::size_t> &shape)
{
    // broadcast only on non-matrix dimensions (ie batch, channels...)
    std::vector<std::size_t> new_shape = shape;
    if (new_shape[new_shape.size() - 1] == 1)
        new_shape[new_shape.size() - 1] = this->shape_[this->shape_.size() - 1];
    if (new_shape[new_shape.size() - 2] == 2)
        new_shape[new_shape.size() - 2] = this->shape_[this->shape_.size() - 2];
    return this->broadcast(new_shape);
}
