#pragma once

#include "src/exception/exception.hh"
#include "tensor.hh"
#include "src/tensor/tensor_io.hxx"

#include <format>
#include <iostream>
#include <numeric>

template <typename T, template <typename> typename B>
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

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
bool Tensor<T, B>::validateAbs(std::size_t abs) const
{
    return abs < this->numel();
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
std::size_t Tensor<T, B>::coordToAbs(const std::vector<std::size_t> &coord) const
{
    // [x] with [x_m] -> x
    // [y,x] with [y_m,x_m] -> y * x_m + x
    // [z,y,x] with [z_m,y_m,x_m] -> z * y_m * x_m + y * x_m + x
    if (!this->validateCoord(coord))
        throw std::invalid_argument(std::format("Coordinates {} are invalid for tensor of shape {}.",
                                                this->shapeToStr(coord), this->shapeToStr(this->shape_)));

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

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
std::vector<std::size_t> Tensor<T, B>::absToCoord(std::size_t abs) const
{
    // [x] with [x_m] -> x
    // [y,x] with [y_m,x_m] -> y * x_m + x
    // [z,y,x] with [z_m,y_m,x_m] -> z * y_m * x_m + y * x_m + x
    if (!this->validateAbs(abs))
        throw std::invalid_argument(std::format("Absolute {} is invalid for tensor of shape {} ({} elements).",
                                                std::to_string(abs), this->shapeToStr(this->shape_),
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

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B>::Tensor()
{
    this->shape_ = std::vector<std::size_t>(0);
    this->stride_ = std::vector<std::size_t>(0);
    this->numel_ = 0;
    this->data_ = B<T>::allocate(0);
}


template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B>::Tensor(const std::vector<std::size_t> &shape)
{
    this->shape_ = shape;
    this->numel_ = std::reduce(shape.begin(), shape.end(), 1, std::multiplies<std::size_t>());
    this->data_ = B<T>::allocate(this->numel_);

    // compute strides
    this->stride_ = std::vector<std::size_t>(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; --i)
        this->stride_[i] = this->stride_[i + 1] * shape[i + 1];
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B>::Tensor(const std::vector<std::size_t> &shape, const Tensor<T, B>::TStorage &data)
{
    std::size_t numel = std::reduce(shape.begin(), shape.end(), 1, std::multiplies<std::size_t>());
    std::size_t dataSize = data.size();
    if (numel != dataSize)
        throw TensorInvalidShapeException(
            std::format("Shape and Buffer number of elements are incompatible : {} and {}.", numel, dataSize));
    this->shape_ = shape;
    this->numel_ = numel;
    this->data_ = data;

    // compute strides
    this->stride_ = std::vector<std::size_t>(shape.size(), 1);
    for (int i = (int)shape.size() - 2; i >= 0; --i)
        this->stride_[i] = this->stride_[i + 1] * shape[i + 1];
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B>::~Tensor()
{
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
std::vector<T>::iterator Tensor<T, B>::begin()
{
    return this->data().begin();
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
std::vector<T>::iterator Tensor<T, B>::const_begin() const
{
    return this->data().begin();
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
std::vector<T>::iterator Tensor<T, B>::end()
{
    return this->data().end();
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
std::vector<T>::iterator Tensor<T, B>::const_end() const
{
    return this->data().end();
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
T &Tensor<T, B>::operator[](std::size_t pos)
{
    return this->data_[pos];
}


template <typename T, template <typename> typename B>
requires IsBackend<T, B>
const T &Tensor<T, B>::operator[](std::size_t pos) const
{
    return this->data_[pos];
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B>::TStorage &Tensor<T, B>::data()
{
    return this->data_;
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
std::vector<size_t> &Tensor<T, B>::shape()
{
    return this->shape_;
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> &Tensor<T, B>::fill(T value)
{
    for (std::size_t i = 0; i < this->numel(); i++)
        this->data()[i] = value;
    return *this;
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
std::size_t Tensor<T, B>::numel() const
{
    return this->numel_;
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
std::vector<std::size_t> Tensor<T, B>::stride() const
{
    return this->stride_;
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
T Tensor<T, B>::item() const
{
    if (this->numel() != 1)
        throw TensorInvalidShapeException(std::format("Tensor .item() only works on single-element tensor : {}",
                                                      this->shapeToStr(this->shape_)));
    return (*this)[0];
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> &Tensor<T, B>::flatten()
{
    this->shape_ = std::vector({this->numel()});
    return *this;
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> &Tensor<T, B>::unsqueeze(std::size_t dim)
{
    this->shape_.insert(this->shape_.begin() + dim, 1);
    return *this;
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> &Tensor<T, B>::squeeze(std::size_t dim)
{
    if (this->shape_[dim] != 1)
        throw TensorSqueezeException(std::format("Cannot squeeze at dim {} : non-1 dimension.", dim));
    this->shape_.erase(this->shape_.begin() + dim);
    return *this;
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> Tensor<T, B>::t(std::size_t dim0, std::size_t dim1)
{
    return this->transpose(dim0, dim1);
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> Tensor<T, B>::transpose(std::size_t dim0, std::size_t dim1)
{
    if (this->shape_.size() < 2)
        throw TensorTransposeException("Cannot transpose tensor with less than 2 dimensions");

    if (dim0 >= this->shape_.size() || dim1 >= this->shape_.size())
        throw TensorTransposeException("Invalid dimension to transpose");

    if (dim0 == dim1)
        return *this;

    std::vector<std::size_t> new_shape = this->shape_;
    std::swap(new_shape[dim0], new_shape[dim1]);
    Tensor<T, B> tensor = Tensor<T, B>(new_shape);

    std::vector<std::size_t> srcStrides = this->stride();
    std::vector<std::size_t> dstStrides = tensor.stride();

    for (size_t flatIdx = 0; flatIdx < this->numel(); flatIdx++) {
        std::vector<size_t> dstIdx = std::vector<size_t>(this->shape_.size());
        size_t srcIdx = 0;
        size_t tmp = flatIdx;

        for (size_t i = 0; i < this->shape_.size(); i++) {
            dstIdx[i] = tmp / dstStrides[i];
            tmp = tmp % dstStrides[i];
        }

        for (size_t i = 0; i < this->shape_.size(); i++) {
            if (i == dim0) 
                srcIdx += dstIdx[i] * srcStrides[dim1];
            else if (i == dim1) 
                srcIdx += dstIdx[i] * srcStrides[dim0];
            else
                srcIdx += dstIdx[i] * srcStrides[i];
        }

        tensor[flatIdx] = (*this)[srcIdx];
    }

    return tensor;
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> &Tensor<T, B>::broadcast(Tensor<T, B> &tensor)
{
    return this->broadcast(tensor.shape_);
}

template <typename T, template <typename> typename B>
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
                                                       this->shapeToStr(tensor_shape),
                                                       this->shapeToStr(target_shape), i));
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

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> &Tensor<T, B>::batch_broadcast(Tensor<T, B> &tensor)
{
    return this->batch_broadcast(tensor.shape_);
}

template <typename T, template <typename> typename B>
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



template <typename T, template <typename> typename B>
requires IsBackend<T, B>
std::string Tensor<T, B>::shapeToStr(const std::vector<std::size_t> &shape)
{
    std::stringstream ssShape;
    for (std::size_t i = 0; i < shape.size(); i++)
        ssShape << shape[i] << (i != shape.size() - 1 ? "," : "");
    return ssShape.str();
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
std::string Tensor<T, B>::dataToStr(const Tensor<T, B>::TStorage &storage, const std::vector<std::size_t> &shape, std::size_t truncate)
{
    std::stringstream ssData;
    using RecLambda = std::function<void(std::vector<std::size_t>, std::size_t)>;
    RecLambda rec = [&](std::vector<std::size_t> shape, std::size_t index) 
    { 
        if (!shape.empty())
        {
            std::size_t step = std::reduce(shape.begin() + 1, shape.end(), 1, std::multiplies<int>());
            std::vector<std::size_t> new_shape = std::vector<std::size_t>(shape.begin() + 1, shape.end());
            ssData << "[";
            for (std::size_t i = 0; i < shape[0]; i++)
            {
                rec(new_shape, index + i * step);
                ssData << (i != shape[0] - 1 ? "," : "");
            }
            ssData << "]";
        }
        else
        { 
            if constexpr (std::is_same_v<T, bool>) 
                ssData << (storage[index] ? "true" : "false");
            else
                ssData << storage[index];
        }
    };
    
    rec(shape, 0);

    if (truncate != 0)
        return ssData.str().substr(0, truncate) + "...";
    else
        return ssData.str();
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
std::string Tensor<T, B>::dataToStr(const Tensor<T, B>::TStorage &storage, std::size_t truncate)
{
    return Tensor<T, B>::dataToStr(storage, {storage.size()}, truncate);
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
std::string Tensor<T, B>::toStr() const
{
    std::string sShape = Tensor<T, B>::shapeToStr(this->shape_);
    std::string sData = Tensor<T, B>::dataToStr(this->data_, this->shape_);
    return "tensor(shape=(" + sShape + "); data=(" + sData + "); dtype=(" + type_name<T>() + "))";
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
std::ostream &operator<<(std::ostream &os, const Tensor<T, B> &t)
{
    return os << t.toStr();
}