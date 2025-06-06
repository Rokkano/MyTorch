#include "tensor.hh"

template <typename T>
Tensor<T> Tensor<T>::min()
{
    T min = this->buffer_[0];
    for (std::size_t i = 0; i < this->numel(); i++)
        if (this->buffer_[i] < min)
            min = this->buffer_[i];
    return Tensor<T>({1}, {min});
}

template <typename T>
Tensor<T> Tensor<T>::min(const T &val)
{
    Tensor<T> tensor = Tensor<T>(this->shape_);
    for (std::size_t i = 0; i < this->numel(); i++)
        tensor.buffer_[i] = val < this->buffer_[i] ? val : this->buffer_[i];
    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::amin(const std::size_t dim)
{
    std::vector<std::size_t> new_shape = this->shape_;
    new_shape.erase(new_shape.begin() + dim);

    Tensor<T> tensor = Tensor<T>(new_shape);
    for (std::size_t i = 0; i < tensor.numel(); i++)
    {
        std::vector<T> min_buff = std::vector<T>();
        for (std::size_t k = 0; k < this->shape_[dim]; k++)
        {
            std::vector<std::size_t> coords = tensor.absToCoord(i);
            coords.insert(coords.begin() + dim, k);
            min_buff.insert(min_buff.end(), this->buffer_[this->coordToAbs(coords)]);
        }
        tensor.buffer_[i] = Tensor<int>({this->shape_[dim]}, min_buff).min().item();
    }
    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::min(const Tensor<T> &other)
{
    if (!this->validateSameShape(other.shape_))
        throw TensorInvalidShapeException(std::format("Shape {} and {} are invalid for comparison.", this->tensorShapeToStr(this->shape_), other.tensorShapeToStr(other.shape_)));

    Tensor<T> tensor = Tensor<T>(this->shape_);
    for (std::size_t i = 0; i < this->numel(); i++)
        tensor.buffer_[i] = other.buffer_[i] < this->buffer_[i] ? other.buffer_[i] : this->buffer_[i];
    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::max()
{
    T min = this->buffer_[0];
    for (std::size_t i = 0; i < this->numel(); i++)
        if (this->buffer_[i] > min)
            min = this->buffer_[i];
    return Tensor<T>({1}, {min});
}

template <typename T>
Tensor<T> Tensor<T>::max(const T &val)
{
    Tensor<T> tensor = Tensor<T>(this->shape_);
    for (std::size_t i = 0; i < this->numel(); i++)
        tensor.buffer_[i] = val > this->buffer_[i] ? val : this->buffer_[i];
    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::max(const Tensor<T> &other)
{
    if (!this->validateSameShape(other.shape_))
        throw TensorInvalidShapeException(std::format("Shape {} and {} are invalid for comparison.", this->tensorShapeToStr(this->shape_), other.tensorShapeToStr(other.shape_)));

    Tensor<T> tensor = Tensor<T>(this->shape_);
    for (std::size_t i = 0; i < this->numel(); i++)
        tensor.buffer_[i] = other.buffer_[i] > this->buffer_[i] ? other.buffer_[i] : this->buffer_[i];
    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::amax(const std::size_t dim)
{
    std::vector<std::size_t> new_shape = this->shape_;
    new_shape.erase(new_shape.begin() + dim);

    Tensor<T> tensor = Tensor<T>(new_shape);
    for (std::size_t i = 0; i < tensor.numel(); i++)
    {
        std::vector<T> min_buff = std::vector<T>();
        for (std::size_t k = 0; k < this->shape_[dim]; k++)
        {
            std::vector<std::size_t> coords = tensor.absToCoord(i);
            coords.insert(coords.begin() + dim, k);
            min_buff.insert(min_buff.end(), this->buffer_[this->coordToAbs(coords)]);
        }
        tensor.buffer_[i] = Tensor<int>({this->shape_[dim]}, min_buff).max().item();
    }
    return tensor;
}