#include <cmath>
#include <format>

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
    T max = this->buffer_[0];
    for (std::size_t i = 0; i < this->numel(); i++)
        if (this->buffer_[i] > max)
            max = this->buffer_[i];
    return Tensor<T>({1}, {max});
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
        std::vector<T> max_buff = std::vector<T>();
        for (std::size_t k = 0; k < this->shape_[dim]; k++)
        {
            std::vector<std::size_t> coords = tensor.absToCoord(i);
            coords.insert(coords.begin() + dim, k);
            max_buff.insert(max_buff.end(), this->buffer_[this->coordToAbs(coords)]);
        }
        tensor.buffer_[i] = Tensor<int>({this->shape_[dim]}, max_buff).max().item();
    }
    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::mean(int bessel_correction)
{
    T mean = 0;
    for (std::size_t i = 0; i < this->numel(); i++)
        mean += this->buffer_[i];
    T val = mean / std::max(std::size_t(0), this->numel() - bessel_correction);
    return Tensor<T>({1}, {val});
}

template <typename T>
Tensor<T> Tensor<T>::amean(const std::size_t dim, int bessel_correction)
{
    std::vector<std::size_t> new_shape = this->shape_;
    new_shape.erase(new_shape.begin() + dim);

    Tensor<T> tensor = Tensor<T>(new_shape);
    for (std::size_t i = 0; i < tensor.numel(); i++)
    {
        T mean = 0;
        for (std::size_t k = 0; k < this->shape_[dim]; k++)
        {
            std::vector<std::size_t> coords = tensor.absToCoord(i);
            coords.insert(coords.begin() + dim, k);
            mean += this->buffer_[this->coordToAbs(coords)];
        }
        tensor.buffer_[i] = mean / std::max(std::size_t(0), this->shape_[dim] - bessel_correction);
    }
    return tensor;
}

template <typename T>
Tensor<T> Tensor<T>::var(int bessel_correction)
{
    T sum_diff_squared = 0;
    T mean = this->mean().item();
    std::size_t num_elements = this->numel();
    for (std::size_t i = 0; i < num_elements; i++)
        sum_diff_squared += std::pow(this->buffer_[i] - mean, 2);
    T val = (sum_diff_squared / std::max(std::size_t(0), num_elements - bessel_correction));
    return Tensor<T>({1}, {val});
}

template <typename T>
Tensor<T> Tensor<T>::std(int bessel_correction)
{
    return Tensor<T>::sqrt(this->var());
}

template <typename T>
Tensor<std::size_t> Tensor<T>::argmin()
{
    T min = this->buffer_[0];
    std::size_t min_index = 0;
    for (std::size_t i = 0; i < this->numel(); i++)
        if (this->buffer_[i] < min)
        {
            min = this->buffer_[i];
            min_index = i;
        }
    return Tensor<std::size_t>({1}, {min_index});
}

template <typename T>
Tensor<std::size_t> Tensor<T>::argmin(const std::size_t dim)
{
    std::vector<std::size_t> new_shape = this->shape_;
    new_shape.erase(new_shape.begin() + dim);

    Tensor<std::size_t> tensor = Tensor<std::size_t>(new_shape);
    for (std::size_t i = 0; i < tensor.numel(); i++)
    {
        std::vector<T> min_buff = std::vector<T>();
        for (std::size_t k = 0; k < this->shape_[dim]; k++)
        {
            std::vector<std::size_t> coords = tensor.absToCoord(i);
            coords.insert(coords.begin() + dim, k);
            min_buff.insert(min_buff.end(), this->buffer_[this->coordToAbs(coords)]);
        }
        std::cout << Tensor<T>({this->shape_[dim]}, min_buff) << std::endl;
        tensor.buffer_[i] = Tensor<T>({this->shape_[dim]}, min_buff).argmin().item();
    }
    return tensor;
}

template <typename T>
Tensor<std::size_t> Tensor<T>::argmax()
{
    T max = this->buffer_[0];
    std::size_t max_index = 0;
    for (std::size_t i = 0; i < this->numel(); i++)
        if (this->buffer_[i] > max)
        {
            max = this->buffer_[i];
            max_index = i;
        }
    return Tensor<std::size_t>({1}, {max_index});
}

template <typename T>
Tensor<std::size_t> Tensor<T>::argmax(const std::size_t dim)
{
    std::vector<std::size_t> new_shape = this->shape_;
    new_shape.erase(new_shape.begin() + dim);

    Tensor<std::size_t> tensor = Tensor<std::size_t>(new_shape);
    for (std::size_t i = 0; i < tensor.numel(); i++)
    {
        std::vector<T> max_buff = std::vector<T>();
        for (std::size_t k = 0; k < this->shape_[dim]; k++)
        {
            std::vector<std::size_t> coords = tensor.absToCoord(i);
            coords.insert(coords.begin() + dim, k);
            max_buff.insert(max_buff.end(), this->buffer_[this->coordToAbs(coords)]);
        }
        std::cout << Tensor<T>({this->shape_[dim]}, max_buff) << std::endl;
        tensor.buffer_[i] = Tensor<T>({this->shape_[dim]}, max_buff).argmax().item();
    }
    return tensor;
}