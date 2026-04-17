#include "tensor.hh"

#include <cmath>
#include <format>

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B> Tensor<T, B>::min() const requires std::is_arithmetic_v<T>
{
    T min = this[0];
    for (std::size_t i = 0; i < this->numel(); i++)
        if (this[i] < min)
            min = this[i];
    return Tensor<T, B>({1}, {min});
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B> Tensor<T, B>::min(const T &val) const requires std::is_arithmetic_v<T>
{
    Tensor<T, B> tensor = Tensor<T, B>(this->shape_);
    for (std::size_t i = 0; i < this->numel(); i++)
        tensor[i] = val < this[i] ? val : this[i];
    return tensor;
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B> Tensor<T, B>::amin(const std::size_t dim) const requires std::is_arithmetic_v<T>
{
    std::vector<std::size_t> new_shape = this->shape_;
    new_shape.erase(new_shape.begin() + dim);

    Tensor<T, B> tensor = Tensor<T, B>(new_shape);
    for (std::size_t i = 0; i < tensor.numel(); i++)
    {
        std::vector<T> min_buff = std::vector<T>();
        for (std::size_t k = 0; k < this->shape_[dim]; k++)
        {
            std::vector<std::size_t> coords = tensor.absToCoord(i);
            coords.insert(coords.begin() + dim, k);
            min_buff.insert(min_buff.end(), this[this->coordToAbs(coords)]);
        }
        tensor[i] = Tensor<int, B>({this->shape_[dim]}, min_buff).min().item();
    }
    return tensor;
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B> Tensor<T, B>::min(const Tensor<T, B> &other) const requires std::is_arithmetic_v<T>
{
    if (!this->validateSameShape(other.shape_))
        throw TensorInvalidShapeException(std::format("Shape {} and {} are invalid for comparison.",
                                                      this->tensorShapeToStr(this->shape_),
                                                      other.tensorShapeToStr(other.shape_)));

    Tensor<T, B> tensor = Tensor<T, B>(this->shape_);
    for (std::size_t i = 0; i < this->numel(); i++)
        tensor[i] = other[i] < this[i] ? other[i] : this[i];
    return tensor;
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B> Tensor<T, B>::max() const requires std::is_arithmetic_v<T>
{
    T max = this[0];
    for (std::size_t i = 0; i < this->numel(); i++)
        if (this[i] > max)
            max = this[i];
    return Tensor<T, B>({1}, {max});
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B> Tensor<T, B>::max(const T &val) const requires std::is_arithmetic_v<T>
{
    Tensor<T, B> tensor = Tensor<T, B>(this->shape_);
    for (std::size_t i = 0; i < this->numel(); i++)
        tensor[i] = val > this[i] ? val : this[i];
    return tensor;
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B> Tensor<T, B>::max(const Tensor<T, B> &other) const requires std::is_arithmetic_v<T>
{
    if (!this->validateSameShape(other.shape_))
        throw TensorInvalidShapeException(std::format("Shape {} and {} are invalid for comparison.",
                                                      this->tensorShapeToStr(this->shape_),
                                                      other.tensorShapeToStr(other.shape_)));

    Tensor<T, B> tensor = Tensor<T, B>(this->shape_);
    for (std::size_t i = 0; i < this->numel(); i++)
        tensor[i] = other[i] > this[i] ? other[i] : this[i];
    return tensor;
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B> Tensor<T, B>::amax(const std::size_t dim) const requires std::is_arithmetic_v<T>
{
    std::vector<std::size_t> new_shape = this->shape_;
    new_shape.erase(new_shape.begin() + dim);

    Tensor<T, B> tensor = Tensor<T, B>(new_shape);
    for (std::size_t i = 0; i < tensor.numel(); i++)
    {
        std::vector<T> max_buff = std::vector<T>();
        for (std::size_t k = 0; k < this->shape_[dim]; k++)
        {
            std::vector<std::size_t> coords = tensor.absToCoord(i);
            coords.insert(coords.begin() + dim, k);
            max_buff.insert(max_buff.end(), this[this->coordToAbs(coords)]);
        }
        tensor[i] = Tensor<int, B>({this->shape_[dim]}, max_buff).max().item();
    }
    return tensor;
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B> Tensor<T, B>::mean(int bessel_correction) const requires std::is_arithmetic_v<T>
{
    T mean = 0;
    for (std::size_t i = 0; i < this->numel(); i++)
        mean += this[i];
    T val = mean / std::max(std::size_t(0), this->numel() - bessel_correction);
    return Tensor<T, B>({1}, {val});
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B> Tensor<T, B>::amean(const std::size_t dim, int bessel_correction) const requires std::is_arithmetic_v<T>
{
    std::vector<std::size_t> new_shape = this->shape_;
    new_shape.erase(new_shape.begin() + dim);

    Tensor<T, B> tensor = Tensor<T, B>(new_shape);
    for (std::size_t i = 0; i < tensor.numel(); i++)
    {
        T mean = 0;
        for (std::size_t k = 0; k < this->shape_[dim]; k++)
        {
            std::vector<std::size_t> coords = tensor.absToCoord(i);
            coords.insert(coords.begin() + dim, k);
            mean += this[this->coordToAbs(coords)];
        }
        tensor[i] = mean / std::max(std::size_t(0), this->shape_[dim] - bessel_correction);
    }
    return tensor;
};

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B> Tensor<T, B>::var(int bessel_correction) const requires std::is_arithmetic_v<T>
{
    T sum_diff_squared = 0;
    T mean = this->mean().item();
    std::size_t num_elements = this->numel();
    for (std::size_t i = 0; i < num_elements; i++)
        sum_diff_squared += std::pow(this[i] - mean, 2);
    T val = (sum_diff_squared / std::max(std::size_t(0), num_elements - bessel_correction));
    return Tensor<T, B>({1}, {val});
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B> Tensor<T, B>::std(int bessel_correction) const requires std::is_arithmetic_v<T>
{
    return Tensor<T, B>::sqrt(this->var());
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<std::size_t, B> Tensor<T, B>::argmin() const requires std::is_arithmetic_v<T>
{
    T min = this[0];
    std::size_t min_index = 0;
    for (std::size_t i = 0; i < this->numel(); i++)
        if (this[i] < min)
        {
            min = this[i];
            min_index = i;
        }
    return Tensor<std::size_t, B>({1}, {min_index});
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<std::size_t, B> Tensor<T, B>::argmin(const std::size_t dim) const requires std::is_arithmetic_v<T>
{
    std::vector<std::size_t> new_shape = this->shape_;
    new_shape.erase(new_shape.begin() + dim);

    Tensor<std::size_t, B> tensor = Tensor<std::size_t, B>(new_shape);
    for (std::size_t i = 0; i < tensor.numel(); i++)
    {
        std::vector<T> min_buff = std::vector<T>();
        for (std::size_t k = 0; k < this->shape_[dim]; k++)
        {
            std::vector<std::size_t> coords = tensor.absToCoord(i);
            coords.insert(coords.begin() + dim, k);
            min_buff.insert(min_buff.end(), this[this->coordToAbs(coords)]);
        }
        std::cout << Tensor<T, B>({this->shape_[dim]}, min_buff) << std::endl;
        tensor[i] = Tensor<T, B>({this->shape_[dim]}, min_buff).argmin().item();
    }
    return tensor;
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<std::size_t, B> Tensor<T, B>::argmax() const requires std::is_arithmetic_v<T>
{
    T max = this[0];
    std::size_t max_index = 0;
    for (std::size_t i = 0; i < this->numel(); i++)
        if (this[i] > max)
        {
            max = this[i];
            max_index = i;
        }
    return Tensor<std::size_t, B>({1}, {max_index});
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<std::size_t, B> Tensor<T, B>::argmax(const std::size_t dim) const requires std::is_arithmetic_v<T>
{
    std::vector<std::size_t> new_shape = this->shape_;
    new_shape.erase(new_shape.begin() + dim);

    Tensor<std::size_t, B> tensor = Tensor<std::size_t, B>(new_shape);
    for (std::size_t i = 0; i < tensor.numel(); i++)
    {
        std::vector<T> max_buff = std::vector<T>();
        for (std::size_t k = 0; k < this->shape_[dim]; k++)
        {
            std::vector<std::size_t> coords = tensor.absToCoord(i);
            coords.insert(coords.begin() + dim, k);
            max_buff.insert(max_buff.end(), this[this->coordToAbs(coords)]);
        }
        tensor[i] = Tensor<T, B>({this->shape_[dim]}, max_buff).argmax().item();
    }
    return tensor;
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B> Tensor<T, B>::sum() const requires std::is_arithmetic_v<T>
{
    T sum = 0;
    for (std::size_t i = 0; i < this->numel(); i++)
        sum += this[i];
    return Tensor<T, B>({1}, {sum});
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B> Tensor<T, B>::heaviside() const requires std::is_arithmetic_v<T>
{
    // This is the regular definition of heaviside step function : H(x) = {1 if
    // x>=0, 0 if x < 0}
    Tensor<T, B> tensor = Tensor<T, B>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor[i] = this[i] >= 0 ? 1 : 0;
    return tensor;
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B> Tensor<T, B>::round() const requires std::is_arithmetic_v<T>
{
    Tensor<T, B> tensor = Tensor<T, B>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor[i] = std::round(this[i]);
    return tensor;
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B> Tensor<T, B>::floor() const requires std::is_arithmetic_v<T>
{
    Tensor<T, B> tensor = Tensor<T, B>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor[i] = std::floor(this[i]);
    return tensor;
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B> Tensor<T, B>::ceil() const requires std::is_arithmetic_v<T>
{
    Tensor<T, B> tensor = Tensor<T, B>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor[i] = std::ceil(this[i]);
    return tensor;
}