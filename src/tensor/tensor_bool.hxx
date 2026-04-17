#include "src/exception/exception.hh"
#include "src/utils.hh"
#include "tensor.hh"

#include <format>

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<bool, B> Tensor<T, B>::operator==(const Tensor<T, B> &other)
{
    if (!this->validateSameShape(other.shape_))
        throw TensorInvalidShapeException(std::format("Shape {} and {} are invalid for comparison.",
                                                      this->tensorShapeToStr(this->shape_),
                                                      other.tensorShapeToStr(other.shape_)));

    Tensor<bool, B> tensor = Tensor<bool, B>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor[i] = (bool)(this[i] == other[i]);
    return tensor;
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<bool, B> Tensor<T, B>::operator==(const T &other)
{
    Tensor<bool, B> tensor = Tensor<bool, B>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor[i] = (bool)(this[i] == other);
    return tensor;
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<bool, B> Tensor<T, B>::operator<(const Tensor<T, B> &other)
{
    if (!this->validateSameShape(other.shape_))
        throw TensorInvalidShapeException(std::format("Shape {} and {} are invalid for comparison.",
                                                      this->tensorShapeToStr(this->shape_),
                                                      other.tensorShapeToStr(other.shape_)));

    Tensor<bool, B> tensor = Tensor<bool, B>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor[i] = (bool)(this[i] < other[i]);
    return tensor;
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<bool, B> Tensor<T, B>::operator<(const T &other)
{
    Tensor<bool, B> tensor = Tensor<bool, B>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor[i] = (bool)(this[i] < other);
    return tensor;
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<bool, B> Tensor<T, B>::operator<=(const Tensor<T, B> &other)
{
    if (!this->validateSameShape(other.shape_))
        throw TensorInvalidShapeException(std::format("Shape {} and {} are invalid for comparison.",
                                                      this->tensorShapeToStr(this->shape_),
                                                      other.tensorShapeToStr(other.shape_)));

    Tensor<bool, B> tensor = Tensor<bool, B>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor[i] = (bool)(this[i] <= other[i]);
    return tensor;
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<bool, B> Tensor<T, B>::operator<=(const T &other)
{
    Tensor<bool, B> tensor = Tensor<bool, B>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor[i] = (bool)(this[i] <= other);
    return tensor;
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<bool, B> Tensor<T, B>::operator>(const Tensor<T, B> &other)
{
    if (!this->validateSameShape(other.shape_))
        throw TensorInvalidShapeException(std::format("Shape {} and {} are invalid for comparison.",
                                                      this->tensorShapeToStr(this->shape_),
                                                      other.tensorShapeToStr(other.shape_)));

    Tensor<bool, B> tensor = Tensor<bool, B>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor[i] = (bool)(this[i] > other[i]);
    return tensor;
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<bool, B> Tensor<T, B>::operator>(const T &other)
{
    Tensor<bool, B> tensor = Tensor<bool, B>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor[i] = (bool)(this[i] > other);
    return tensor;
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<bool, B> Tensor<T, B>::operator>=(const Tensor<T, B> &other)
{
    if (!this->validateSameShape(other.shape_))
        throw TensorInvalidShapeException(std::format("Shape {} and {} are invalid for comparison.",
                                                      this->tensorShapeToStr(this->shape_),
                                                      other.tensorShapeToStr(other.shape_)));

    Tensor<bool, B> tensor = Tensor<bool, B>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor[i] = (bool)(this[i] >= other[i]);
    return tensor;
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<bool, B> Tensor<T, B>::operator>=(const T &other)
{
    Tensor<bool, B> tensor = Tensor<bool, B>(this->shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor[i] = (bool)(this[i] >= other);
    return tensor;
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<T, B>::operator bool() const
{
    if constexpr (std::is_same_v<T, bool>)
    {
        if (this->numel() != 1)
            throw CastException("Boolean cast of a non single element Tensor<bool> is "
                                "ambiguous. Use .all() or .any() "
                                "depending on the behaviour you want.");
        else
            return this[0];
    }
    else
    {
        throw TensorInvalidTypeException(
            std::format("bool() can only be used on Tensor<bool>. Got Tensor<{}>.", type_name<T>()));
    }
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<bool, B> Tensor<T, B>::all() const
{
    if constexpr (std::is_same_v<T, bool>)
    {
        for (std::size_t i = 0; i < this->numel(); i++)
            if (!this[i])
                return Tensor<bool, B>({1}, {false});
        return Tensor<bool, B>({1}, {true});
    }
    else
    {
        throw TensorInvalidTypeException(
            std::format("all() can only be used on Tensor<bool>. Got Tensor<{}>.", type_name<T>()));
    }
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<bool, B> Tensor<T, B>::any() const
{
    if constexpr (std::is_same_v<T, bool>)
    {
        for (std::size_t i = 0; i < this->numel(); i++)
            if (this[i])
                return Tensor<bool, B>({1}, {true});
        return Tensor<bool, B>({1}, {false});
    }
    else
    {
        throw TensorInvalidTypeException(
            std::format("any() can only be used on Tensor<bool>. Got Tensor<{}>.", type_name<T>()));
    }
}

template <typename T, typename B>
requires IsBackend<T, B>
Tensor<bool, B> Tensor<T, B>::none() const
{
    if constexpr (std::is_same_v<T, bool>)
    {
        for (std::size_t i = 0; i < this->numel(); i++)
            if (this[i])
                return Tensor<bool, B>({1}, {false});
        return Tensor<bool, B>({1}, {true});
    }
    else
    {
        throw TensorInvalidTypeException(
            std::format("none() can only be used on Tensor<bool>. Got Tensor<{}>.", type_name<T>()));
    }
}