#include "cpp_backend.hh"

#include <cmath>
#include <format>
#include <optional>
#include <typeinfo>

template <typename T>
Tensor<T, CppBackend<T>> CppBackend<T>::affine(const Tensor<T, CppBackend<T>> &rhs, std::optional<T> a,
                                               std::optional<T> b) requires std::is_arithmetic_v<T>
{
    Tensor<T, CppBackend<T>> tensor = Tensor<T, CppBackend<T>>(rhs.shape_);
    for (std::size_t i = 0; i < rhs.numel(); i++)
    {
        T value = rhs.buffer_[i];
        if (a.has_value())
            value = value * a.value();
        if (b.has_value())
            value = value + b.value();
        tensor.buffer_[i] = value;
    }
    return tensor;
}

template <typename T>
Tensor<T, CppBackend<T>> CppBackend<T>::exp(const Tensor<T, CppBackend<T>> &t) requires std::is_arithmetic_v<T>
{
    Tensor<T, CppBackend<T>> tensor = Tensor<T, CppBackend<T>>(t.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = std::exp(t.buffer_[i]);
    return tensor;
}

template <typename T>
Tensor<T, CppBackend<T>> CppBackend<T>::log(const Tensor<T, CppBackend<T>> &t) requires std::is_arithmetic_v<T>
{
    Tensor<T, CppBackend<T>> tensor = Tensor<T, CppBackend<T>>(t.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = std::log(t.buffer_[i]);
    return tensor;
}

template <typename T>
Tensor<T, CppBackend<T>> CppBackend<T>::pow(const Tensor<T, CppBackend<T>> &t, const double exponent)
    requires std::is_arithmetic_v<T>
{
    Tensor<T, CppBackend<T>> tensor = Tensor<T, CppBackend<T>>(t.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = std::pow(t.buffer_[i], exponent);
    return tensor;
}

template <typename T>
Tensor<T, CppBackend<T>> CppBackend<T>::sqrt(const Tensor<T, CppBackend<T>> &t) requires std::is_arithmetic_v<T>
{
    Tensor<T, CppBackend<T>> tensor = Tensor<T, CppBackend<T>>(t.shape_);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        tensor.buffer_[i] = std::sqrt(t.buffer_[i]);
    return tensor;
}

template <typename T>
Tensor<T, CppBackend<T>> CppBackend<T>::dot(const Tensor<T, CppBackend<T>> &lhs, const Tensor<T, CppBackend<T>> &rhs)
    requires std::is_arithmetic_v<T>
{
    if (lhs.shape_.size() != 1)
        throw TensorInvalidShapeException(std::format("Dot product only applies for 1-dimensional tensors : {}.",
                                                      Tensor<T, CppBackend<T>>::tensorShapeToStr(lhs.shape_)));
    if (rhs.shape_.size() != 1)
        throw TensorInvalidShapeException(std::format("Dot product only applies for 1-dimensional tensors : {}.",
                                                      Tensor<T, CppBackend<T>>::tensorShapeToStr(rhs.shape_)));
    if (lhs.shape_[0] != rhs.shape_[0])
        throw TensorInvalidShapeException(std::format("Lengths are incompatible for dot product : {} and {}.",
                                                      Tensor<T, CppBackend<T>>::tensorShapeToStr(rhs.shape_),
                                                      Tensor<T, CppBackend<T>>::tensorShapeToStr(lhs.shape_)));

    Tensor<T, CppBackend<T>> tensor = Tensor<T, CppBackend<T>>({1});
    tensor.buffer_[0] = 0;
    for (std::size_t i = 0; i < lhs.shape_[0]; i++)
        tensor.buffer_[0] += lhs.buffer_[i] * rhs.buffer_[i];
    return tensor;
}

template <typename T>
Tensor<T, CppBackend<T>> CppBackend<T>::mm(const Tensor<T, CppBackend<T>> &lhs, const Tensor<T, CppBackend<T>> &rhs)
    requires std::is_arithmetic_v<T>
{
    // matrix multiplication (2x2 tensors)
    if (lhs.shape_.size() != 2)
        throw TensorInvalidShapeException(std::format("mm only applies for 2-dimensional tensors : {}.",
                                                      Tensor<T, CppBackend<T>>::tensorShapeToStr(lhs.shape_)));
    if (rhs.shape_.size() != 2)
        throw TensorInvalidShapeException(std::format("mm only applies for 2-dimensional tensors : {}.",
                                                      Tensor<T, CppBackend<T>>::tensorShapeToStr(rhs.shape_)));
    if (lhs.shape_[1] != rhs.shape_[0])
        throw TensorInvalidShapeException(
            std::format("Tensors are not compatible for matrix multiplication : {} and {}.",
                        Tensor<T, CppBackend<T>>::tensorShapeToStr(lhs.shape_),
                        Tensor<T, CppBackend<T>>::tensorShapeToStr(rhs.shape_)));

    Tensor<T, CppBackend<T>> tensor = Tensor<T, CppBackend<T>>({lhs.shape_[0], rhs.shape_[1]});
    tensor.fill(0);
    for (std::size_t y = 0; y < tensor.shape_[1]; y++)
        for (std::size_t x = 0; x < tensor.shape_[0]; x++)
            for (std::size_t k = 0; k < lhs.shape_[1]; k++)
                tensor.buffer_[x * tensor.shape_[1] + y] +=
                    lhs.buffer_[x * lhs.shape_[1] + k] * rhs.buffer_[k * rhs.shape_[1] + y];
    return tensor;
}

template <typename T>
Tensor<T, CppBackend<T>> CppBackend<T>::omm(const Tensor<T, CppBackend<T>> &lhs, const Tensor<T, CppBackend<T>> &rhs)
    requires std::is_arithmetic_v<T>
{
    // optimized matrix multiplication (2x2 tensors) with rhs transpose for
    // quicker data reading
    if (lhs.shape_.size() != 2)
        throw TensorInvalidShapeException(std::format("mm only applies for 2-dimensional tensors : {}.",
                                                      Tensor<T, CppBackend<T>>::tensorShapeToStr(lhs.shape_)));
    if (rhs.shape_.size() != 2)
        throw TensorInvalidShapeException(std::format("mm only applies for 2-dimensional tensors : {}.",
                                                      Tensor<T, CppBackend<T>>::tensorShapeToStr(rhs.shape_)));
    if (lhs.shape_[1] != rhs.shape_[0])
        throw TensorInvalidShapeException(
            std::format("Tensors are not compatible for matrix multiplication : {} and {}.",
                        Tensor<T, CppBackend<T>>::tensorShapeToStr(lhs.shape_),
                        Tensor<T, CppBackend<T>>::tensorShapeToStr(rhs.shape_)));

    Tensor<T, CppBackend<T>> tensor = Tensor<T, CppBackend<T>>({lhs.shape_[0], rhs.shape_[1]});
    Tensor<T, CppBackend<T>> rhs_t = rhs.transpose();
    for (std::size_t y = 0; y < lhs.shape_[0]; y++)
        for (std::size_t x = 0; x < rhs.shape_[1]; x++)
            for (std::size_t k = 0; k < lhs.shape_[1]; k++)
                tensor.buffer_[x + y * rhs.shape_[1]] +=
                    lhs.buffer_[k + y * lhs.shape_[1]] * rhs_t.buffer_[k + x * rhs_t.shape_[1]];

    return tensor;
}

template <typename T>
Tensor<T, CppBackend<T>> CppBackend<T>::mvm(const Tensor<T, CppBackend<T>> &lhs, const Tensor<T, CppBackend<T>> &rhs)
    requires std::is_arithmetic_v<T>
{
    // matrix vector multiplication
    if (lhs.shape_.size() != 2)
        throw TensorInvalidShapeException(std::format("mvm only applies for 2-dimensional tensors for lhs : {}.",
                                                      Tensor<T, CppBackend<T>>::tensorShapeToStr(lhs.shape_)));
    if (rhs.shape_.size() != 1)
        throw TensorInvalidShapeException(std::format("mvm only applies for 1-dimensional tensors for rhs : {}.",
                                                      Tensor<T, CppBackend<T>>::tensorShapeToStr(rhs.shape_)));
    if (lhs.shape_[1] != rhs.shape_[0])
        throw TensorInvalidShapeException(std::format("Tensors are not compatible for matrix-vector "
                                                      "multiplication : {} and {}.",
                                                      Tensor<T, CppBackend<T>>::tensorShapeToStr(lhs.shape_),
                                                      Tensor<T, CppBackend<T>>::tensorShapeToStr(rhs.shape_)));

    Tensor<T, CppBackend<T>> tensor = Tensor<T, CppBackend<T>>({lhs.shape_[0]});
    for (std::size_t y = 0; y < lhs.shape_[0]; y++)
        for (std::size_t x = 0; x < lhs.shape_[1]; x++)
            tensor.buffer_[y] += lhs.buffer_[x + y * lhs.shape_[1]] * rhs.buffer_[x];
    return tensor;
}

template <typename T>
Tensor<T, CppBackend<T>> CppBackend<T>::bmm(const Tensor<T, CppBackend<T>> &lhs, const Tensor<T, CppBackend<T>> &rhs)
    requires std::is_arithmetic_v<T>
{
    if (lhs.shape_.size() < 3)
        throw TensorInvalidShapeException(std::format("bmm only applies for 3 or + dimensional tensors (batched): {}.",
                                                      Tensor<T, CppBackend<T>>::tensorShapeToStr(lhs.shape_)));
    if (rhs.shape_.size() < 3)
        throw TensorInvalidShapeException(std::format("bmm only applies for 3 or + dimensional tensors (batched): {}.",
                                                      Tensor<T, CppBackend<T>>::tensorShapeToStr(rhs.shape_)));
    if (lhs.shape_[lhs.shape_.size() - 1] != rhs.shape_[rhs.shape_.size() - 2])
        throw TensorInvalidShapeException(std::format("Tensors are not compatible for batched matrix "
                                                      "multiplication : {} and {}.",
                                                      Tensor<T, CppBackend<T>>::tensorShapeToStr(lhs.shape_),
                                                      Tensor<T, CppBackend<T>>::tensorShapeToStr(rhs.shape_)));
    if (lhs.shape_.size() != rhs.shape_.size())
        throw TensorInvalidShapeException(std::format("Tensors dimensions are not compatible for batched matrix "
                                                      "multiplication (use broadcasting) : {} and {}.",
                                                      Tensor<T, CppBackend<T>>::tensorShapeToStr(lhs.shape_),
                                                      Tensor<T, CppBackend<T>>::tensorShapeToStr(rhs.shape_)));
    for (std::size_t i = 0; i < lhs.shape_.size() - 2; i++)
        if (lhs.shape_[i] != rhs.shape_[i])
            throw TensorInvalidShapeException(std::format("Tensors dimensions are not compatible for batched matrix "
                                                          "multiplication (use broadcasting) : {} and {}.",
                                                          Tensor<T, CppBackend<T>>::tensorShapeToStr(lhs.shape_),
                                                          Tensor<T, CppBackend<T>>::tensorShapeToStr(rhs.shape_)));

    std::vector<std::size_t> new_shape = std::vector(rhs.shape_.begin(), rhs.shape_.end() - 2);
    new_shape.push_back(*(lhs.shape_.end() - 2));
    new_shape.push_back(*(rhs.shape_.end() - 1));
    Tensor<T, CppBackend<T>> tensor = Tensor<T, CppBackend<T>>(new_shape);

    std::size_t p = *(tensor.shape_.end() - 1);
    for (std::size_t i = 0; i < tensor.numel(); i++)
        for (std::size_t k = 0; k < lhs.shape_[lhs.shape_.size() - 1]; k++)
            tensor.buffer_[i] += lhs.buffer_[k + (i / p) * lhs.shape_[lhs.shape_.size() - 1]] *
                                 rhs.buffer_[(i % p) + k * rhs.shape_[rhs.shape_.size() - 1]];
    return tensor;
}

template <typename T>
Tensor<T, CppBackend<T>> CppBackend<T>::matmul(const Tensor<T, CppBackend<T>> &lhs, const Tensor<T, CppBackend<T>> &rhs)
    requires std::is_arithmetic_v<T>
{
    if (lhs.shape_.size() == 1 && rhs.shape_.size() == 1)
        return Tensor<T, CppBackend<T>>::dot(lhs, rhs);
    if (lhs.shape_.size() == 2 && rhs.shape_.size() == 2)
        return Tensor<T, CppBackend<T>>::mm(lhs, rhs);
    if (lhs.shape_.size() == 1 && rhs.shape_.size() == 2)
        return Tensor<T, CppBackend<T>>::mm(Tensor<T, CppBackend<T>>(lhs).unsqueeze(0), rhs).squeeze(0);
    if (lhs.shape_.size() == 2 && rhs.shape_.size() == 1)
        return Tensor<T, CppBackend<T>>::mvm(lhs, rhs);

    bool nlhs_unsqueeze_flg = false;
    bool nrhs_unsqueeze_flg = false;

    std::vector<std::size_t> nlhs_shape = lhs.shape_;
    std::vector<std::size_t> nrhs_shape = rhs.shape_;
    Tensor<T, CppBackend<T>> nlhs = Tensor<T, CppBackend<T>>(nlhs_shape, lhs.buffer_);
    Tensor<T, CppBackend<T>> nrhs = Tensor<T, CppBackend<T>>(nrhs_shape, lhs.buffer_);

    if (nlhs.shape_.size() == 1)
    {
        nlhs = nlhs.unsqueeze(0);
        nlhs_unsqueeze_flg = true;
    }

    if (nrhs.shape_.size() == 1)
    {
        nrhs = nrhs.unsqueeze(0);
        nrhs_unsqueeze_flg = true;
    }

    nlhs = nlhs.batch_broadcast(nrhs);
    nrhs = nrhs.batch_broadcast(nlhs);

    Tensor<T, CppBackend<T>> tensor = Tensor<T, CppBackend<T>>::bmm(nlhs, nrhs);

    if (nlhs_unsqueeze_flg)
        tensor = tensor.unsqueeze(tensor.shape_.size() - 2);
    if (nrhs_unsqueeze_flg)
        tensor = tensor.unsqueeze(tensor.shape_.size() - 1);
    return tensor;
}