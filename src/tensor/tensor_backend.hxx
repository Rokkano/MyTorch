#pragma once

#include "tensor.hh"

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> Tensor<T, B>::affine(const Tensor<T, B> &tensor, std::optional<T> a, std::optional<T> b)
requires std::is_arithmetic_v<T>
{
    return Tensor<T, B>(tensor.shape_, B<T>::affine(tensor.data_, tensor.shape_, a, b));
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> Tensor<T, B>::exp(const Tensor<T, B> &tensor) requires std::is_arithmetic_v<T>
{
    return Tensor<T, B>(tensor.shape_, B<T>::exp(tensor.data_, tensor.shape_));
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> Tensor<T, B>::log(const Tensor<T, B> &tensor) requires std::is_arithmetic_v<T>
{
    return Tensor<T, B>(tensor.shape_, B<T>::log(tensor.data_, tensor.shape_));
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> Tensor<T, B>::pow(const Tensor<T, B> &tensor, double exponent) requires std::is_arithmetic_v<T>
{
    return Tensor<T, B>(tensor.shape_, B<T>::pow(tensor.data_, tensor.shape_, exponent));
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> Tensor<T, B>::sqrt(const Tensor<T, B> &tensor) requires std::is_arithmetic_v<T>
{
    return Tensor<T, B>(tensor.shape_, B<T>::sqrt(tensor.data_, tensor.shape_));
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> Tensor<T, B>::dot(const Tensor<T, B> &lhs, const Tensor<T, B> &rhs) requires std::is_arithmetic_v<T>
{
    return Tensor<T, B>(std::vector<std::size_t>({1}), B<T>::dot(lhs.data_, lhs.shape_, rhs.data_, rhs.shape_));
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> Tensor<T, B>::mvm(const Tensor<T, B> &lhs, const Tensor<T, B> &rhs) requires std::is_arithmetic_v<T>
{

    return Tensor<T, B>({lhs.shape_[0]}, B<T>::mvm(lhs.data_, lhs.shape_, rhs.data_, rhs.shape_));
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> Tensor<T, B>::mm(const Tensor<T, B> &lhs, const Tensor<T, B> &rhs) requires std::is_arithmetic_v<T>
{
    return Tensor<T, B>({lhs.shape_[0], rhs.shape_[1]}, B<T>::mm(lhs.data_, lhs.shape_, rhs.data_, rhs.shape_));
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> Tensor<T, B>::bmm(const Tensor<T, B> &lhs, const Tensor<T, B> &rhs) requires std::is_arithmetic_v<T>
{
    return Tensor<T, B>({rhs.shape_[0], lhs.shape_[1], rhs.shape_[2]}, B<T>::bmm(lhs.data_, lhs.shape_, rhs.data_, rhs.shape_));
}

template <typename T, template <typename> typename B>
requires IsBackend<T, B>
Tensor<T, B> Tensor<T, B>::matmul(const Tensor<T, B> &lhs, const Tensor<T, B> &rhs) requires std::is_arithmetic_v<T>
{
    if (lhs.shape_.size() == 1 && rhs.shape_.size() == 1)
        return Tensor<T, B>::dot(lhs, rhs);
    if (lhs.shape_.size() == 2 && rhs.shape_.size() == 2)
        return Tensor<T, B>::mm(lhs, rhs);
    if (lhs.shape_.size() == 1 && rhs.shape_.size() == 2)
        return Tensor<T, B>::mm(Tensor<T, B>(lhs).unsqueeze(0), rhs).squeeze(0);
    if (lhs.shape_.size() == 2 && rhs.shape_.size() == 1)
        return Tensor<T, B>::mvm(lhs, rhs);

    bool nlhs_unsqueeze_flg = false;
    bool nrhs_unsqueeze_flg = false;

    std::vector<std::size_t> nlhs_shape = rhs.shape_;
    std::vector<std::size_t> nrhs_shape = rhs.shape_;
    Tensor<T, B> nlhs = Tensor<T, B>(nlhs_shape, lhs.data_);
    Tensor<T, B> nrhs = Tensor<T, B>(nrhs_shape, lhs.data_);

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

    Tensor<T, B> tensor = Tensor<T, B>::bmm(nlhs, nrhs);

    if (nlhs_unsqueeze_flg)
        tensor = tensor.unsqueeze(tensor.shape_.size() - 2);
    if (nrhs_unsqueeze_flg)
        tensor = tensor.unsqueeze(tensor.shape_.size() - 1);
    return tensor;
}
