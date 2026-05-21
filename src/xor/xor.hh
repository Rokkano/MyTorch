#pragma once

#include "../dataset/supervised.hh"
#include "../tensor/tensor.hh"

template <template <typename> typename B>
class XorDataset : public SupervisedDataset<Tensor<int, B>, int>
{
public:
    XorDataset(std::size_t num_samples = 1024);
};

template <template <typename> typename B>
class OrDataset : public XorDataset<B>
{
public:
    OrDataset(std::size_t num_samples = 1024);
};