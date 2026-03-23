#pragma once

#include "../tensor/tensor.hh"
#include "../dataset/dataset.hh"

class XorDataset : public SupervisedDataset<Tensor<int>, int>
{
public:
    XorDataset(std::size_t);
};

class OrDataset : public XorDataset
{
public:
    OrDataset(std::size_t);
};