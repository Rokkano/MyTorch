#pragma once

#include "../dataset/dataset.hh"
#include "../tensor/tensor.hh"

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