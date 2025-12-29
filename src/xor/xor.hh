#pragma once

#include "../tensor/tensor.hh"
#include "../dataset/dataset.hh"

class XorDataset : Dataset<Tensor<int>, int>
{
    XorDataset(std::size_t num_samples = 1024);
};
