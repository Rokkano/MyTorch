#pragma once

#include "../tensor/tensor.hh"
#include "../dataset/dataset.hh"

class XorDataset : public Dataset<Tensor<int>, int>
{
public:
    XorDataset(std::vector<std::tuple<Tensor<int>, int>>);
    XorDataset(std::size_t);

    std::pair<XorDataset, XorDataset> split(std::size_t);
};

// tensorDataToStr specialization for XorDataset (tuple)
template<>
std::string Tensor<std::tuple<Tensor<int>, int>>::tensorDataToStr(const std::vector<std::size_t> &shape, const std::vector<std::tuple<Tensor<int>, int>> &buffer);