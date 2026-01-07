#include "../tensor/tensor.hh"
#include "../dataset/dataset.hh"
#include "xor.hh"

XorDataset::XorDataset(std::vector<std::tuple<Tensor<int>, int>> data)
{
    this->name_ = "Xor";
    this->data_ = data;
}

XorDataset::XorDataset(std::size_t num_samples = 1024)
{
    this->name_ = "Xor";
    this->data_ = std::vector<std::tuple<Tensor<int>, int>>();
    
    std::size_t i = 0;
    int i0 = 0;
    int i1 = 0; // TODO : shuffle instead of that
    while (i < num_samples)
    {
        i0 = (i0 + i % 2) % 2;
        i1 = (i1 + (i + 1) % 2) % 2;
        this->data_.push_back(std::tuple<Tensor<int>, int>{Tensor<int>::from_vector({i0, i1}, {2}), i0 != i1});
        i++;
    }
}

std::pair<XorDataset, XorDataset> XorDataset::split(std::size_t split)
{
    auto&&[data1, data2] = this->data_split(split);
    return std::make_pair(XorDataset(data1), XorDataset(data2));
}
