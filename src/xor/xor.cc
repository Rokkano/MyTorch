#include "xor.hh"

XorDataset::XorDataset(std::size_t num_samples = 1024)
{
    this->name_ = "Xor";
    this->data_ = std::vector<SupervisedDatasetItem<Tensor<int>, int>>();

    for (std::size_t i = 0; i < num_samples / 4; i++)
    {
        this->data_.push_back(SupervisedDatasetItem<Tensor<int>, int>{Tensor<int>::from_vector({0, 0}, {1, 2}), 0});
        this->data_.push_back(SupervisedDatasetItem<Tensor<int>, int>{Tensor<int>::from_vector({0, 1}, {1, 2}), 1});
        this->data_.push_back(SupervisedDatasetItem<Tensor<int>, int>{Tensor<int>::from_vector({1, 0}, {1, 2}), 1});
        this->data_.push_back(SupervisedDatasetItem<Tensor<int>, int>{Tensor<int>::from_vector({1, 1}, {1, 2}), 0});
    }
};

OrDataset::OrDataset(std::size_t num_samples = 1024)
{
    this->name_ = "Or";
    this->data_ = std::vector<SupervisedDatasetItem<Tensor<int>, int>>();

    for (std::size_t i = 0; i < num_samples / 4; i++)
    {
        this->data_.push_back(SupervisedDatasetItem<Tensor<int>, int>{Tensor<int>::from_vector({0, 0}, {1, 2}), 0});
        this->data_.push_back(SupervisedDatasetItem<Tensor<int>, int>{Tensor<int>::from_vector({0, 1}, {1, 2}), 1});
        this->data_.push_back(SupervisedDatasetItem<Tensor<int>, int>{Tensor<int>::from_vector({1, 0}, {1, 2}), 1});
        this->data_.push_back(SupervisedDatasetItem<Tensor<int>, int>{Tensor<int>::from_vector({1, 1}, {1, 2}), 1});
    }
}