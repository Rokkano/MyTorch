#include "xor.hh"

template <typename B>
XorDataset<B>::XorDataset(std::size_t num_samples = 1024)
{
    this->name_ = "Xor";
    this->data_ = std::vector<SupervisedDatasetItem<Tensor<int, B>, int>>();

    for (std::size_t i = 0; i < num_samples / 4; i++)
    {
        this->data_.push_back(
            SupervisedDatasetItem<Tensor<int, B>, int>{Tensor<int, B>::from_vector({0, 0}, {1, 2}), 0});
        this->data_.push_back(
            SupervisedDatasetItem<Tensor<int, B>, int>{Tensor<int, B>::from_vector({0, 1}, {1, 2}), 1});
        this->data_.push_back(
            SupervisedDatasetItem<Tensor<int, B>, int>{Tensor<int, B>::from_vector({1, 0}, {1, 2}), 1});
        this->data_.push_back(
            SupervisedDatasetItem<Tensor<int, B>, int>{Tensor<int, B>::from_vector({1, 1}, {1, 2}), 0});
    }
};

template <typename B>
OrDataset<B>::OrDataset(std::size_t num_samples = 1024)
{
    this->name_ = "Or";
    this->data_ = std::vector<SupervisedDatasetItem<Tensor<int, B>, int>>();

    for (std::size_t i = 0; i < num_samples / 4; i++)
    {
        this->data_.push_back(
            SupervisedDatasetItem<Tensor<int, B>, int>{Tensor<int, B>::from_vector({0, 0}, {1, 2}), 0});
        this->data_.push_back(
            SupervisedDatasetItem<Tensor<int, B>, int>{Tensor<int, B>::from_vector({0, 1}, {1, 2}), 1});
        this->data_.push_back(
            SupervisedDatasetItem<Tensor<int, B>, int>{Tensor<int, B>::from_vector({1, 0}, {1, 2}), 1});
        this->data_.push_back(
            SupervisedDatasetItem<Tensor<int, B>, int>{Tensor<int, B>::from_vector({1, 1}, {1, 2}), 1});
    }
}