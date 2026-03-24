
#include "supervised.hh"

#include "../tensor/tensor.hh"
#include "../utils/random.hh"

#include <ostream>
#include <vector>

LinearDataset::LinearDataset(std::size_t num_samples, float a, float b, float min, float max)
{
    this->name_ = "Linear";
    this->data_ = std::vector<SupervisedDatasetItem<Tensor<float>, float>>();

    UniformDistribution dis = Random::getInstance().uniformDistribution(min, max);

    std::size_t i = 0;
    while (i < num_samples)
    {
        float x = dis();
        this->data_.push_back(
            SupervisedDatasetItem<Tensor<float>, float>{Tensor<float>::from_vector({x}, {1, 1}), a * x + b});
        i++;
    }
}

NoisedLinearDataset::NoisedLinearDataset(std::size_t num_samples, float a, float b, float noise, float min, float max)
{
    this->name_ = "Linear";
    this->data_ = std::vector<SupervisedDatasetItem<Tensor<float>, float>>();

    UniformDistribution xDis = Random::getInstance().uniformDistribution(min, max);
    UniformDistribution noiseDis = Random::getInstance().uniformDistribution(-noise, +noise);
    std::size_t i = 0;
    while (i < num_samples)
    {
        float x = xDis();
        this->data_.push_back(SupervisedDatasetItem<Tensor<float>, float>{Tensor<float>::from_vector({x}, {1, 1}),
                                                                          a * x + b + noiseDis()});
        i++;
    }
}