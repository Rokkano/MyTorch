#include "supervised.hh"

#include <ostream>

template <typename T, typename U>
std::ostream &operator<<(std::ostream &os, const SupervisedDatasetItem<T, U> &I)
{
    return os << "(label:" << I.label << ", data:" << I.sample << ")";
}

template <typename B>
LinearDataset<B>::LinearDataset(std::size_t num_samples, float a, float b, float min, float max)
{
    this->name_ = "Linear";
    this->data_ = std::vector<SupervisedDatasetItem<Tensor<float, B>, float>>();

    UniformDistribution dis = Random::getInstance().uniformDistribution(min, max);

    std::size_t i = 0;
    while (i < num_samples)
    {
        float x = dis();
        this->data_.push_back(
            SupervisedDatasetItem<Tensor<float, B>, float>{Tensor<float, B>::from_vector({x}, {1, 1}), a * x + b});
        i++;
    }
}

template <typename B>
NoisedLinearDataset<B>::NoisedLinearDataset(std::size_t num_samples, float a, float b, float noise, float min,
                                            float max)
{
    this->name_ = "Linear";
    this->data_ = std::vector<SupervisedDatasetItem<Tensor<float, B>, float>>();

    UniformDistribution xDis = Random::getInstance().uniformDistribution(min, max);
    UniformDistribution noiseDis = Random::getInstance().uniformDistribution(-noise, +noise);
    std::size_t i = 0;
    while (i < num_samples)
    {
        float x = xDis();
        this->data_.push_back(SupervisedDatasetItem<Tensor<float, B>, float>{Tensor<float, B>::from_vector({x}, {1, 1}),
                                                                             a * x + b + noiseDis()});
        i++;
    }
}