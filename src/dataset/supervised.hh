#pragma once

#include "dataset.hh"

template <typename T, typename U>
struct SupervisedDatasetItem
{
    T sample;
    U label;

    template <typename V, typename W>
    friend std::ostream &operator<<(std::ostream &, const SupervisedDatasetItem<V, W> &);
};

template <typename T, typename U>
class SupervisedDataset : public Dataset<SupervisedDatasetItem<T, U>>
{
};


class LinearDataset : public SupervisedDataset<Tensor<float>, float>
{
public:
    LinearDataset(std::size_t num_samples = 1024, float a = 1, float b = 0, float min = 0, float max = 10);
};

class NoisedLinearDataset : public LinearDataset
{
public:
    NoisedLinearDataset(std::size_t num_samples = 1024, float a = 1, float b = 0, float noise = 0.1, float min = 0, float max = 10);
};

#include "supervised.hxx"
