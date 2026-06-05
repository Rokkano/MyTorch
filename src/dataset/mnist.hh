#pragma once

#include "supervised.hh"

inline float MNIST_MEAN = 0.1307f;
inline float MNIST_STD = 0.3081f;

template <template <typename> typename B>
requires IsBackend<float, B>
class MNISTSample : public SupervisedDatasetItem<Tensor<float, B>, unsigned char>
{
public:
    void show(OpenCVWindowOpts opts = {}, bool normalized = false);
};

template <template <typename> typename B>
requires IsBackend<float, B>
class MNISTDataset : public Dataset<MNISTSample<B>>
{
private:
    bool normalized_ = false;
    std::vector<MNISTSample<B>> data_validation_;

public:
    MNISTDataset(std::string mnistDirectoryPath);
    MNISTDataset(std::vector<MNISTSample<B>> data);

    MNISTDataset validation();
    void normalize();
};