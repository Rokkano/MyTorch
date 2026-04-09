#pragma once

#include "supervised.hh"

inline float MNIST_MEAN = 0.1307f;
inline float MNIST_STD = 0.3081f;

class MNISTSample : public SupervisedDatasetItem<Tensor<float>, unsigned char>
{
public:
    void show(OpenCVWindowOpts opts = {}, bool normalized = false);
};

class MNISTDataset : public Dataset<MNISTSample>
{
private:
    bool normalized_ = false;
    std::vector<MNISTSample> data_validation_;

public:
    MNISTDataset(std::string mnistDirectoryPath);
    MNISTDataset(std::vector<MNISTSample> data);

    MNISTDataset validation();
    void normalize();
};