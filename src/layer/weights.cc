#include "weights.hh"

#include "../utils/random.hh"

#include <cmath>
#include <random>
#include <string>

void initialize_weights(Tensor<float> &tensor, enum Initialization initialization)
{
    switch (initialization)
    {
    case Initialization::ZEROS:
        tensor.fill(0);
        break;
    case Initialization::ONES:
        tensor.fill(1);
        break;
    case Initialization::NORMAL:
    {
        NormalDistribution dis = Random::getInstance().normalDistribution(0.f, 1.f);
        for (std::size_t i = 0; i < tensor.numel(); i++)
            tensor[i] = dis();
        break;
    }
    case Initialization::UNIFORM:
    {
        UniformDistribution dis = Random::getInstance().uniformDistribution(-1.f, +1.f);
        for (std::size_t i = 0; i < tensor.numel(); i++)
            tensor[i] = dis();
        break;
    }
    case Initialization::LECUN:
    {
        if (tensor.shape().size() != 2)
            throw;
        std::size_t prevLayer = tensor.shape()[0]; // number of nodes in the previous layer
        float bound = std::sqrt(3 / prevLayer);
        UniformDistribution dis = Random::getInstance().uniformDistribution(-bound, +bound);
        for (std::size_t i = 0; i < tensor.numel(); i++)
            tensor[i] = dis();
        break;
    }
    case Initialization::XAVIER:
    {
        if (tensor.shape().size() != 2)
            throw;
        std::size_t prevLayer = tensor.shape()[0]; // number of nodes in the previous layer
        float bound = 1.f / std::sqrt(prevLayer);
        UniformDistribution dis = Random::getInstance().uniformDistribution(-bound, +bound);
        for (std::size_t i = 0; i < tensor.numel(); i++)
            tensor[i] = dis();
        break;
    }
    case Initialization::XAVIER_NORMALIZED:
    {
        if (tensor.shape().size() != 2)
            throw;
        std::size_t prevLayer = tensor.shape()[0]; // number of nodes in the previous layer
        std::size_t nextLayer = tensor.shape()[1]; // number of nodes in the next layer
        float bound = std::sqrt(6.f / (prevLayer + nextLayer));
        UniformDistribution dis = Random::getInstance().uniformDistribution(-bound, +bound);
        for (std::size_t i = 0; i < tensor.numel(); i++)
            tensor[i] = dis();
        break;
    }
    case Initialization::HE:
    {
        if (tensor.shape().size() != 2)
            throw;
        std::size_t prevLayer = tensor.shape()[0]; // number of nodes in the previous layer
        float std = std::sqrt(2.f / prevLayer);
        NormalDistribution dis = Random::getInstance().normalDistribution(0, std);
        for (std::size_t i = 0; i < tensor.numel(); i++)
            tensor[i] = dis();
        break;
    }
    default:
        throw Exception(std::format("Unknown initialization method."));
    }
}