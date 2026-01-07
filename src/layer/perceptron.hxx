#pragma once

#include "../tensor/tensor.hh"

class Perceptron 
{
    
public:
    Tensor<float> weights = Tensor<float>({});

public:
    Perceptron(std::size_t);
};

Perceptron::Perceptron(std::size_t parameters = 6)
{
    this->weights = Tensor<float>({parameters}, std::vector<float>(parameters, 0));
}