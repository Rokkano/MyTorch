#include <stdlib.h>
#include "tensor/tensor.hh"
#include "dataset/dataset.hh"
#include "xor/xor.hh"
#include "layer/layer.hh"
#include "exception/exception.hh"

#include <tuple>

int main()
{
    std::size_t num_elements = 10000;
    std::size_t num_validation = 1000;

    XorDataset dataset = XorDataset(num_elements);
    dataset.shuffle();
    auto&&[training, validation] = dataset.split(num_elements - num_validation);
    
    Perceptron perceptron = Perceptron(2);
    float learning_rate = 0.1;
    
    // Training
    for (auto&&[data, expected] : training)
    {
        Tensor<float> data_f = data.to_type<float>();
        float y = (perceptron.weights * data_f).sum().heaviside().item();
        perceptron.weights = perceptron.weights + data_f * (learning_rate * (expected - y));
    }

    // Validation
    std::size_t res = 0;
    for (auto&&[data, expected] : validation)
    {
        float y = (perceptron.weights * data.to_type<float>()).sum().heaviside().item();
        res += (y == expected) ? 1 : 0;
    }
    std::cout << ((float)(res)/num_validation) * 100 << "%" << std::endl;    
}

