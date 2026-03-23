#include <stdlib.h>
#include <fstream>
#include <cmath>
#include <tuple>

#include "functional.hh"

bool testORDataset(std::size_t num_elements = 10000, std::size_t num_validation = 1000, float target = 99.f, float epsilon = 1.f)
{
    OrDataset dataset = OrDataset(num_elements);
    dataset.shuffle();
    auto&&[training, validation] = dataset.split(num_elements - num_validation);
    
    Linear<float> linear = Linear<float>(2, 1, UNIFORM, 0.1f);
    Sigmoid<float> sigm = Sigmoid<float>();
    MeanSquaredErrorLoss<float> mseloss = MeanSquaredErrorLoss<float>();

    linear.training = true;
    sigm.training = true;
    
    for (auto&&[data, expected] : training)
    {
        Tensor<float> t = data.to_type<float>();
        t = linear.forward(t);
        t = sigm.forward(t);
        
        Tensor<float> t_loss = mseloss.forward(t, Tensor<float>({1, 1}, {(float)expected}));
        Tensor<float> d = mseloss.backward(t, Tensor<float>({1, 1}, {(float)expected}));

        d = sigm.backward(d);
        d = linear.backward(d);
    }
    
    linear.training = false;
    sigm.training = false;

    std::size_t res = 0;
    for (auto&&[data, expected] : validation)
    {
        Tensor<float> t = data.to_type<float>();
        t = linear.forward(t);
        t = sigm.forward(t);
        res += (expected == (t.item() >= 0.5f ? 1 : 0)) ? 1 : 0;
    }

    float precision = res / float(num_validation) * 100;

    ASSERT(
        precision >= target - epsilon && precision <= target + epsilon, 
        std::format("Precision criteria not met : {:.2f} != {} ± {}.", precision, target, epsilon),
        std::format("Precision criteria met : {:.2f}% precision.", precision)
    );

    return true;
}
PARAMETRIZE_FUNCTIONAL(testORDataset, 10000, 1000, 99.f, 1.f)
