#include "functional.hh"

#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <tuple>

bool testXORDataset(std::size_t num_epoch = 10, std::size_t num_elements = 600, std::size_t num_validation = 100, 
    float target = 99.f, float epsilon = 1.f)
{
    XorDataset dataset = XorDataset(num_elements);
    dataset.shuffle();
    auto &&[training, validation] = dataset.split(num_elements - num_validation);

    Linear<float> linear1 = Linear<float>(2, 4, HE, 0.1f);
    Linear<float> linear2 = Linear<float>(4, 1, XAVIER, 0.1f);
    ReLu<float> relu = ReLu<float>();
    Sigmoid<float> sigm = Sigmoid<float>();
    BinaryCrossEntropyLoss<float> bceloss = BinaryCrossEntropyLoss<float>();

    linear1.training = true;
    linear2.training = true;
    relu.training = true;
    sigm.training = true;

    for (std::size_t epoch = 0; epoch < num_epoch; epoch++)
    {
        for (auto &&[data, expected] : training)
        {
            Tensor<float> t = data.to_type<float>();
            t = linear1.forward(t);
            t = relu.forward(t);
            t = linear2.forward(t);
            t = sigm.forward(t);

            Tensor<float> t_loss = bceloss.forward(t, Tensor<float>({1, 1}, {(float)expected}));
            Tensor<float> d = bceloss.backward(t, Tensor<float>({1, 1}, {(float)expected}));
            
            d = sigm.backward(d);
            d = linear2.backward(d);
            d = relu.backward(d);
            d = linear1.backward(d);
        }
    }

    linear1.training = false;
    linear2.training = false;
    relu.training = false;
    sigm.training = false;

    std::size_t res = 0;
    for (auto &&[data, expected] : validation)
    {
        Tensor<float> t = data.to_type<float>();
        t = linear1.forward(t);
        t = relu.forward(t);
        t = linear2.forward(t);
        t = sigm.forward(t);
        res += (expected == (t.item() >= 0.5f ? 1 : 0)) ? 1 : 0;
    }

    float precision = res / float(num_validation) * 100;
    ASSERT(precision >= target - epsilon && precision <= target + epsilon,
           std::format("Precision criteria not met : {:.2f} != {} ± {}.", precision, target, epsilon),
           std::format("Precision criteria met : {:.2f}% precision.", precision));

    return true;
}
PARAMETRIZE_FUNCTIONAL(testXORDataset, 10, 1500, 1000, 99.f, 1.f)