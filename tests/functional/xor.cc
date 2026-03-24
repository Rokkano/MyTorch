#include "functional.hh"

#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <tuple>

bool testXORDataset(std::size_t num_elements = 10000, std::size_t num_validation = 1000, float target = 99.f,
                    float epsilon = 1.f)
{
    XorDataset dataset = XorDataset(num_elements);
    dataset.shuffle();
    auto &&[training, validation] = dataset.split(num_elements - num_validation);

    Linear<float> linear1 = Linear<float>(2, 4, UNIFORM, 0.1f);
    Linear<float> linear2 = Linear<float>(4, 1, UNIFORM, 0.1f);
    Sigmoid<float> sigm1 = Sigmoid<float>();
    Sigmoid<float> sigm2 = Sigmoid<float>();
    BinaryCrossEntropyLoss<float> mseloss = BinaryCrossEntropyLoss<float>();

    linear1.training = true;
    linear2.training = true;
    sigm1.training = true;
    sigm2.training = true;

    std::vector<float> losses = std::vector<float>();
    for (auto &&[data, expected] : training)
    {
        Tensor<float> t = data.to_type<float>();
        t = linear1.forward(t);
        t = sigm1.forward(t);
        t = linear2.forward(t);
        t = sigm2.forward(t);

        Tensor<float> t_loss = mseloss.forward(t, Tensor<float>({1, 1}, {(float)expected}));
        Tensor<float> d = mseloss.backward(t, Tensor<float>({1, 1}, {(float)expected}));

        d = sigm2.backward(d);
        d = linear2.backward(d);
        d = sigm1.backward(d);
        d = linear1.backward(d);
    }

    linear1.training = false;
    linear2.training = false;
    sigm1.training = false;
    sigm2.training = false;

    std::size_t res = 0;
    for (auto &&[data, expected] : validation)
    {
        Tensor<float> t = data.to_type<float>();
        t = linear1.forward(t);
        t = sigm1.forward(t);
        t = linear2.forward(t);
        t = sigm2.forward(t);
        res += (expected == (t.item() >= 0.5f ? 1 : 0)) ? 1 : 0;
    }

    float precision = res / float(num_validation) * 100;
    ASSERT(precision >= target - epsilon && precision <= target + epsilon,
           std::format("Precision criteria not met : {:.2f} != {} ± {}.", precision, target, epsilon),
           std::format("Precision criteria met : {:.2f}% precision.", precision));

    return true;
}
PARAMETRIZE_FUNCTIONAL(testXORDataset, 10000, 1000, 99.f, 1.f)
