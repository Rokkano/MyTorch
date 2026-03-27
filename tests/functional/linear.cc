#include "functional.hh"

#include <cmath>
#include <fstream>
#include <stdlib.h>
#include <tuple>

bool testLinearRegression(LinearDataset linearDataset, std::size_t num_elements = 1000,
                          std::size_t num_validation = 100, float target = 99.f, float epsilon = 1.f)
{
    linearDataset.shuffle();
    auto &&[training, validation] = linearDataset.split(num_elements - num_validation);

    Linear<float> linear = Linear<float>(1, 1, UNIFORM, 0.1f);
    MeanSquaredErrorLoss<float> mseloss = MeanSquaredErrorLoss<float>();

    linear.training = true;

    for (auto &&[data, expected] : training)
    {
        Tensor<float> t = data.to_type<float>();
        t = linear.forward(t);

        Tensor<float> t_loss = mseloss.forward(t, Tensor<float>({1, 1}, {(float)expected}));
        Tensor<float> d = mseloss.backward(t, Tensor<float>({1, 1}, {(float)expected}));

        d = linear.backward(d);
    }

    linear.training = false;

    float res = 0;
    for (auto &&[data, expected] : validation)
    {
        Tensor<float> t = data.to_type<float>();
        t = linear.forward(t);
        res += std::abs(expected - t.item());
    }

    float precision = 100 - res / float(num_validation) * 100;

    ASSERT(precision >= target - epsilon && precision <= target + epsilon,
           std::format("Precision criteria not met : {:.2f} != {} ± {}.", precision, target, epsilon),
           std::format("Precision criteria met : {:.2f}% precision.", precision));

    return true;
}
PARAMETRIZE_FUNCTIONAL(testLinearRegression, LinearDataset(1100, 2, 5, 0, 1), 1100, 100, 99.f, 1.f)
PARAMETRIZE_FUNCTIONAL(testLinearRegression, NoisedLinearDataset(1100, 2, 5, 0, 0, 1), 1100, 100, 99.f, 1.f)
PARAMETRIZE_FUNCTIONAL(testLinearRegression, NoisedLinearDataset(1100, 2, 5, 0.1, 0, 1), 1100, 100, 99.f, 1.f)
PARAMETRIZE_FUNCTIONAL(testLinearRegression, NoisedLinearDataset(1100, 2, 5, 0.5, 0, 1), 1100, 100, 99.f, 1.f)
PARAMETRIZE_FUNCTIONAL(testLinearRegression, NoisedLinearDataset(1100, 2, 5, 2., 0, 1), 1100, 100, 99.f, 1.f)
