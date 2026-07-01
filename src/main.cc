#include "dataset/dataset.hh"
#include "dataset/mnist.hxx"
#include "layer/activation.hxx"
#include "layer/block/block.hh"
#include "layer/layer.hh"
#include "layer/linear.hxx"
#include "layer/loss.hxx"
#include "mt/mt.hh"
#include "src/utils/progress.hh"
#include "tensor/backend/backend.hh"
#include "tensor/tensor.hh"
#include "tensor/tensor_io.hxx"
#include "tensor/tensor_serialize.hxx"
#include "tensor/tensor_utils.hxx"
#include "xor/xor.hh"

#include <CvPlot/cvplot.h>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <list>
#include <stdlib.h>
#include <tuple>
#include <unistd.h>

int main()
{
    auto identity = [](int x) { return std::pow(x, 2); };
    Tensor<float> tensor = Tensor<float>::from_function(identity, {100});
    Tensor<float>::plot({tensor}, "-");
}

// int main()
// {
//     std::size_t num_epoch = 1;

//     MNISTDataset mnist = MNISTDataset("./data/");
//     mnist.normalize();
//     auto &&[training_60k, validation_10k] = std::make_pair(mnist, mnist.validation());
//     training_60k.shuffle();
//     validation_10k.shuffle();

//     MultiLayerPerceptron<float> mlp = MultiLayerPerceptron<float>(
//         784,
//         10,
//         512,
//         0,
//         Initialization::HE,
//         Initialization::XAVIER,
//         0.01f
//     );
//     SoftmaxCrossEntropyLoss<float> sceloss = SoftmaxCrossEntropyLoss<float>();

//     mlp.training(true);

//     std::vector<float> losses = std::vector<float>();
//     for (std::size_t epoch = 0; epoch < num_epoch; epoch++)
//     {
//         ETAProgress progress = ETAProgress(60000);
//         float epoch_losses = 0;
//         for (auto &&[data, expected] : training_60k)
//         {
//             Tensor<float> t = data.to_type<float>().flatten().unsqueeze(0);
//             t = mlp.forward({t});

//             Tensor<float> expected_oh = Tensor<float>::one_hot(expected, {1, 10});
//             Tensor<float> t_loss = sceloss.forward({t, expected_oh});
//             epoch_losses += t_loss.sum().item();

//             Tensor<float> d = sceloss.backward({t, expected_oh});
//             d = mlp.backward({d});

//             progress.update(1);
//         }
//         losses.insert(losses.end(), epoch_losses / training_60k.length());
//         std::cout << epoch + 1 << "/" << num_epoch << " : " << epoch_losses / training_60k.length() << std::endl;
//     }
//     // MTFile<MultiLayerPerceptron<float>>::write("mlp-mnist-normalized.mt", mlp);

//     Tensor<float> t_losses = Tensor<float>(std::vector<std::size_t>{losses.size()}, losses);
//     std::cout << t_losses << std::endl;
//     // t_losses.plot();

//     mlp.training(false);

//     std::size_t res = 0;
//     for (auto &&[data, expected] : validation_10k)
//     {
//         Tensor<float> t = data.to_type<float>().flatten().unsqueeze(0);
//         t = mlp.forward({t});
//         res += (expected == t.argmax().item()) ? 1 : 0;
//     }

//     float precision = res / float(validation_10k.length()) * 100;
//     std::cout << precision << "%" << std::endl;
//     // t_losses.plot("-");
// }

// int main()
// {
//     std::size_t num_epoch = 1;

//     MNISTDataset mnist = MNISTDataset("./data/");
//     auto &&[training_60k, validation_10k] = std::make_pair(mnist, mnist.validation());
//     training_60k.shuffle();
//     validation_10k.shuffle();

//     Linear<float> linear1 = Linear<float>(784, 512, HE, 0.01f);
//     ReLu<float> relu = ReLu<float>();
//     Linear<float> linear2 = Linear<float>(512, 10, XAVIER, 0.01f);
//     SoftmaxCrossEntropyLoss<float> sceloss = SoftmaxCrossEntropyLoss<float>();

//     linear1.training(true);
//     relu.training(true);
//     linear2.training(true);

//     std::vector<float> losses = std::vector<float>();
//     for (std::size_t epoch = 0; epoch < num_epoch; epoch++)
//     {
//         ETAProgress progress = ETAProgress(60000);
//         float epoch_losses = 0;
//         for (auto &&[data, expected] : training_60k)
//         {
//             Tensor<float> t = data.to_type<float>().flatten().unsqueeze(0);
//             t = linear1.forward({t});
//             t = relu.forward({t});
//             t = linear2.forward({t});

//             Tensor<float> expected_oh = Tensor<float>::one_hot(expected, {1, 10});
//             Tensor<float> t_loss = sceloss.forward({t, expected_oh});
//             epoch_losses += t_loss.sum().item();

//             Tensor<float> d = sceloss.backward({t, expected_oh});

//             d = linear2.backward({d});
//             d = relu.backward({d});
//             d = linear1.backward({d});
//             progress.update(1);

//         }
//         losses.insert(losses.end(), epoch_losses / training_60k.length());
//         std::cout << epoch + 1 << "/" << num_epoch << " : " << epoch_losses / training_60k.length() << std::endl;
//     }

//     Tensor<float> t_losses = Tensor<float>(std::vector<std::size_t>{losses.size()}, losses);
//     std::cout << t_losses << std::endl;
//     // t_losses.plot();

//     linear1.training(false);
//     relu.training(false);
//     linear2.training(false);

//     std::size_t res = 0;
//     for (auto &&[data, expected] : validation_10k)
//     {
//         Tensor<float> t = data.to_type<float>().flatten().unsqueeze(0);
//         t = linear1.forward({t});
//         t = relu.forward({t});
//         t = linear2.forward({t});
//         res += (expected == t.argmax().item()) ? 1 : 0;
//     }

//     float precision = res / float(validation_10k.length()) * 100;
//     std::cout << precision << "%" << std::endl;

//     // MTFile<Linear<float>>::write("linear1.mt", linear1);
//     // MTFile<Linear<float>>::write("linear2.mt", linear2);

// // [██████████████████████████████████████████████████] 100% (60000/60000)  |  13ms/it  |  ETA 0ms  |  Completed in
// 13m53s
// // 1/1 : 0.196513
// // tensor(shape=(1); data=([0.196513]...); dtype=(float))
// // 96.73%

//     return 0;
// }
