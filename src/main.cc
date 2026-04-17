// #include "dataset/dataset.hh"
// #include "dataset/mnist.hh"
// #include "layer/layer.hh"
// #include "layer/block/block.hh"
#include "tensor/tensor.hh"
// #include "xor/xor.hh"
// #include "mt/mt.hh"

#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <list>
#include <stdlib.h>
#include <tuple>

int main()
{
    // Eigen::MatrixXd mat = Eigen::MatrixXd(3, 2);
    // mat << 1, 2, 3, 4 , 5, 6;
    // std::cout << "Here is mat*mat:\n" << mat * mat.transpose() << std::endl;
    // constexpr bool backend_cpp = BACKEND_CPP;
    // constexpr bool backend_eigen = BACKEND_EIGEN;
    // std::cout << backend_cpp << std::endl;
    // std::cout << backend_eigen << std::endl;
}

// int main()
// {
//     std::size_t num_epoch = 10;

//     MNISTDataset mnist = MNISTDataset("./data/");
//     mnist.normalize();
//     auto &&[training_60k, validation_10k] = std::make_pair(mnist, mnist.validation());
//     training_60k.shuffle();
//     validation_10k.shuffle();

//     std::cout << HAVE_EIGEN << std::endl;

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

//     mlp.training(false);

//     std::vector<float> losses = std::vector<float>();
//     for (std::size_t epoch = 0; epoch < num_epoch; epoch++)
//     {
//         std::size_t i = 0;
//         float epoch_losses = 0;
//         for (auto &&[data, expected] : training_60k)
//         {
//             Tensor<float> t = data.to_type<float>()
//                 .flatten()
//                 .unsqueeze(0);
//             t = mlp.forward(t);

//             Tensor<float> expected_oh = Tensor<float>::one_hot(expected, {1, 10});
//             Tensor<float> t_loss = sceloss.forward(t, expected_oh);
//             epoch_losses += t_loss.sum().item();

//             Tensor<float> d = sceloss.backward(t, expected_oh);

//             // d = mlp.backward(d);
//             i += 1;
//             if (i % 1000 == 0)
//                 std::cout << i << "/60" << std::endl;
//         }
//         losses.insert(losses.end(), epoch_losses / training_60k.length());
//         std::cout << epoch + 1 << "/" << num_epoch << " : " << epoch_losses / training_60k.length() << std::endl;
//     }
//     MTFile<MultiLayerPerceptron<float>>::write("mlp-mnist-normalized.mt", mlp);

//     Tensor<float> t_losses = Tensor<float>(std::vector<std::size_t>{losses.size()}, losses);
//     std::cout << t_losses << std::endl;

// mlp.training(false);

// std::size_t res = 0;
// for (auto &&[data, expected] : validation_10k)
// {
//     Tensor<float> t = data.to_type<float>().flatten().unsqueeze(0);
//     t = mlp.forward(t);
//     res += (expected == t.argmax().item()) ? 1 : 0;
// }

// float precision = res / float(validation_10k.length()) * 100;
// std::cout << precision << "%" << std::endl;
// t_losses.plot("-");

// }

// int main()
// {
//     MNISTDataset mnist = MNISTDataset("./data/");
//     auto &&[_, validation_10k] = std::make_pair(mnist, mnist.validation());
//     validation_10k.shuffle();
//     validation_10k.normalize();

//     Linear<float> linear1 = MTFile<Linear<float>>::read("linear1.mt");
//     ReLu<float> relu = ReLu<float>();
//     Linear<float> linear2 = MTFile<Linear<float>>::read("linear2.mt");
//     SoftmaxCrossEntropyLoss<float> sceloss = SoftmaxCrossEntropyLoss<float>();

//     linear1.training = false;
//     relu.training = false;
//     linear2.training = false;

//     std::size_t res = 0;
//     for (auto mnistSample : validation_10k)
//     {
//         Tensor<float> t = mnistSample.sample.to_type<float>().flatten().unsqueeze(0);
//         t = linear1.forward(t);
//         t = relu.forward(t);
//         t = linear2.forward(t);
//         res += (mnistSample.label == t.argmax().item()) ? 1 : 0;
//     }

//     float precision = res / float(validation_10k.length()) * 100;
//     std::cout << precision << "%" << std::endl;

//     return 0;
// }

// int main()
// {
//     std::size_t num_epoch = 5;

//     MNISTDataset mnist = MNISTDataset("./data/");
//     auto &&[training_60k, validation_10k] = std::make_pair(mnist, mnist.validation());
//     training_60k.shuffle();
//     validation_10k.shuffle();

//     Linear<float> linear1 = Linear<float>(784, 512, HE, 0.01f);
//     ReLu<float> relu = ReLu<float>();
//     Linear<float> linear2 = Linear<float>(512, 10, XAVIER, 0.01f);
//     SoftmaxCrossEntropyLoss<float> sceloss = SoftmaxCrossEntropyLoss<float>();

//     linear1.training = true;
//     relu.training = true;
//     linear2.training = true;

//     std::vector<float> losses = std::vector<float>();
//     for (std::size_t epoch = 0; epoch < num_epoch; epoch++)
//     {
//         float epoch_losses = 0;
//         for (auto &&[data, expected] : training_60k)
//         {
//             Tensor<float> t = data.to_type<float>().flatten().unsqueeze(0);
//             t = linear1.forward(t);
//             t = relu.forward(t);
//             t = linear2.forward(t);

//             Tensor<float> expected_oh = Tensor<float>::one_hot(expected, {1, 10});
//             Tensor<float> t_loss = sceloss.forward(t, expected_oh);
//             epoch_losses += t_loss.sum().item();

//             Tensor<float> d = sceloss.backward(t, expected_oh);

//             d = linear2.backward(d);
//             d = relu.backward(d);
//             d = linear1.backward(d);
//         }
//         losses.insert(losses.end(), epoch_losses / training_60k.length());
//         std::cout << epoch + 1 << "/" << num_epoch << " : " << epoch_losses / training_60k.length() << std::endl;

//     }

//     Tensor<float> t_losses = Tensor<float>(std::vector<std::size_t>{losses.size()}, losses);
//     std::cout << t_losses << std::endl;
//     // t_losses.plot();

//     linear1.training = false;
//     relu.training = false;
//     linear2.training = false;

//     std::size_t res = 0;
//     for (auto &&[data, expected] : validation_10k)
//     {
//         Tensor<float> t = data.to_type<float>().flatten().unsqueeze(0);
//         t = linear1.forward(t);
//         t = relu.forward(t);
//         t = linear2.forward(t);
//         res += (expected == t.argmax().item()) ? 1 : 0;
//     }

//     float precision = res / float(validation_10k.length()) * 100;
//     std::cout << precision << "%" << std::endl;

//     MTFile<Linear<float>>::write("linear1.mt", linear1);
//     MTFile<Linear<float>>::write("linear2.mt", linear2);

//     return 0;
// }
