#include "dataset/dataset.hh"
#include "dataset/mnist.hxx"
#include "layer/layer.hh"
#include "layer/block/block.hh"
#include "tensor/tensor.hh"
#include "tensor/tensor_io.hxx"
#include "tensor/backend/backend.hh"
#include "tensor/tensor_utils.hxx"
#include "tensor/tensor_serialize.hxx"
#include "layer/linear.hxx"
#include "layer/activation.hxx"
#include "layer/loss.hxx"
#include "xor/xor.hh"
#include "mt/mt.hh"
#include "src/utils/progress.hh"


#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <list>
#include <stdlib.h>
#include <tuple>
#include <unistd.h>

// int main()
// {
//     std::size_t num_epoch = 10;

//     MNISTDataset<CppBackend> mnist = MNISTDataset<CppBackend>("./data/");
//     mnist.normalize();
//     auto &&[training_60k, validation_10k] = std::make_pair(mnist, mnist.validation());
//     auto &&[training_10k, _] = training_60k.split(10000);
//     training_10k.shuffle();
//     validation_10k.shuffle();

//     // std::cout << BACKEND_EIGEN << std::endl;
//     // std::cout << BACKEND_CPP << std::endl;

//     MultiLayerPerceptron<float, CppBackend> mlp = MultiLayerPerceptron<float, CppBackend>(
//         784,
//         10,
//         512,
//         0,
//         Initialization::HE,
//         Initialization::XAVIER,
//         0.01f
//     );
//     SoftmaxCrossEntropyLoss<float, CppBackend> sceloss = SoftmaxCrossEntropyLoss<float, CppBackend>();

//     mlp.training(true);

//     std::cout << mlp.layers_.size() << std::endl;
//     std::cout << mlp.activations_.size() << std::endl;
//     std::cout << Tensor<float, CppBackend>::shapeToStr(mlp.layers_[0].weights_.shape()) << std::endl;
//     std::cout << Tensor<float, CppBackend>::shapeToStr(mlp.layers_[1].weights_.shape()) << std::endl;

//     std::vector<float> losses = std::vector<float>();
//     for (std::size_t epoch = 0; epoch < num_epoch; epoch++)
//     {
//         std::size_t i = 0;
//         float epoch_losses = 0;
//         for (auto &&[data, expected] : training_10k)
//         {
//             Tensor<float, CppBackend> t = data.to_type<float>()
//                 .flatten()
//                 .unsqueeze(0);
//             t = mlp.forward(t);

//             Tensor<float, CppBackend> expected_oh = Tensor<float, CppBackend>::one_hot(expected, {1, 10});
//             Tensor<float, CppBackend> t_loss = sceloss.forward(t, expected_oh);
//             epoch_losses += t_loss.sum().item();

//             Tensor<float, CppBackend> d = sceloss.backward(t, expected_oh);
//             std::cout << d << std::endl;
//             d = mlp.backward(d);
//             i += 1;
//             if (i % 100 == 0)
//                 std::cout << i << "/10000" << std::endl;
//         }
//         losses.insert(losses.end(), epoch_losses / training_10k.length());
//         std::cout << epoch + 1 << "/" << num_epoch << " : " << epoch_losses / training_10k.length() << std::endl;
//     }
//     MTFile<MultiLayerPerceptron<float, CppBackend>>::write("mlp-mnist-normalized.mt", mlp);

//     Tensor<float, CppBackend> t_losses = Tensor<float, CppBackend>(std::vector<std::size_t>{losses.size()}, losses);
//     std::cout << t_losses << std::endl;

//     mlp.training(false);

//     std::size_t res = 0;
//     for (auto &&[data, expected] : validation_10k)
//     {
//         Tensor<float, CppBackend> t = data.to_type<float>().flatten().unsqueeze(0);
//         t = mlp.forward(t);
//         res += (expected == t.argmax().item()) ? 1 : 0;
//     }

//     float precision = res / float(validation_10k.length()) * 100;
//     std::cout << precision << "%" << std::endl;
//     // t_losses.plot("-");

// }

// int main()
// {
//     MNISTDataset<CppBackend> mnist = MNISTDataset<CppBackend>("./data/");
//     auto &&[_, validation_10k] = std::make_pair(mnist, mnist.validation());
//     validation_10k.shuffle();
//     validation_10k.normalize();

//     Linear<float, CppBackend> linear1 = MTFile<Linear<float, CppBackend>>::read("./data/linear1.mt");
//     ReLu<float, CppBackend> relu = ReLu<float, CppBackend>();
//     Linear<float, CppBackend> linear2 = MTFile<Linear<float, CppBackend>>::read("./data/linear2.mt");
//     SoftmaxCrossEntropyLoss<float, CppBackend> sceloss = SoftmaxCrossEntropyLoss<float, CppBackend>();

//     linear1.training(false);
//     relu.training(false);
//     linear2.training(false);

//     std::size_t res = 0;
//     for (auto mnistSample : validation_10k)
//     {
//         Tensor<float, CppBackend> t = mnistSample.sample.to_type<float>().flatten().unsqueeze(0);
//         t = linear1.forward(t);
//         t = relu.forward(t);
//         t = linear2.forward(t);
//         res += (mnistSample.label == t.argmax().item()) ? 1 : 0;
//     }

//     float precision = res / float(validation_10k.length()) * 100;
//     std::cout << precision << "%" << std::endl;

//     return 0;
// }

int main()
{
    std::size_t num_epoch = 1;

    MNISTDataset<CppBackend> mnist = MNISTDataset<CppBackend>("./data/");
    auto &&[training_60k, validation_10k] = std::make_pair(mnist, mnist.validation());
    auto &&[training_10k, _] = training_60k.split(10000);
    training_10k.shuffle();
    validation_10k.shuffle();

    Linear<float, CppBackend> linear1 = Linear<float, CppBackend>(784, 512, HE, 0.01f);
    ReLu<float, CppBackend> relu = ReLu<float, CppBackend>();
    Linear<float, CppBackend> linear2 = Linear<float, CppBackend>(512, 10, XAVIER, 0.01f);
    SoftmaxCrossEntropyLoss<float, CppBackend> sceloss = SoftmaxCrossEntropyLoss<float, CppBackend>();

    linear1.training(true);
    relu.training(true);
    linear2.training(true);

    std::vector<float> losses = std::vector<float>();
    for (std::size_t epoch = 0; epoch < num_epoch; epoch++)
    {
        ETAProgress progress = ETAProgress(10000);
        float epoch_losses = 0;
        for (auto &&[data, expected] : training_10k)
        {
            Tensor<float, CppBackend> t = data.to_type<float>().flatten().unsqueeze(0);
            t = linear1.forward(t);
            t = relu.forward(t);
            t = linear2.forward(t);

            Tensor<float, CppBackend> expected_oh = Tensor<float, CppBackend>::one_hot(expected, {1, 10});
            Tensor<float, CppBackend> t_loss = sceloss.forward(t, expected_oh);
            epoch_losses += t_loss.sum().item();

            Tensor<float, CppBackend> d = sceloss.backward(t, expected_oh);

            d = linear2.backward(d);
            d = relu.backward(d);
            d = linear1.backward(d);
            progress.update(1);

        }
        losses.insert(losses.end(), epoch_losses / training_10k.length());
        std::cout << epoch + 1 << "/" << num_epoch << " : " << epoch_losses / training_10k.length() << std::endl;
    }

    Tensor<float, CppBackend> t_losses = Tensor<float, CppBackend>(std::vector<std::size_t>{losses.size()}, losses);
    std::cout << t_losses << std::endl;
    // t_losses.plot();

    linear1.training(false);
    relu.training(false);
    linear2.training(false);

    std::size_t res = 0;
    for (auto &&[data, expected] : validation_10k)
    {
        Tensor<float, CppBackend> t = data.to_type<float>().flatten().unsqueeze(0);
        t = linear1.forward(t);
        t = relu.forward(t);
        t = linear2.forward(t);
        res += (expected == t.argmax().item()) ? 1 : 0;
    }

    float precision = res / float(validation_10k.length()) * 100;
    std::cout << precision << "%" << std::endl;

    MTFile<Linear<float, CppBackend>>::write("linear1.mt", linear1);
    MTFile<Linear<float, CppBackend>>::write("linear2.mt", linear2);

    return 0;
}
