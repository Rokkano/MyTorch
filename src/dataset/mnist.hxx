#include "mnist.hh"
#include "src/cv/cv.hh"

#include <mnist/mnist_reader.hpp>
#include <mnist/mnist_utils.hpp>

template <typename B>
MNISTDataset<B>::MNISTDataset(std::string mnistDirectoryPath)
{
    this->name_ = "MNIST";
    this->data_ = std::vector<MNISTSample<B>>();

    auto dataset = mnist::read_dataset<std::vector, std::vector, float, uint8_t>(mnistDirectoryPath);

    for (std::size_t i = 0; i < dataset.training_images.size(); i++)
    {
        std::vector<float> buffer = dataset.training_images[i];
        std::vector<std::size_t> shape = std::vector<std::size_t>{28, 28};
        unsigned char label = dataset.training_labels[i];

        this->data_.push_back(MNISTSample<B>{
            Tensor<float>(shape, buffer) / 255.f,
            label,
        });
    }

    for (std::size_t i = 0; i < dataset.test_images.size(); i++)
    {
        std::vector<float> buffer = dataset.test_images[i];
        std::vector<std::size_t> shape = std::vector<std::size_t>{28, 28};
        unsigned char label = dataset.test_labels[i];

        this->data_validation_.push_back(MNISTSample{
            Tensor<float>(shape, buffer) / 255.f,
            label,
        });
    }
}

template <typename B>
MNISTDataset<B>::MNISTDataset(std::vector<MNISTSample<B>> data)
{
    this->name_ = "MNIST";
    this->data_ = data;
}

template <typename B>
MNISTDataset<B> MNISTDataset<B>::validation()
{
    if (this->data_validation_.size() == 0)
        throw Exception("Cannot build another validation set from a validation set.");
    return MNISTDataset<B>(this->data_validation_);
}

template <typename B>
void MNISTDataset<B>::normalize()
{
    this->transform(
        [](MNISTSample<B> var)
        {
            var.sample = (var.sample - MNIST_MEAN) / MNIST_STD;
            return var;
        });
    this->normalized_ = true;
}

template <typename B>
void MNISTSample<B>::show(OpenCVWindowOpts opts, bool normalized)
{
    cv::Mat image = cv::Mat(28, 28, CV_8UC1);

    for (int j = 0; j < image.rows; j++)
        for (int i = 0; i < image.cols; i++)
            image.at<uchar>(j, i) = std::round(
                (this->sample[i + j * 28] * (normalized ? MNIST_STD : 1) + (normalized ? MNIST_MEAN : 0)) * 255);

    CvPlot::Axes axes = CvPlot::plotImage(image);
    ::show(axes, opts);
}
