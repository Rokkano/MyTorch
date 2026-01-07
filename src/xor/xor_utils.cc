#include "../tensor/tensor.hh"
#include "../dataset/dataset.hh"
#include "xor.hh"

template<>
std::string Tensor<std::tuple<Tensor<int>, int>>::tensorDataToStr(const std::vector<std::size_t> &shape, const std::vector<std::tuple<Tensor<int>, int>> &buffer)
{
    // New tensorDataToStr to allow Xor data to be printed
    auto tuple_to_str = [](std::tuple<Tensor<int>, int> tuple)
    {
        Tensor<int> temp = std::get<0>(tuple);
        return Tensor<int>::tensorDataToStr(temp.shape_, temp.buffer_) + "|" + std::to_string(std::get<1>(tuple));
    };

    if (shape.empty())
        return tuple_to_str(buffer[0]);
    std::size_t step = std::reduce(shape.begin() + 1, shape.end(), 1, std::multiplies<int>());
    std::string str = "[";
    for (std::size_t i = 0; i < shape[0]; i++)
    {
        std::vector<std::size_t> new_shape = std::vector<std::size_t>(shape.begin() + 1, shape.end());
        std::vector<std::tuple<Tensor<int>, int>> new_buffer = std::vector<std::tuple<Tensor<int>, int>>(buffer.begin() + i * step, buffer.end());
        str += tensorDataToStr(new_shape, new_buffer) + (i != shape[0] - 1 ? "," : "");
    }
    str = str + "]";
    return str;
}
