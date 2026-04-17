#include "layer.hh"

template <typename T>
std::vector<std::byte> Layer<T>::serialize()
{
    return std::vector<std::byte>();
}

template <typename T>
std::size_t Layer<T>::deserialize(std::vector<std::byte> &bytes)
{
    (void)bytes;
    return 0;
}

template <typename T>
void Layer<T>::training(bool flag)
{
    this->training_ = flag;
}