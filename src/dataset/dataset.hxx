#include "dataset.hh"

template <typename T, typename U>
Dataset<T, U>::Dataset(std::vector<std::tuple<T, U>> data)
{
    this->data_ = data;
}