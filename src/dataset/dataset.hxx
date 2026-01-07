#include "dataset.hh"

#include <algorithm>
#include <iterator>

template <typename T, typename U>
void Dataset<T, U>::shuffle()
{
    std::random_shuffle(this->data_.begin(), this->data_.end());
}

template <typename T, typename U>
std::pair<std::vector<std::tuple<T, U>>, std::vector<std::tuple<T, U>>> Dataset<T, U>::data_split(std::size_t split)
{
    std::vector<std::tuple<T, U>> data_1 = std::vector<std::tuple<T, U>>(this->data_.begin(), this->data_.begin() + split);
    std::vector<std::tuple<T, U>> data_2 = std::vector<std::tuple<T, U>>(this->data_.begin() + split, this->data_.end());
    return std::make_pair(data_1, data_2);
}

template <typename T, typename U>
std::vector<std::tuple<T, U>>::iterator Dataset<T, U>::begin()
{
    return this->data_.begin();
}

template <typename T, typename U>
std::vector<std::tuple<T, U>>::iterator Dataset<T, U>::const_begin() const
{
    return this->data_.begin();
}

template <typename T, typename U>
std::vector<std::tuple<T, U>>::iterator Dataset<T, U>::end()
{
    return this->data_.end();
}

template <typename T, typename U>
std::vector<std::tuple<T, U>>::iterator Dataset<T, U>::const_end() const
{
        return this->data_.end();
}