#include "dataset.hh"

#include <algorithm>
#include <chrono>
#include <iterator>
#include <random>

template <typename T>
std::pair<Dataset<T>, Dataset<T>> Dataset<T>::split(std::size_t split)
{
    std::vector<T> data_1 = std::vector<T>(this->data_.begin(), this->data_.begin() + split);
    std::vector<T> data_2 = std::vector<T>(this->data_.begin() + split, this->data_.end());
    return std::make_pair(Dataset<T>(data_1), Dataset<T>(data_2));
}

template <typename T>
void Dataset<T>::shuffle()
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(this->data_.begin(), this->data_.end(), std::default_random_engine(seed));
}

template <typename T>
std::size_t Dataset<T>::length()
{
    return this->data_.size();
}

template <typename T>
std::vector<T>::iterator Dataset<T>::begin()
{
    return this->data_.begin();
}

template <typename T>
std::vector<T>::iterator Dataset<T>::const_begin() const
{
    return this->data_.begin();
}

template <typename T>
std::vector<T>::iterator Dataset<T>::end()
{
    return this->data_.end();
}

template <typename T>
std::vector<T>::iterator Dataset<T>::const_end() const
{
    return this->data_.end();
}