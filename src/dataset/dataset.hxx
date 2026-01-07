#include "dataset.hh"

#include <algorithm>

template <typename T, typename U>
void Dataset<T, U>::shuffle()
{
    std::random_shuffle(this->data_.begin(), this->data_.end());
}
