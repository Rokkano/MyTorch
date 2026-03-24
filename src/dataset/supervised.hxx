#include "supervised.hh"

#include <ostream>

template <typename T, typename U>
std::ostream &operator<<(std::ostream &os, const SupervisedDatasetItem<T, U> &I)
{
    return os << "(label:" << I.label << ", data:" << I.sample << ")";
}