#pragma once

#include <cstddef>
#include <vector>
#include <tuple>

template <typename T, typename U>
class Dataset
{
public:
    std::vector<std::tuple<T, U>> data_;

public:
    Dataset(std::vector<std::tuple<T, U>>);
};

#include "dataset.hxx"