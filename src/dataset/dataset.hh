#pragma once

#include <cstddef>
#include <vector>
#include <tuple>
#include <string>

template <typename T, typename U>
class Dataset
{

protected:
    std::vector<std::tuple<T, U>> data_;
    std::string name_ = "";

    std::pair<std::vector<std::tuple<T, U>>, std::vector<std::tuple<T, U>>> data_split(std::size_t split);

private:
    static std::string datasetDataToStr(std::vector<std::tuple<T, U>>);
    static std::string datasetToStr(std::vector<std::tuple<T, U>>, std::string, std::size_t = 1024);

public:
    void shuffle();

    template <typename V, typename W>
    friend std::ostream &operator<<(std::ostream &, const Dataset<V, W> &);
};

#include "dataset.hxx"
#include "dataset_io.hxx"