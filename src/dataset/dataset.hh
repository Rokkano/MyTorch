#pragma once

#include "../utils/random.hh"

#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

template <typename T>
class Dataset
{

public:
    std::vector<T> data_;
    std::string name_ = "";

private:
    static std::string datasetDataToStr(std::vector<T>);
    static std::string datasetToStr(std::vector<T>, std::string, std::size_t = 1024);

public:
    void shuffle();
    std::pair<Dataset<T>, Dataset<T>> split(std::size_t);
    std::size_t length();

    std::vector<T>::iterator begin();
    std::vector<T>::iterator const_begin() const;
    std::vector<T>::iterator end();
    std::vector<T>::iterator const_end() const;

    T &operator[](std::size_t index) { return this->data_[index]; };

    template <typename V>
    friend std::ostream &operator<<(std::ostream &, const Dataset<V> &);
};

#include "dataset.hxx"
#include "dataset_io.hxx"
#include "supervised.hh"