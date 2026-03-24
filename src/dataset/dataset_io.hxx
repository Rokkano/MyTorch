#pragma once

#include "../tensor/tensor.hh"
#include "dataset.hh"

template <typename T>
std::string Dataset<T>::datasetDataToStr(std::vector<T> data)
{
    Tensor<T> temp = Tensor<T>({data.size()}, data);
    std::stringstream ss;
    ss << temp;
    return ss.str();
}

template <typename T>
std::string Dataset<T>::datasetToStr(std::vector<T> data, std::string name, std::size_t clip)
{
    std::string data_str = Dataset<T>::datasetDataToStr(data);
    if (data_str.length() > clip)
    {
        data_str.resize(clip);
        data_str.append("...");
    }
    return "dataset(name=" + name + "; data=(" + data_str + "))";
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const Dataset<T> &t)
{

    return os << Dataset<T>::datasetToStr(t.data_, t.name_);
}
