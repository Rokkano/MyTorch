#pragma once

#include "dataset.hh"
#include "../tensor/tensor.hh"



template <typename T, typename U>
std::string Dataset<T, U>::datasetDataToStr(std::vector<std::tuple<T, U>> data)
{
    Tensor<std::tuple<T, U>> temp = Tensor<std::tuple<T, U>>({data.size()}, data);
    std::stringstream ss;
    ss << temp;
    return ss.str();
}

template <typename T, typename U>
std::string Dataset<T, U>::datasetToStr(std::vector<std::tuple<T, U>> data, std::string name, uint clip)
{
    std::string data_str = Dataset<T, U>::datasetDataToStr(data);
    if (data_str.length() > clip)
    {
        data_str.resize(clip);
        data_str.append("...");
    }
    return "dataset(name=" + name + "; data=(" + data_str + "))";
}



template <typename T, typename U>
std::ostream &operator<<(std::ostream &os, const Dataset<T, U> &t)
{

    return os << Dataset<T, U>::datasetToStr(t.data_, t.name_);
}

