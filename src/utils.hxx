#pragma once

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <numeric>
#include <cxxabi.h>

template <typename T>
std::string reccursiveStr(std::vector<std::size_t> shape, T *buffer)
{
    if (shape.empty())
        return std::to_string(*buffer);
    std::size_t step = std::reduce(shape.begin() + 1, shape.end(), 1, std::multiplies<int>());
    std::string str = "[";
    for (std::size_t i = 0; i < shape[0]; i++)
        str += reccursiveBuild(std::vector<std::size_t>(shape.begin() + 1, shape.end()), buffer + i * step) + (i != shape[0] - 1 ? "," : "");
    str = str + "]";
    return str;
}

template <typename T>
std::string type_name()
{
    // https://stackoverflow.com/questions/1055452/c-get-name-of-type-in-template
    int status;
    std::string tname = typeid(T).name();
    char *demangled_name = abi::__cxa_demangle(tname.c_str(), NULL, NULL, &status);
    if (status == 0)
    {
        tname = demangled_name;
        std::free(demangled_name);
    }
    return tname;
}