#include <chrono>
#include <iostream>
#include <thread>
#include <functional>
#include <source_location>

#include "../src/tensor/tensor.hh"

#include "../include/nameof.hpp"

using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::milliseconds;

template <typename F, typename... Args>
void benchmark(F &func, std::size_t num_run, Args &&...args)
{
    std::chrono::milliseconds ms_int_acc = std::chrono::milliseconds(0);
    for (std::size_t i = 0; i < num_run; i++)
    {
        auto t1 = high_resolution_clock::now();
        auto t = func(std::forward<Args>(args)...);
        auto t2 = high_resolution_clock::now();
        ms_int_acc += duration_cast<milliseconds>(t2 - t1);
    }
    std::cout << NAMEOF_FULL_TYPE_EXPR(func) << ": " << (ms_int_acc / num_run).count() << "ms" << std::endl;
}

int main()
{
    std::size_t num_run = 25;
    auto identity = [](int x)
    { return x; };
    Tensor<int> tensor1 = Tensor<int>::from_function(identity, {512, 1024});
    Tensor<int> tensor2 = Tensor<int>::from_function(identity, {1024, 256});

    benchmark(Tensor<int>::mm, num_run, tensor1, tensor2);
    benchmark(Tensor<int>::omm, num_run, tensor1, tensor2);
    return 0;
}