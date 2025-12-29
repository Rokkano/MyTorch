#include <stdlib.h>
#include "tensor/tensor.hh"
// #include "dataset/dataset.hh"
// #include "xor/xor.hh"

#include "exception/exception.hh"

template <typename T>
using member_callable = Tensor<T> (Tensor<T>::*)() const;

template <typename T>
std::function<Tensor<T>(member_callable<T>)> make_caller(const Tensor<T> &tensor)
{
    return [tensor](member_callable<T> func)
    {
        return (tensor.*func)();
    };
}

int main()
{
    // auto identity = [](int x)
    // { return x; };
    // Tensor<int> tensor1 = Tensor<int>::from_function(identity, {1, 2, 3});
    // Tensor<int> tensor2 = Tensor<int>::from_function(identity, {1, 3, 7});

    // using bool_callable = std::function<Tensor<bool>(Tensor<bool> &)>;
    // using bool_callable = Tensor<bool> (Tensor<bool>::*)() const;
    // bool_callable func = &Tensor<bool>::all;
    Tensor<bool> tensor = Tensor<bool>({1}, {true});

    // std::invoke([tensor](bool_callable func)
    //             { return (tensor.*func)(); }, func);

    auto caller = make_caller<bool>(tensor);
    Tensor<bool> result = caller(&Tensor<bool>::all);

    // using bool_callable = Tensor<bool> (Tensor<bool>::*)() const;
    // bool_callable func = &Tensor<bool>::all;
    // Tensor<bool> tensor = Tensor<bool>({1}, {true});

    // std::invoke([tensor](bool_callable func)
    //             { return (tensor.*func)(); }, func);

    return 0;
}
