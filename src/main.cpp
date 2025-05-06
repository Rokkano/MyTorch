#include <stdlib.h>
#include "tensor/tensor.hh"

int main()
{
    std::size_t size = 3 * 5 * 5;
    int *buffer = (int *)malloc(size * sizeof(int)); // 125
    for (std::size_t i = 0; i < size; i++)
        buffer[i] = i;

    Tensor<int> *tensor1 = new Tensor<int>({3, 5, 5}, buffer);
    Tensor<int> *tensor2 = Tensor<int>::affine(*tensor1, 2, 0);
    Tensor<bool> *tensor3 = (*tensor1 == *tensor1);

    std::cout << (*tensor3).none() << std::endl;

    delete tensor1;
    delete tensor2;
    delete tensor3;
    return 0;
}