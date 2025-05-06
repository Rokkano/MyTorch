#include <stdlib.h>
#include "tensor/tensor.hh"

int main()
{
    std::size_t size = 3 * 5 * 5;
    int *buffer = (int *)malloc(size * sizeof(int)); // 125
    for (std::size_t i = 0; i < size; i++)
        buffer[i] = i;

    Tensor<int> *tensor1 = new Tensor<int>({3, 5, 5}, buffer);

    Tensor<int> *tensor2 = *tensor1 + 5;

    std::cout << +*tensor2 << std::endl;

    std::free(tensor1);
    std::free(tensor2);
    return 0;
}