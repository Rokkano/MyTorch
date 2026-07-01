#pragma once

#include "layer.hh"


template <typename T, template <typename> typename B = DefaultBackend>
class MeanSquaredErrorLoss : public Layer<T, B>
{
public:
    Tensor<T, B> forward(Tensor<T, B>::TensorSpan args)
    {
        if (args.size() != 2)
            throw TensorInvalidArgException("Forward argument does not match this layer.");
        const Tensor<T, B> &pred = args[0].get();
        const Tensor<T, B> &gt = args[1].get();

        Tensor<T, B> sum = Tensor<T, B>::pow(pred - gt, 2).sum();
        // 0.5 is for an easier gradient compute
        // See
        // https://aew61.github.io/blog/artificial_neural_networks/1_background/1.c_loss_functions_and_derivatives.html
        return 1.f / (2.f * (float)pred.numel()) * sum;
    }

    Tensor<T, B> backward(Tensor<T, B>::TensorSpan args) 
    {     
        if (args.size() != 2)
            throw TensorInvalidArgException("Backward argument does not match this layer.");
        const Tensor<T, B> &pred = args[0].get();    
        const Tensor<T, B> &gt = args[1].get();    

        return (pred - gt) / (float)pred.numel(); 
    }
};

template <typename T, template <typename> typename B = DefaultBackend>
using L2Loss = MeanSquaredErrorLoss<T, B>;

template <typename T, template <typename> typename B = DefaultBackend>
class SoftmaxCrossEntropyLoss : public Layer<T, B>
{
public:
    Tensor<T, B> forward(Tensor<T, B>::TensorSpan args)
    {
        if (args.size() != 2)
            throw TensorInvalidArgException("Forward argument does not match this layer.");
        const Tensor<T, B> &pred = args[0].get();
        const Tensor<T, B> &gt = args[1].get();

        Tensor<T, B> sm = Tensor<T, B>::softmax(pred);
        return -(gt * Tensor<T, B>::log(sm + 1.e-7f)).sum();
    }

    Tensor<T, B> backward(Tensor<T, B>::TensorSpan args) 
    {     
        if (args.size() != 2)
            throw TensorInvalidArgException("Backward argument does not match this layer.");
        const Tensor<T, B> &pred = args[0].get();    
        const Tensor<T, B> &gt = args[1].get();  
        
        return Tensor<T, B>::softmax(pred) - gt; 
    }
};

template <typename T, template <typename> typename B = DefaultBackend>
class BinaryCrossEntropyLoss : public Layer<T, B>
{
public:
    Tensor<T, B> forward(Tensor<T, B>::TensorSpan args)
    {
        if (args.size() != 2)
            throw TensorInvalidArgException("Forward argument does not match this layer.");
        const Tensor<T, B> &pred = args[0].get();
        const Tensor<T, B> &gt = args[1].get();

        return -(gt * Tensor<T, B>::log(pred) + (1.f - gt) * Tensor<T, B>::log(1.f - pred)).mean();
    }

    Tensor<T, B> backward(Tensor<T, B>::TensorSpan args) 
    {     
        if (args.size() != 2)
            throw TensorInvalidArgException("Backward argument does not match this layer.");
        const Tensor<T, B> &pred = args[0].get();    
        const Tensor<T, B> &gt = args[1].get();  

        return -(gt / pred - (1.f - gt) / (1.f - pred)); 
    }
};