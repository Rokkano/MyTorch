#include "mt.hh"
#include "../tensor/tensor.hh"

//  Header :
//  mt version : 2 uint -> 1.0
//  shape_len: ulong
//  shape_size: ulong
//  shape_ptr: ulong
//  dtype_len: ulong 
//  dtype_size: ulong
//  dtype_ptr: ulong
//  buffer_len: ulong
//  buffer_size: ulong
//  buffer_ptr: ulong

// shape    : array<std::size_t>
// dtype    : array<char>
// buffer   : array<T>

template <typename T>
class MTFile<Tensor<T>> 
{
public:
    void write(Tensor<T> tensor) {
        std::vector<std::byte> buffer;

        auto write = [&](const T& val) {
            const auto* ptr = reinterpret_cast<const std::byte*>(&val);
            buffer.insert(buffer.end(), ptr, ptr + sizeof(T));
        };

        write(MTFILE_VERSION_MAJOR);
        write(MTFILE_VERSION_MINOR);

        write(tensor.shape().size());
        write(sizeof(std::size_t));
        write(MTFile<Tensor<T>>::shape_ptr(tensor));
        
        std::string dtype = type_name(T);
        write(dtype.size());
        write(sizeof(char));
        write(MTFile<Tensor<T>>::dtype_ptr(tensor));

        write(tensor.buffer_.size());
        write(sizeof(Tensor<T>));
        write(MTFile<Tensor<T>>::buffer_ptr(tensor));

        for (std::size_t s : tensor.shape())
            write(s);

        for (char c : dtype)
            write(c);

        for (T e : tensor.buffer_)
            write(e);

        return buffer;
    }

    static ulong shape_ptr(Tensor<T> tensor)
    {
        // Calculate shape ptr based on tensor info
        return 2 * sizeof(uint) + 9 * sizeof(ulong);
    }

    static ulong dtype_ptr(Tensor<T> tensor)
    {
        // Calculate dtype ptr based on tensor info
        return MTFile<Tensor<T>>::shape_ptr(tensor) + tensor.shape().size() * sizeof(std::size_t);
    }

    static ulong buffer_ptr(Tensor<T> tensor)
    {
        // Calculate buffer ptr based on tensor info
        return MTFile<Tensor<T>>::dtype_ptr(tensor) + type_name(T).size() * sizeof(char);
    }
};