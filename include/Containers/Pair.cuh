#pragma once

#include <cstdio>
#include <type_traits>
#include <thrust/swap.h>

template<typename T>
class Pair
{
    // Members
    public:
    T first;
    T second;

    // Functions
    public:
    __host__ __device__ Pair(T const & first, T const & second);
    __host__ __device__ Pair<T>& operator=(Pair<T> const & other);
    __host__ __device__ void print(bool endLine = true) const;
    __host__ __device__ inline static void swap(Pair<T>& p0, Pair<T>& p1);
};

template<typename T>
__host__ __device__
Pair<T>::Pair(T const & first, T const & second) :
    first(first),
    second(second)
{}

template<typename T>
__host__ __device__
Pair<T>& Pair<T>::operator=(Pair<T> const & other)
{
    first = other.first;
    second = other.second;
    return *this;
}

template<typename T>
__host__ __device__
void Pair<T>::print(bool endLine) const
{
    if constexpr (std::is_integral_v<T>)
    {
        printf(endLine ? "(%d,%d)\n" : "(%d,%d)", static_cast<int>(first), static_cast<int>(second));
    }
}

template<typename T>
__host__ __device__
void Pair<T>::swap(Pair<T>& p0, Pair<T>& p1)
{
    thrust::swap(p0.first, p1.first);
    thrust::swap(p0.second, p1.second);
}