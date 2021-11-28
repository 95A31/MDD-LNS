#pragma once

#include <cstdio>
#include <type_traits>
#include <thrust/swap.h>

template<typename T>
class Triple
{
    // Members
    public:
    T first;
    T second;
    T third;

    // Functions
    public:
    __host__ __device__ Triple(T const & first, T const & second, T const & third);
    __host__ __device__ Triple<T>& operator=(Triple<T> const & other);
    __host__ __device__ void print(bool endLine = true) const;
    __host__ __device__ inline static void swap(Triple<T>& y0, Triple<T>& t1);
};

template<typename T>
__host__ __device__
Triple<T>::Triple(T const & first, T const & second, T const & third) :
    first(first),
    second(second),
    third(third)
{}

template<typename T>
__host__ __device__
Triple<T>& Triple<T>::operator=(Triple<T> const & other)
{
    first = other.first;
    second = other.second;
    third = other.third;
    return *this;
}

template<typename T>
__host__ __device__
void Triple<T>::print(bool endLine) const
{
    if constexpr (std::is_integral_v<T>)
    {
        printf(endLine ? "(%d,%d,%d)\n" : "(%d,%d,%d)", static_cast<int>(first), static_cast<int>(second), static_cast<int>(third));
    }
}

template<typename T>
__host__ __device__
void Triple<T>::swap(Triple<T>& t0, Triple<T>& t1)
{
    thrust::swap(t0.first, t1.first);
    thrust::swap(t0.second, t1.second);
    thrust::swap(t0.third, t1.third);
}