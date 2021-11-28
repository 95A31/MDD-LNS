#pragma once

#include <cassert>
#include <algorithm>
#include <random>
#include <Utils/TypeAlias.h>
#include <Containers/Pair.cuh>
#include <Containers/LightVector.cuh>

namespace Algorithms
{
    template<typename T>
    __host__ __device__ inline T min(T a, T b);

    template<typename T>
    __host__ __device__ inline T max(T a, T b);

    template<typename T>
    __host__ __device__ inline T ceilIntDivision(T a, T b);

    template<typename T>
    __host__ __device__ inline void sortedInsert(T* element, LightVector<T>* vector);

    __host__ __device__ inline Pair<u32> getBeginEndIndices(u32 index, u32 threads, u32 elements);
}

template<typename T>
__host__ __device__
T Algorithms::min(T a, T b)
{
    assert(std::is_arithmetic_v<T>);
    return a <= b ? a : b;
}

template<typename T>
__host__ __device__
T Algorithms::max(T a, T b)
{
    assert(std::is_arithmetic_v<T>);
    return a >= b ? a : b;
}

template<typename T>
__host__ __device__
void Algorithms::sortedInsert(T* element, LightVector<T>* vector)
{
    if(not vector->isEmpty())
    {
        if(*element < *vector->back())
        {
            if (vector->isFull())
            {
                *vector->back() = *element;
            }
            else
            {
                vector->pushBack(element);
            }

            for(u32 i = vector->getSize() - 1; i > 0; i -= 1)
            {
                if(*vector->at(i) < *vector->at(i-1))
                {
                    T::swap(*vector->at(i), *vector->at(i-1));
                }
                else
                {
                    break;
                }
            }
        }
    }
    else
    {
        vector->pushBack(element);
    }
}

__host__ __device__
Pair<u32> Algorithms::getBeginEndIndices(u32 index, u32 threads, u32 elements)
{
    u32 const elementsPerThread = ceilIntDivision(elements, threads);
    u32 const beginIdx = elementsPerThread * index;
    u32 endIdx = Algorithms::min(elements, beginIdx + elementsPerThread);
    return Pair<u32>(beginIdx, endIdx);
}

template<typename T>
__host__ __device__
T Algorithms::ceilIntDivision(T a, T b)
{
    assert(std::is_unsigned_v<T>);
    return (a + b - 1) / b;
}
