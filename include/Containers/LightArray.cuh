#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <type_traits>
#include <thrust/copy.h>
#include <thrust/swap.h>
#include <Utils/Memory.cuh>
#include <Utils/TypeAlias.h>

template<typename T>
class LightArray
{
    // Members
    protected:
    u32 capacity;
    T* storage;

    // Functions
    public:
    __host__ __device__ LightArray(u32 capacity, T* storage);
    __host__ __device__ ~LightArray();
    __host__ __device__ inline T* at(u32 index) const;
    __host__ __device__ inline T* begin() const;
    __host__ __device__ inline T* end() const;
    __host__ __device__ inline std::byte* endOfStorage() const;
    __host__ __device__ inline u32 getCapacity() const;
    __host__ __device__ inline u32 indexOf(T const * t) const;
    __host__ __device__ LightArray<T>& operator=(LightArray<T> const & other);
    __host__ __device__ inline T* operator[](u32 index) const;
    __host__ __device__ void print(bool endLine = true, bool reverse = false) const;
    __host__ __device__ inline static u32 sizeOfStorage(u32 capacity);
    __host__ __device__ inline static void swap(LightArray<T>& a0, LightArray<T>& a1);
    protected:
    __host__ __device__ void print(u32 beginIdx, u32 endIdx, bool endLine, bool reverse) const;

};

template<typename T>
__host__ __device__
LightArray<T>::LightArray(u32 capacity, T* storage) :
    capacity(capacity),
    storage(storage)
{}

template<typename T>
__host__ __device__
LightArray<T>::~LightArray()
{}

template<typename T>
__host__ __device__
T* LightArray<T>::at(u32 index) const
{
    assert(capacity > 0);
    assert(index < capacity);
    return storage + index;
}

template<typename T>
__host__ __device__
T* LightArray<T>::begin() const
{
    return storage;
}

template<typename T>
__host__ __device__
T* LightArray<T>::end() const
{
    return storage + capacity;
}

template<typename T>
__host__ __device__
std::byte* LightArray<T>::endOfStorage() const
{
    return reinterpret_cast<std::byte*>(storage + capacity);
}

template<typename T>
__host__ __device__
u32 LightArray<T>::getCapacity() const
{
    return capacity;
}

template<typename T>
__host__ __device__
u32 LightArray<T>::indexOf(T const * t) const
{
    T const * const begin = this->begin();
    assert(begin <= t);
    assert(t < end());
    return static_cast<u32>(t - begin);
}

template<typename T>
__host__ __device__
LightArray<T>& LightArray<T>::operator=(LightArray<T> const & other)
{
    capacity = other.capacity;
    storage = other.storage;
    return *this;
}

template<typename T>
__host__ __device__
T* LightArray<T>::operator[](u32 index) const
{
    return at(index);
}

template<typename T>
__host__ __device__
void LightArray<T>::print(bool endLine, bool reverse) const
{
    print(0, capacity, endLine, reverse);
}

template<typename T>
__host__ __device__
u32 LightArray<T>::sizeOfStorage(u32 capacity)
{
    return sizeof(T) * capacity;
}

template<typename T>
__host__ __device__
void LightArray<T>::swap(LightArray<T>& a0, LightArray<T>& a1)
{
    thrust::swap(a0.capacity, a1.capacity);
    thrust::swap(a0.storage, a1.storage);
}

template<typename T>
__host__ __device__
void LightArray<T>::print(u32 beginIdx, u32 endIdx, bool endLine, bool reverse) const
{
    if constexpr (std::is_integral_v<T>)
    {
        assert(beginIdx < endIdx);

        i32 begin = not reverse ? static_cast<i32>(beginIdx) : static_cast<i32>(endIdx) - 1;
        i32 end = not reverse ? static_cast<i32>(endIdx) : static_cast<i32>(beginIdx) - 1;
        i32 step = not reverse ? 1 : -1;

        printf("[");
        {
            if (beginIdx != endIdx)
            {
                printf("%d", static_cast<int>(*at(begin)));
                for (i32 index = begin + step; index != end; index += step)
                {
                    printf(",");
                    printf("%d", static_cast<int>(*at(index)));
                }
            }
            printf(endLine ? "]\n" : "]");
        }
    }
}