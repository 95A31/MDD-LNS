#pragma once

#include <Containers/Vector.cuh>
#include <thrust/swap.h>
#include <thrust/functional.h>

template<typename T>
class LightMinMaxHeap
{
    // Members
    private:
    Vector<T*> vector;

    // Functions
    public:
    __host__ __device__ LightMinMaxHeap(u32 capacity, Memory::MallocType mallocType);
    __host__ __device__ inline T const * getMin() const;
    __host__ __device__ inline T const * getMax() const;
    __host__ __device__ void popMin();
    __host__ __device__ void popMax();
    __host__ __device__ void insert(T const * t);
    __host__ __device__ inline u32 getSize() const;
    __host__ __device__ inline bool isEmpty() const;
    __host__ __device__ inline bool isFull() const;
    __host__ __device__ inline void clear();
    private:
    __host__ __device__ u32 level(u32 index);
    __host__ __device__ i32 parent(i32 index);
    __host__ __device__ u32 leftChild(u32 index);
    __host__ __device__ u32 rightChild(u32 index);
    __host__ __device__ void pushDown(u32 index);
    __host__ __device__ void pushDownMin(u32 index);
    __host__ __device__ void pushDownMax(u32 index);
    __host__ __device__ void pushUp(u32 index);
    __host__ __device__ void pushUpMin(u32 index);
    __host__ __device__ void pushUpMax(u32 index);
};

template<typename T>
__host__ __device__
LightMinMaxHeap<T>::LightMinMaxHeap(u32 capacity, Memory::MallocType mallocType) :
    vector(capacity, mallocType)
{}

template<typename T>
__host__ __device__
u32 LightMinMaxHeap<T>::level(u32 index)
{
    double result = log2(static_cast<double>(index + 1));
    return static_cast<u32>(result);
}

template<typename T>
__host__ __device__
i32 LightMinMaxHeap<T>::parent(i32 index)
{

    return (index == 0) ? -1 : static_cast<i32>((index - 1) / 2);
}


template<typename T>
__host__ __device__
u32 LightMinMaxHeap<T>::leftChild(u32 index)
{
    return 2 * index + 1;
}

template<typename T>
__host__ __device__
u32 LightMinMaxHeap<T>::rightChild(u32 index)
{
    return 2 * index + 2;
}

template<typename T>
__host__ __device__
void LightMinMaxHeap<T>::pushDown(u32 index)
{
    if (level(index) % 2 == 0)
    {
        pushDownMin(index);
    }
    else
    {
        pushDownMax(index);
    }
}

template<typename T>
__host__ __device__
void LightMinMaxHeap<T>::pushDownMin(u32 index)
{
    u32 lc = leftChild(index);
    u32 rc = rightChild(index);
    u32 directChildren[2] = {lc, rc};
    u32 grandChildren[4] = {leftChild(lc), rightChild(lc), leftChild(rc), rightChild(rc)};
    u32 minIndex = index;
    char smallestChild = 'p';
    for(u32 dcIdx = 0; dcIdx < 2; dcIdx += 1)
    {
        u32 const i  = directChildren[dcIdx];
        if(i < vector.getSize() and **vector[i] < **vector[minIndex])
        {
            minIndex = i;
            smallestChild = 'd';
        }
    }
    for(u32 gcIdx = 0; gcIdx < 4; gcIdx += 1)
    {
        u32 const i  = grandChildren[gcIdx];
        if(i < vector.getSize() and **vector[i] < **vector[minIndex]) {
            minIndex = i;
            smallestChild = 'g';
        }
    }
    if(minIndex == index)
    {
        return;
    }
    else
    {
        thrust::swap(*vector[minIndex], *vector[index]);
        if(smallestChild == 'g')
        {
            if(**vector[parent(minIndex)] < **vector[minIndex])
            {
                thrust::swap(*vector[minIndex], *vector[parent(minIndex)]);
            }
            pushDownMin(minIndex);
        }
    }
}

template<typename T>
__host__ __device__
void LightMinMaxHeap<T>::pushDownMax(u32 index)
{
    u32 lc = leftChild(index);
    u32 rc = rightChild(index);
    u32 directChildren[2] = {lc, rc};
    u32 grandChildren[4] = {leftChild(lc), rightChild(lc), leftChild(rc), rightChild(rc)};
    u32 maxIndex = index;
    char biggestChild = 'p';
    for(u32 dcIdx = 0; dcIdx < 2; dcIdx += 1)
    {
        u32 const i  = directChildren[dcIdx];
        if(i < vector.getSize() and **vector[maxIndex] < **vector[i])
        {
            maxIndex = i;
            biggestChild = 'd';
        }
    }
    for(u32 gcIdx = 0; gcIdx < 4; gcIdx += 1)
    {
        u32 const i = grandChildren[gcIdx];
        if(i < vector.getSize() and  **vector[maxIndex] < **vector[i])
        {
            maxIndex = i;
            biggestChild = 'g';
        }
    }
    if(maxIndex == index)
    {
        return;
    }
    else
    {
        thrust::swap(*vector[maxIndex], *vector[index]);
        if(biggestChild == 'g')
        {
            if(**vector[maxIndex] < **vector[parent(maxIndex)])
            {
                thrust::swap(*vector[maxIndex], *vector[parent(maxIndex)]);
            }
            pushDownMax(maxIndex);
        }
    }
}

template<typename T>
__host__ __device__
void LightMinMaxHeap<T>::pushUp(u32 index)
{
    if(index == 0)
    {
        return;
    }
    if (level(index) % 2 == 0)
    {
        if(**vector[parent(index)] < **vector[index])
        {
            thrust::swap(*vector[index], *vector[parent(index)]);
            pushUpMax(parent(index));
        }
        else
        {
            pushUpMin(index);
        }
    } else {
        if(**vector[index] < **vector[parent(index)])
        {
            thrust::swap(*vector[index], *vector[parent(index)]);
            pushUpMin(parent(index));
        }
        else
        {
            pushUpMax(index);
        }
    }
}

template<typename T>
__host__ __device__
void LightMinMaxHeap<T>::pushUpMin(u32 index)
{
    i32 grandParent = parent(parent(index));
    if (grandParent >= 0 and **vector[index] < **vector[grandParent])
    {
        thrust::swap(*vector[index], *vector[grandParent]);
        pushUpMin(grandParent);
    }
}

template<typename T>
__host__ __device__
void LightMinMaxHeap<T>::pushUpMax(u32 index)
{
    i32 grandParent = parent(parent(index));
    if(grandParent >= 0 and **vector[grandParent] < **vector[index])
    {
        thrust::swap(*vector[index], *vector[grandParent]);
        pushUpMax(grandParent);
    }
}


template<typename T>
__host__ __device__
void LightMinMaxHeap<T>::insert(T const * t)
{
    vector.pushBack(const_cast<T* const *>(&t));
    pushUp(vector.getSize() - 1);
}

template<typename T>
__host__ __device__
void LightMinMaxHeap<T>::popMin()
{
    if (not vector.isEmpty())
    {
        *vector.front() = *vector.back();
        vector.popBack();
        pushDown(0);
    }
}

template<typename T>
__host__ __device__
void LightMinMaxHeap<T>::popMax()
{
    if(not vector.isEmpty())
    {
        if(vector.getSize() <= 2)
        {
            vector.popBack();
        }
        else
        {
            u32 maxIndex = (**vector[2] < **vector[1]) ? 1 : 2;
            *vector[maxIndex] = *vector.back();
            vector.popBack();
            pushDown(maxIndex);
        }
    }
}

template<typename T>
__host__ __device__
T const * LightMinMaxHeap<T>::getMin() const
{
    assert(not vector.isEmpty());
    return *vector[0];
}

template<typename T>
__host__ __device__
T const * LightMinMaxHeap<T>::getMax() const
{
    assert(not vector.isEmpty());

    if(vector.getSize() <= 2)
    {
        return *vector.back();
    }
    else
    {

        return **vector[1] < **vector[2] ? *vector[2] : *vector[1] ;
    }
}


template<typename T>
__host__ __device__
bool LightMinMaxHeap<T>::isEmpty() const
{
    return vector.isEmpty();
}

template<typename T>
__host__ __device__
bool LightMinMaxHeap<T>::isFull() const
{
    return vector.isFull();
}

template<typename T>
__host__ __device__
void LightMinMaxHeap<T>::clear()
{
    vector.clear();
}

