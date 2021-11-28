#pragma once

#include <cassert>
#include <cstdio>
#include <cstring>
#include <thrust/copy.h>
#include <thrust/swap.h>
#include <Utils/Memory.cuh>
#include <Utils/TypeAlias.h>

// https://stackoverflow.com/questions/47981/how-do-you-set-clear-and-toggle-a-single-bit

class BitSet
{
    // Members
    protected:
    u32 maxValue;
    u32* storage;

    // Functions
    public:
    __host__ __device__ BitSet(u32 maxValue, u32* storage);
    __host__ __device__ BitSet(u32 maxValue, Memory::MallocType mallocType);
    __host__ __device__ ~BitSet();
    __host__ __device__ inline bool contains(u32 value) const;
    __host__ __device__ void clear();
    __host__ __device__ inline std::byte* endOfStorage() const;
    __host__ __device__ inline void erase(u32 value);
    __host__ __device__ inline u32 getMaxValue() const;
    __host__ __device__ u32 getSize() const;
    __host__ __device__ inline void insert(u32 value);
    __host__ __device__ bool intersect(BitSet const & other);
    __host__ __device__ bool isEmpty() const;
    __host__ __device__ bool isFull() const;
    __host__ __device__ void merge(BitSet const & other);
    __host__ __device__ BitSet& operator=(BitSet const & other);
    __host__ __device__ void print(bool endLine = true) const;
    __host__ __device__ inline static u32 sizeOfStorage(u32 maxValue);
    __host__ __device__ inline static void swap(BitSet& bs0, BitSet& bs1);
    private:
    __host__ __device__ inline u32* begin() const;
    __host__ __device__ inline static u32 chunksCount(u32 maxValue);
    __host__ __device__ inline static u32 chunkIndex(u32 value);
    __host__ __device__ inline static u32 chunkOffset(u32 value);
    __host__ __device__ inline u32* end() const;
    __host__ __device__ static u32* mallocStorage(u32 maxValue, Memory::MallocType mallocType);
};

__host__ __device__
BitSet::BitSet(u32 maxValue, u32* storage) :
    maxValue(maxValue),
    storage(storage)
{
    clear();
}

__host__ __device__
BitSet::BitSet(u32 maxValue, Memory::MallocType mallocType) :
    BitSet(maxValue, mallocStorage(maxValue, mallocType))
{}

__host__ __device__
BitSet::~BitSet()
{
    free(storage);
}

__host__ __device__
bool BitSet::contains(u32 value) const
{
    if (value <= maxValue)
    {
        u32 const chunkIndex = BitSet::chunkIndex(value);
        u32 const chunkOffset = BitSet::chunkOffset(value);
        u32 const mask = 1u;
        return static_cast<bool>((storage[chunkIndex] >> chunkOffset) & mask);
    }
    else
    {
      return false;
    }
}

__host__ __device__
void BitSet::clear()
{
    u32 const chunkEndIdx = chunkIndex(maxValue) + 1;
    for(u32 chunkIdx = 0; chunkIdx < chunkEndIdx; chunkIdx +=1)
    {
        storage[chunkIdx] = 0;
    }
}

__host__ __device__
std::byte* BitSet::endOfStorage() const
{
    u32 const chunksCount = BitSet::chunksCount(maxValue);
    return reinterpret_cast<std::byte*>(storage + chunksCount);
}

__host__ __device__
void BitSet::erase(u32 value)
{
    assert(value <= maxValue);
    u32 const chunkIndex = BitSet::chunkIndex(value);
    u32 const chunkOffset = BitSet::chunkOffset(value);
    u32 const mask = ~(1u << chunkOffset);
    storage[chunkIndex] &= mask;
}

__host__ __device__
u32 BitSet::getMaxValue() const
{
    return maxValue;
}

__host__ __device__
u32 BitSet::getSize() const
{
    u32 size = 0;
    u32 const chunksCount = BitSet::chunksCount(maxValue);
    for(u32 chunkIndex = 0; chunkIndex < chunksCount; chunkIndex += 1)
    {
#ifdef __CUDA_ARCH__
        size += __popc(storage[chunkIndex]);
#else
        size += __builtin_popcount(storage[chunkIndex]);
#endif
    }
    return size;
}

__host__ __device__
void BitSet::insert(u32 value)
{
    assert(value <= maxValue);
    u32 const chunkIndex = BitSet::chunkIndex(value);
    u32 const chunkOffset = BitSet::chunkOffset(value);
    u32 const mask = 1u << chunkOffset;
    storage[chunkIndex] |= mask;
}

__host__ __device__
bool BitSet::intersect(BitSet const& other)
{
    assert(maxValue == other.maxValue);
    u32 const chunksCount = BitSet::chunksCount(maxValue);
    for(u32 chunkIndex = 0; chunkIndex < chunksCount; chunkIndex += 1)
    {
        if((storage[chunkIndex] & other.storage[chunkIndex]) != 0)
        {
            return true;
        }
    }
    return false;
}

__host__ __device__
bool BitSet::isEmpty() const
{
    return getSize() == 0;
}

__host__ __device__
bool BitSet::isFull() const
{
    return getSize() == maxValue + 1;
}

__host__ __device__
void BitSet::merge(BitSet const & other)
{
    assert(maxValue == other.maxValue);
    u32 const chunksCount = BitSet::chunksCount(maxValue);
    for(u32 chunkIndex = 0; chunkIndex < chunksCount; chunkIndex += 1)
    {
        storage[chunkIndex] |= other.storage[chunkIndex];
    }
}

__host__ __device__
BitSet& BitSet::operator=(BitSet const & other)
{
    assert(maxValue == other.maxValue);
    thrust::copy(thrust::seq, other.begin(), other.end(), this->begin());
    return *this;
}

__host__ __device__
void BitSet::print(bool endLine) const
{
    auto printValue = [&](u32 value) -> void
    {
        printf(contains(value) ? "1" : "0");
    };

    printf("[");
    if(maxValue > 0)
    {
        printValue(0);
        for (u32 value = 1; value <= maxValue; value += 1)
        {
            printf(",");
            printValue(value);
        }
    }
    printf(endLine ? "]\n" : "]");
}

__host__ __device__
u32 BitSet::sizeOfStorage(u32 maxValue)
{
    return
        sizeof(u32) * BitSet::chunksCount(maxValue);
}

__host__ __device__
void BitSet::swap(BitSet& bs0, BitSet& bs1)
{
    thrust::swap(bs0.maxValue, bs1.maxValue);
    thrust::swap(bs0.storage, bs1.storage);
}

__host__ __device__
u32* BitSet::begin() const
{
    return storage;
}

__host__ __device__
u32 BitSet::chunksCount(u32 maxValue)
{
    return chunkIndex(maxValue) + 1;
}

__host__ __device__
u32 BitSet::chunkIndex(u32 value)
{
    return value / 32;
}

__host__ __device__
u32 BitSet::chunkOffset(u32 value)
{
    return 31 - (value % 32);
}

__host__ __device__
u32* BitSet::end() const
{
    return storage + BitSet::chunksCount(maxValue);
}

__host__ __device__
u32* BitSet::mallocStorage(u32 maxValue, Memory::MallocType mallocType)
{
    u32 const storageSize = sizeOfStorage(maxValue);
    return reinterpret_cast<u32*>(Memory::safeMalloc(storageSize, mallocType));
}
