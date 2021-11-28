#pragma once

#include <cstddef>
#include <cstdlib>
#include <cassert>
#include <Utils/TypeAlias.h>

namespace Memory
{
    enum MallocType {Managed, Std};
    u32 const DefaultAlignment = 8;
    u32 const DefaultAlignmentPadding = 8;
    template<typename T>
    __host__ __device__ inline T* align(std::byte const * ptr);
    __host__ __device__ inline std::byte* align(std::byte const * ptr, u32 alignment = DefaultAlignment);
    __host__ __device__ inline uintptr_t align(uintptr_t address, u32 alignment);
    __host__ __device__ std::byte* safeMalloc(u64 size, MallocType type);
    __host__ __device__ std::byte* safeStdMalloc(u64 size);
    std::byte* safeManagedMalloc(u64 size);
}

template<typename T>
__host__ __device__
T* Memory::align(std::byte const * ptr)
{
    return reinterpret_cast<T*>(align(ptr, sizeof(T)));
}

__host__ __device__
std::byte* Memory::align(std::byte const * ptr, u32 alignment)
{
    return reinterpret_cast<std::byte*>(align(reinterpret_cast<uintptr_t>(ptr), alignment));
}

__host__ __device__
uintptr_t Memory::align(uintptr_t address, u32 alignment)
{
    return address % alignment == 0 ? address : address + alignment - (address % alignment);
}

__host__ __device__
std::byte* Memory::safeMalloc(u64 size, MallocType type)
{
    switch (type)
    {
        case Std:
            return safeStdMalloc(size);
#ifndef __CUDA_ARCH__
        case Managed:
            return safeManagedMalloc(size);
#endif
        default:
            assert(false);
            return nullptr;
    }
}

__host__ __device__
std::byte* Memory::safeStdMalloc(u64 size)
{
    void* memory = malloc(size);
    assert(memory != nullptr);
    return static_cast<std::byte*>(memory);
}

std::byte* Memory::safeManagedMalloc(u64 size)
{
    void* memory;
    cudaError_t status = cudaMallocManaged(& memory, size);
    assert(status == cudaSuccess);
    assert(memory != nullptr or size == 0);
    return static_cast<std::byte*>(memory);
}