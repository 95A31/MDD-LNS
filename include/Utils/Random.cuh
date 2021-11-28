#pragma once

#include <random>
#include <curand_kernel.h>
#include <Utils/Memory.cuh>

class RandomEngine
{
    private:
    std::mt19937 hostEngine;
    curandState deviceEngine;

    public:
    __host__ __device__ void initialize(u32 randomSeed);
    __host__ __device__ inline float getFloat01();
};

__host__ __device__
void RandomEngine::initialize(u32 randomSeed)
{
#ifdef __CUDA_ARCH__
    curand_init(randomSeed, 0, 0, &deviceEngine);
#else
    new (&hostEngine) std::mt19937(randomSeed);
#endif
}

__host__ __device__
float RandomEngine::getFloat01()
{
#ifdef __CUDA_ARCH__
    return curand_uniform(&deviceEngine);
#else
    std::uniform_real_distribution<float> uniform01(0.0,1.0);
    return uniform01(hostEngine);
#endif
}