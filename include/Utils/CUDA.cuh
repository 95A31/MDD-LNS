#pragma once

#ifdef __CUDA_ARCH__
    #define CUDA_BLOCK_BARRIER __syncthreads();
    #define CUDA_ONLY_FIRST_THREAD if (threadIdx.x == 0)
    #define ABORT __trap();
#else
    #define CUDA_BLOCK_BARRIER
    #define CUDA_ONLY_FIRST_THREAD
    #define ABORT abort();
#endif