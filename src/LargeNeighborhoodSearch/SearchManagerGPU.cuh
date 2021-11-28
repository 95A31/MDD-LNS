#pragma once

#include <thread>
#include <curand_kernel.h>
#include <LargeNeighborhoodSearch/OffloadBuffer.cuh>
#include <Options.h>
#include <Utils/CUDA.cuh>
#include <Utils/Chrono.cuh>

template<typename ProblemType, typename StateType>
class SearchManagerGPU
{
    // Members
    public:
    u64 speed;
    bool done;
    u64 iteration;
    SyncState<ProblemType, StateType> bestSolution;
    SyncState<ProblemType, StateType> neighborhoodSolution;
    private:
    ProblemType const * const problem;
    Options const * const options;
    OffloadBuffer<ProblemType,StateType> offloadBuffer;

    // Functions
    public:
    SearchManagerGPU(ProblemType const * problem, Options const * options, Memory::MallocType mallocType);
    void initializeRandomEngines();
    void searchLnsLoop(StateType const * root, bool * timeout);
    private:
    void waitDevice() const;
    void doOffloadsAsync(LargeNeighborhoodSearch::SearchPhase searchPhase);
    void generateNeighbourhoodsAsync();
};


template<typename ProblemType, typename StateType>
SearchManagerGPU<ProblemType, StateType>::SearchManagerGPU(ProblemType const * problem, Options const * options, Memory::MallocType mallocType) :
    speed(0),
    done(false),
    iteration(0),
    problem(problem),
    options(options),
    bestSolution(problem, mallocType),
    neighborhoodSolution(problem, mallocType),
    offloadBuffer(problem, options->widthGpu, options->mddsGpu, options->probEq, mallocType)
{
    bestSolution.state.invalidate();
    neighborhoodSolution.state.invalidate();
}

template<typename ProblemType, typename StateType>
__global__
void doOffloadKernel(OffloadBuffer<ProblemType,StateType>* offloadBuffer, LargeNeighborhoodSearch::SearchPhase searchPhase)
{
    u32 const index = blockIdx.x;
    offloadBuffer->doOffload(searchPhase, index);
}

template<typename ProblemType, typename StateType>
void SearchManagerGPU<ProblemType, StateType>::doOffloadsAsync(LargeNeighborhoodSearch::SearchPhase searchPhase)
{
    u32 const blockSize = Algorithms::min(options->widthGpu * problem->maxBranchingFactor, 1024u);
    u32 const blockCount = offloadBuffer.getSize();
    u32 const sharedMemSize = DecisionDiagram::MDD<ProblemType, StateType>::sizeOfScratchpadMemory(problem, options->widthGpu);
    doOffloadKernel<<<blockCount, blockSize, sharedMemSize>>>(&offloadBuffer, searchPhase);
}

template<typename ProblemType, typename StateType>
__global__
void initializeRandomEnginesKernel(OffloadBuffer<ProblemType,StateType>* offloadBuffer, u32 randomSeed)
{
    if(threadIdx.x == 0)
    {
        u32 const index = blockIdx.x;
        offloadBuffer->initializeRandomEngines(randomSeed, index);
    }
}


template<typename ProblemType, typename StateType>
void SearchManagerGPU<ProblemType, StateType>::initializeRandomEngines()
{
    // Random Numbers Generators
    if(options->mddsGpu > 0)
    {
        u32 const blockSize = 1;
        u32 const blockCount = offloadBuffer.getCapacity();
        initializeRandomEnginesKernel<<<blockCount, blockSize>>>(&offloadBuffer, options->randomSeed);
        waitDevice();
    }
}

template<typename ProblemType, typename StateType>
void SearchManagerGPU<ProblemType, StateType>::searchLnsLoop(StateType const * root, bool * timeout)
{
    done = false;
    iteration = 0;

    if(options->mddsGpu > 0)
    {
        offloadBuffer.initializeOffload(root);

        while(not *timeout)
        {
            u64 const startTime = Chrono::now();

            // Generate neighborhoods
            generateNeighbourhoodsAsync();
            waitDevice();

            // Offload
            doOffloadsAsync(LargeNeighborhoodSearch::SearchPhase::LNS);
            waitDevice();

            //Finalize offload
            offloadBuffer.getSolutions(LargeNeighborhoodSearch::SearchPhase::LNS, &bestSolution);

            u64 const elapsedTime = max(Chrono::now() - startTime, 1ul);
            speed = offloadBuffer.getCapacity() * 1000 / elapsedTime;

            iteration += 1;
        }
    }
    done = true;
}


template<typename ProblemType,typename StateType>
__global__
void generateNeighbourhoodKernel(OffloadBuffer<ProblemType,StateType>* offloadBuffer, StateType * solution)
{
    if(threadIdx.x == 0)
    {
        u32 const index = blockIdx.x;
        offloadBuffer->generateNeighborhood(&solution->selectedValues, index);
    }
}

template<typename ProblemType, typename StateType>
void SearchManagerGPU<ProblemType, StateType>::generateNeighbourhoodsAsync()
{
    u32 const blockSize = 1;
    u32 const blockCount = options->mddsGpu;
    generateNeighbourhoodKernel<<<blockCount, blockSize>>>(&offloadBuffer, &neighborhoodSolution.state);
}


template<typename ProblemType, typename StateType>
void SearchManagerGPU<ProblemType, StateType>::waitDevice() const
{
    cudaDeviceSynchronize();
}