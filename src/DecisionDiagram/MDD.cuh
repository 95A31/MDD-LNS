#pragma once

#include <algorithm>
#include <thrust/find.h>
#include <Containers/Vector.cuh>
#include <Utils/Algorithms.cuh>
#include <Utils/CUDA.cuh>
#include <Utils/Random.cuh>
#include <OptimizationProblems/Problem.cuh>
#include <LargeNeighborhoodSearch/Neighbourhood.cuh>
#include <DecisionDiagram/Context.h>
#include <DecisionDiagram/StateMetadata.cuh>

namespace DecisionDiagram
{
    template<typename ProblemType, typename StateType>
    class MDD
    {
        // Members
        private:
        static u32 const statesMetadataBufferCapacity = 1000;
        u32 const width;
        ProblemType const * const problem;
        std::byte* scratchpadMemory;
        LightVector<StateType>* currentStates;
        LightArray<StateMetadata>* statesMetadataBuffer;
        LightVector<StateMetadata>* nextStatesMetadata;
        LightVector<StateType>* nextStates;

        // Functions
        public:
        MDD(ProblemType const * problem, u32 width);
        __host__ __device__ inline void buildTopDown(StateType const * top, StateType * bottom, Vector<StateType> * cutset, Neighbourhood const * neighbourhood, RandomEngine* randomEngine, bool lns);
        static u32 sizeOfScratchpadMemory(ProblemType const * problem, u32 width);
        private:
        __host__ __device__ inline void calcNextStatesMetadata(u32 variableIdx, Neighbourhood const * neighbourhood, RandomEngine * randomEngine, bool lns);
        __host__ __device__ inline void resetStatesMetadataBuffer(RandomEngine * randomEngine);
        __host__ __device__ inline void resetNextStatesMetadata();
        __host__ __device__ inline void updateNextStatesMetadata();
        __host__ __device__ inline void calcNextStates(u32 variableIdx);
        __host__ __device__ inline void initializeScratchpadMemory();
        __host__ __device__ inline void initializeTop(StateType const * top);
        __host__ __device__ inline void saveCutset(Vector<StateType> * cutset);
        __host__ __device__ void saveInvalidBottom(StateType * bottom);
        __host__ __device__ void saveBottom(StateType * bottom);
        __host__ __device__ void swapCurrentAndNextStates();
    };
}

template<typename ProblemType, typename StateType>
DecisionDiagram::MDD<ProblemType, StateType>::MDD(ProblemType const * problem, u32 width) :
    width(width),
    problem(problem),
    scratchpadMemory(Memory::safeMalloc(sizeOfScratchpadMemory(problem, width), Memory::MallocType::Std)),
    currentStates(nullptr),
    statesMetadataBuffer(nullptr),
    nextStatesMetadata(nullptr),
    nextStates(nullptr)

{}

template<typename ProblemType, typename StateType>
__host__ __device__
void DecisionDiagram::MDD<ProblemType, StateType>::buildTopDown(StateType const * top, StateType * bottom, Vector<StateType> * cutset, Neighbourhood const * neighbourhood, RandomEngine * randomEngine, bool lns)
{
    initializeScratchpadMemory();
    initializeTop(top);
    bool cutsetStatesSaved = false;
    u32 const variableIdxBegin = currentStates->at(0)->selectedValues.getSize();
    u32 const variableIdxEnd = problem->variables.getCapacity();
    for (u32 variableIdx = variableIdxBegin; variableIdx < variableIdxEnd; variableIdx += 1)
    {
        calcNextStatesMetadata(variableIdx, neighbourhood, randomEngine, lns);
        if (nextStatesMetadata->isEmpty())
        {
            saveInvalidBottom(bottom);
            return;
        }
        calcNextStates(variableIdx);
        if (not (lns or cutsetStatesSaved))
        {
            saveCutset(cutset);
            cutsetStatesSaved = true;
        }
        swapCurrentAndNextStates();
    }
    saveBottom(bottom);

}

template<typename ProblemType, typename StateType>
u32 DecisionDiagram::MDD<ProblemType, StateType>::sizeOfScratchpadMemory(ProblemType const * problem, u32 width)
{
    u32 const currentStatesMemSize = sizeof(LightVector<StateType>) + LightVector<StateType>::sizeOfStorage(width) + (StateType::sizeOfStorage(problem) * width) + Memory::DefaultAlignmentPadding * 3;
    u32 const statesMetadataBufferMemSize = sizeof(LightArray<StateMetadata>) + LightArray<StateMetadata>::sizeOfStorage(statesMetadataBufferCapacity) + Memory::DefaultAlignmentPadding * 2;
    u32 const nextStatesMetadataMemSize = sizeof(LightVector<StateMetadata>) + LightVector<StateMetadata>::sizeOfStorage(width) + Memory::DefaultAlignmentPadding * 2;
    u32 const nextStatesMemSize = sizeof(LightVector<StateType>) + LightVector<StateType>::sizeOfStorage(width) + (StateType::sizeOfStorage(problem) * width) + Memory::DefaultAlignmentPadding * 3;
    u32 const scratchpadMemSize = currentStatesMemSize + statesMetadataBufferMemSize + nextStatesMetadataMemSize + nextStatesMemSize + Memory::DefaultAlignmentPadding;
    return scratchpadMemSize;
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DecisionDiagram::MDD<ProblemType, StateType>::calcNextStatesMetadata(u32 variableIdx, Neighbourhood const * neighbourhood, RandomEngine * randomEngine, bool lns)
{
    resetNextStatesMetadata();
    resetStatesMetadataBuffer(randomEngine);

    u32 nextStatesMetadataMaxCount = currentStates->getSize() * problem->maxBranchingFactor;
    u32 const nextStatesMetadataChunksCount = Algorithms::ceilIntDivision(nextStatesMetadataMaxCount, statesMetadataBufferCapacity);

    for (u32 chunkIdx = 0; chunkIdx < nextStatesMetadataChunksCount; chunkIdx += 1)
    {
        u32 const chunkBeginIdx = chunkIdx * statesMetadataBufferCapacity;
        u32 const chunkEndIdx = Algorithms::min(nextStatesMetadataMaxCount, chunkBeginIdx + statesMetadataBufferCapacity);
        u32 const chunkSize = chunkEndIdx - chunkBeginIdx;

#ifdef __CUDA_ARCH__
        Pair<u32> indicesBeginEnd = Algorithms::getBeginEndIndices(threadIdx.x, blockDim.x, chunkSize);
#else
        Pair<u32> indicesBeginEnd = Algorithms::getBeginEndIndices(0, 1, chunkSize);
#endif

        indicesBeginEnd.first += chunkBeginIdx;
        indicesBeginEnd.second += chunkBeginIdx;

        for (u32 nextStateMetadataIdx = indicesBeginEnd.first; nextStateMetadataIdx < indicesBeginEnd.second; nextStateMetadataIdx += 1)
        {
            u32 const currentStateIdx = nextStateMetadataIdx / problem->maxBranchingFactor;
            OptimizationProblems::ValueType const value = nextStateMetadataIdx % problem->maxBranchingFactor;
            StateType const* const currentState = currentStates->at(currentStateIdx);
            if (currentState->admissibleValuesMap.contains(value))
            {
                if ((not lns) or neighbourhood->constraintsCheck(variableIdx, value))
                {
                    u32 const stateMetadataBufferIdx = nextStateMetadataIdx - chunkBeginIdx;
                    statesMetadataBuffer->at(stateMetadataBufferIdx)->index = nextStateMetadataIdx;
                    statesMetadataBuffer->at(stateMetadataBufferIdx)->cost = calcCost(problem, currentState, value);
                }
            }
        }
        CUDA_BLOCK_BARRIER

        updateNextStatesMetadata();
    }
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DecisionDiagram::MDD<ProblemType, StateType>::calcNextStates(u32 variableIdx)
{
    u32 const nextStatesMetadataSize = nextStatesMetadata->getSize();

    //Resize
    CUDA_ONLY_FIRST_THREAD
    {
        nextStates->resize(nextStatesMetadataSize);
    }
    CUDA_BLOCK_BARRIER

#ifdef __CUDA_ARCH__
    Pair<u32> indicesBeginEnd = Algorithms::getBeginEndIndices(threadIdx.x, blockDim.x, nextStatesMetadataSize);
#else
    Pair<u32> indicesBeginEnd = Algorithms::getBeginEndIndices(0, 1, nextStatesMetadataSize);
#endif
    for (u32 nextStateIdx = indicesBeginEnd.first; nextStateIdx < indicesBeginEnd.second; nextStateIdx += 1)
    {
        StateMetadata const sm = *nextStatesMetadata->at(nextStateIdx);
        u32 const currentStateIdx = sm.index / problem->maxBranchingFactor;
        OptimizationProblems::ValueType const value = sm.index % problem->maxBranchingFactor;
        makeState(problem, currentStates->at(currentStateIdx), value, nextStatesMetadata->at(nextStateIdx)->cost, nextStates->at(nextStateIdx));
    }
    CUDA_BLOCK_BARRIER
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DecisionDiagram::MDD<ProblemType, StateType>::initializeScratchpadMemory()
{
    CUDA_ONLY_FIRST_THREAD
    {
        std::byte* freeScratchpadMemory = nullptr;
#ifdef __CUDA_ARCH__
        extern __shared__ u32 sharedMemory[];
        freeScratchpadMemory = reinterpret_cast<std::byte*>(&sharedMemory);
#else
        freeScratchpadMemory = scratchpadMemory;
#endif
        // Current states
        std::byte const * const freeScratchpadMemoryBeforeCurrentStates = freeScratchpadMemory;
        currentStates = reinterpret_cast<LightVector<StateType>*>(freeScratchpadMemory);
        freeScratchpadMemory += sizeof(LightVector<StateType>);
        freeScratchpadMemory = Memory::align(freeScratchpadMemory);
        new (currentStates) LightVector<StateType>(width, reinterpret_cast<StateType*>(freeScratchpadMemory));
        freeScratchpadMemory = Memory::align(currentStates->endOfStorage());
        for (u32 currentStateIdx = 0; currentStateIdx < currentStates->getCapacity(); currentStateIdx += 1)
        {
            StateType* currentState = currentStates->LightArray<StateType>::at(currentStateIdx);
            new (currentState) StateType(problem, freeScratchpadMemory);
            freeScratchpadMemory = Memory::align(currentState->endOfStorage(), Memory::DefaultAlignment);
        }
        u32 const sizeCurrentStates = reinterpret_cast<uintptr_t>(freeScratchpadMemory) - reinterpret_cast<uintptr_t>(freeScratchpadMemoryBeforeCurrentStates);

        // States metadata buffer
        std::byte* const freeScratchpadMemoryBeforeStatesMetadataBuffer = freeScratchpadMemory;
        statesMetadataBuffer = reinterpret_cast<LightArray<StateMetadata>*>(freeScratchpadMemory);
        freeScratchpadMemory += sizeof(LightArray<StateMetadata>);
        freeScratchpadMemory = Memory::align(freeScratchpadMemory);
        new (statesMetadataBuffer) LightArray<StateMetadata>(statesMetadataBufferCapacity, reinterpret_cast<StateMetadata*>(freeScratchpadMemory));
        freeScratchpadMemory = Memory::align(statesMetadataBuffer->endOfStorage());
        u32 const sizeStatesMetadataBuffer = reinterpret_cast<uintptr_t>(freeScratchpadMemory) - reinterpret_cast<uintptr_t>(freeScratchpadMemoryBeforeStatesMetadataBuffer);

        // Next states metadata
        std::byte* const freeScratchpadMemoryBeforeNextStatesMetadata = freeScratchpadMemory;
        nextStatesMetadata = reinterpret_cast<LightVector<StateMetadata>*>(freeScratchpadMemory);
        freeScratchpadMemory += sizeof(LightVector<StateMetadata>);
        freeScratchpadMemory = Memory::align(freeScratchpadMemory);
        new (nextStatesMetadata) LightVector<StateMetadata>(width, reinterpret_cast<StateMetadata*>(freeScratchpadMemory));
        freeScratchpadMemory = Memory::align(nextStatesMetadata->endOfStorage());
        u32 const sizeNextStatesMetadata = reinterpret_cast<uintptr_t>(freeScratchpadMemory) - reinterpret_cast<uintptr_t>(freeScratchpadMemoryBeforeNextStatesMetadata);

        // Next states
        std::byte const * const freeScratchpadMemoryBeforeNextStates = freeScratchpadMemory;
        nextStates = reinterpret_cast<LightVector<StateType>*>(freeScratchpadMemory);
        freeScratchpadMemory += sizeof(LightVector<StateType>);
        freeScratchpadMemory = Memory::align(freeScratchpadMemory);
        new (nextStates) LightVector<StateType>(width, reinterpret_cast<StateType*>(freeScratchpadMemory));
        freeScratchpadMemory = Memory::align(nextStates->endOfStorage());
        for (u32 nextStateIdx = 0; nextStateIdx < nextStates->getCapacity(); nextStateIdx += 1)
        {
            StateType* nextState = nextStates->LightArray<StateType>::at(nextStateIdx);
            new (nextState) StateType(problem, freeScratchpadMemory);
            freeScratchpadMemory = Memory::align(nextState->endOfStorage());
        }
        u32 const sizeNextStates = reinterpret_cast<uintptr_t>(freeScratchpadMemory) - reinterpret_cast<uintptr_t>(freeScratchpadMemoryBeforeNextStates);

        // Memory
        [[maybe_unused]]
        u32 const usedScratchpadMemory = sizeCurrentStates + sizeStatesMetadataBuffer + sizeNextStatesMetadata + sizeNextStates;
    }
    CUDA_BLOCK_BARRIER
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DecisionDiagram::MDD<ProblemType, StateType>::initializeTop(StateType const * top)
{
    CUDA_ONLY_FIRST_THREAD
    {
        currentStates->resize(1);
        *currentStates->at(0) = *top;
    }
    CUDA_BLOCK_BARRIER
}


template<typename ProblemType, typename StateType>
__host__ __device__
void DecisionDiagram::MDD<ProblemType, StateType>::saveCutset(Vector<StateType> * cutset)
{
    u32 const elements = nextStates->getSize();

    //Resize
    CUDA_ONLY_FIRST_THREAD
    {
        cutset->resize(elements);
    }
    CUDA_BLOCK_BARRIER

#ifdef __CUDA_ARCH__
    Pair<u32> indicesBeginEnd = Algorithms::getBeginEndIndices(threadIdx.x, blockDim.x, elements);
#else
    Pair<u32> indicesBeginEnd = Algorithms::getBeginEndIndices(0, 1, elements);
#endif
    for (u32 stateIdx = indicesBeginEnd.first; stateIdx < indicesBeginEnd.second; stateIdx += 1)
    {
        *cutset->at(stateIdx) = *nextStates->at(stateIdx);
    }
    CUDA_BLOCK_BARRIER
}


template<typename ProblemType, typename StateType>
__host__ __device__
void DecisionDiagram::MDD<ProblemType, StateType>::swapCurrentAndNextStates()
{
    CUDA_ONLY_FIRST_THREAD
    {
        LightVector<StateType>::swap(*currentStates, *nextStates);
    }
    CUDA_BLOCK_BARRIER
}


template<typename ProblemType, typename StateType>
__host__ __device__
void DecisionDiagram::MDD<ProblemType, StateType>::resetStatesMetadataBuffer(RandomEngine* randomEngine)
{
#ifdef __CUDA_ARCH__
    Pair<u32> indicesBeginEnd = Algorithms::getBeginEndIndices(threadIdx.x, blockDim.x, statesMetadataBufferCapacity);
#else
    Pair<u32> indicesBeginEnd = Algorithms::getBeginEndIndices(0, 1, statesMetadataBufferCapacity);
#endif
    for (u32 stateMetadataIdx = indicesBeginEnd.first; stateMetadataIdx < indicesBeginEnd.second; stateMetadataIdx += 1)
    {
        statesMetadataBuffer->at(stateMetadataIdx)->cost = DynamicProgramming::MaxCost;
        statesMetadataBuffer->at(stateMetadataIdx)->random = randomEngine->getFloat01(); //Race condition on GPU
    }
    CUDA_BLOCK_BARRIER
}


template<typename ProblemType, typename StateType>
__host__ __device__
void DecisionDiagram::MDD<ProblemType, StateType>::updateNextStatesMetadata()
{
    CUDA_ONLY_FIRST_THREAD
    {
        for(u32 i = 0; i < statesMetadataBufferCapacity; i += 1)
        {
            StateMetadata * stateMetadata = statesMetadataBuffer->at(i);
            if (StateMetadata::isValid(*stateMetadata))
            {
                Algorithms::sortedInsert(stateMetadata, nextStatesMetadata);
            }
        }
    }
    CUDA_BLOCK_BARRIER
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DecisionDiagram::MDD<ProblemType, StateType>::saveInvalidBottom(StateType * bottom)
{
    CUDA_ONLY_FIRST_THREAD
    {
        bottom->invalidate();
    }
    CUDA_BLOCK_BARRIER
}


template<typename ProblemType, typename StateType>
__host__ __device__
void DecisionDiagram::MDD<ProblemType, StateType>::saveBottom(StateType* bottom)
{
    CUDA_ONLY_FIRST_THREAD
    {
        *bottom = *currentStates->at(0);
    }
    CUDA_BLOCK_BARRIER
}

template<typename ProblemType, typename StateType>
__host__ __device__
void DecisionDiagram::MDD<ProblemType, StateType>::resetNextStatesMetadata()
{
    CUDA_ONLY_FIRST_THREAD
    {
        nextStatesMetadata->clear();
    }
    CUDA_BLOCK_BARRIER
}

