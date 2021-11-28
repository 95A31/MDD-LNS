#pragma once

#include <thread>
#include <Utils/Memory.cuh>
#include <OptimizationProblems/Problem.cuh>

template<typename ProblemType, typename StateType>
class SyncState
{
    public:
    StateType state;
    std::mutex mutex;

    SyncState(ProblemType const * problem, std::byte* storage);
    SyncState(ProblemType const * problem, Memory::MallocType mallocType);
};

template<typename ProblemType, typename StateType>
SyncState<ProblemType,StateType>::SyncState(ProblemType const * problem, std::byte* storage) :
    state(problem, storage),
    mutex()
{}

template<typename ProblemType, typename StateType>
SyncState<ProblemType,StateType>::SyncState(ProblemType const * problem, Memory::MallocType mallocType) :
    state(problem, mallocType),
    mutex()
{}
