#pragma once

#include <DynamicProgramming/State.cuh>
#include <OptimizationProblems/SOProblem.cuh>

namespace DynamicProgramming
{
    class SOPState : public State
    {
        // Members
        public:
        Array<u16> precedencesCount;

        // Functions
        public:
        __host__ __device__ SOPState(OptimizationProblems::SOProblem const * problem, std::byte* storage);
        __host__ __device__ SOPState(OptimizationProblems::SOProblem const * problem, Memory::MallocType mallocType);
        __host__ __device__ inline std::byte* endOfStorage() const;
        __host__ __device__ static std::byte* mallocStorages(OptimizationProblems::SOProblem const *  problem, unsigned int statesCount, Memory::MallocType mallocType);
        __host__ __device__ SOPState& operator=(SOPState const & other);
        __host__ __device__ void print(bool endLine = true) const;
        __host__ __device__ static unsigned int sizeOfStorage(OptimizationProblems::SOProblem const * problem);
        __host__ __device__ static void swap(SOPState& sops0, SOPState& sops1);
    };
}

__host__ __device__
DynamicProgramming::SOPState::SOPState(OptimizationProblems::SOProblem const * problem, std::byte* storage) :
    State(problem, storage),
    precedencesCount(problem->variables.getCapacity(), Memory::align<u16>(this->State::endOfStorage()))
{}

__host__ __device__
DynamicProgramming::SOPState::SOPState(OptimizationProblems::SOProblem const* problem, Memory::MallocType mallocType) :
    SOPState(problem, mallocStorages(problem,1,mallocType))
{}

__host__ __device__
std::byte* DynamicProgramming::SOPState::endOfStorage() const
{
    return precedencesCount.endOfStorage();
}

__host__ __device__
std::byte* DynamicProgramming::SOPState::mallocStorages(OptimizationProblems::SOProblem const* problem, unsigned int statesCount, Memory::MallocType mallocType)
{
    return Memory::safeMalloc(sizeOfStorage(problem) * statesCount, mallocType);
}

__host__ __device__
DynamicProgramming::SOPState& DynamicProgramming::SOPState::operator=(DynamicProgramming::SOPState const & other)
{
    State::operator=(other);
    precedencesCount = other.precedencesCount;
    return *this;
}

__host__ __device__
void DynamicProgramming::SOPState::print(bool endLine) const
{
    State::print(endLine);
}

__host__ __device__
unsigned int DynamicProgramming::SOPState::sizeOfStorage(OptimizationProblems::SOProblem const * problem)
{
    return
        State::sizeOfStorage(problem) +
        Array<u16>::sizeOfStorage(problem->variables.getCapacity()) + // precedencesCount
        Memory::DefaultAlignmentPadding * 2;
}

__host__ __device__
void DynamicProgramming::SOPState::swap(DynamicProgramming::SOPState& sops0, DynamicProgramming::SOPState& sops1)
{
    State::swap(sops0, sops1);
    Array<u16>::swap(sops0.precedencesCount, sops1.precedencesCount);
}
