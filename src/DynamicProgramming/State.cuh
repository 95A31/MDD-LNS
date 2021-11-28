#pragma once

#include <Containers/BitSet.cuh>
#include <Containers/Vector.cuh>
#include <OptimizationProblems/Problem.cuh>
#include <DynamicProgramming/Context.h>

namespace DynamicProgramming
{
    class  State
    {
        // Members
        public:
        CostType cost;
        BitSet selectedValuesMap;
        BitSet admissibleValuesMap;
        Vector<OptimizationProblems::ValueType> selectedValues;

        // Functions
        public:
        __host__ __device__ State(OptimizationProblems::Problem const * problem, std::byte* storage);
        __host__ __device__ State(OptimizationProblems::Problem const * problem, Memory::MallocType mallocType);
        __host__ __device__ inline std::byte* endOfStorage() const;
        __host__ __device__ static std::byte* mallocStorages(OptimizationProblems::Problem const*  problem, u32 statesCount, Memory::MallocType mallocType);
        __host__ __device__ State& operator=(State const & other);
        __host__ __device__ bool operator<(State const & other) const;
        __host__ __device__ inline void invalidate();
        __host__ __device__ inline bool isValid();
        __host__ __device__ void print(bool endLine = true, bool reverse = false) const;
        __host__ __device__ inline void selectValue(OptimizationProblems::ValueType value);
        __host__ __device__ static u32 sizeOfStorage(OptimizationProblems::Problem const* problem);
        __host__ __device__ static void swap(State& s0, State& s1);
    };
}

__host__ __device__
DynamicProgramming::State::State(OptimizationProblems::Problem const * problem, std::byte* storage) :
    cost(0),
    selectedValuesMap(problem->maxValue, reinterpret_cast<u32*>(storage)),
    admissibleValuesMap(problem->maxValue, Memory::align<u32>(selectedValuesMap.endOfStorage())),
    selectedValues(problem->variables.getCapacity(), Memory::align<OptimizationProblems::ValueType>(admissibleValuesMap.endOfStorage()))
{}

__host__ __device__
DynamicProgramming::State::State(OptimizationProblems::Problem const * problem, Memory::MallocType mallocType) :
    State(problem, mallocStorages(problem,1,mallocType))
{}

__host__ __device__
std::byte* DynamicProgramming::State::endOfStorage() const
{
    return selectedValues.endOfStorage();
}

__host__ __device__
std::byte* DynamicProgramming::State::mallocStorages(const OptimizationProblems::Problem* problem, u32 statesCount, Memory::MallocType mallocType)
{
    return Memory::safeMalloc(sizeOfStorage(problem) * statesCount, mallocType);
}

__host__ __device__
DynamicProgramming::State& DynamicProgramming::State::operator=(DynamicProgramming::State const & other)
{
    cost = other.cost;
    selectedValuesMap = other.selectedValuesMap;
    admissibleValuesMap = other.admissibleValuesMap;
    selectedValues = other.selectedValues;
    return *this;
}

__host__ __device__
void DynamicProgramming::State::invalidate()
{
    cost = DynamicProgramming::MaxCost;
}

__host__ __device__
bool DynamicProgramming::State::isValid()
{
    return cost != DynamicProgramming::MaxCost;
}

__host__ __device__
void DynamicProgramming::State::print(bool endLine, bool reverse) const
{
    selectedValues.print(endLine, reverse);
}

__host__ __device__
void DynamicProgramming::State::selectValue(OptimizationProblems::ValueType value)
{
    selectedValues.pushBack(&value);
    selectedValuesMap.insert(value);
}

__host__ __device__
u32 DynamicProgramming::State::sizeOfStorage(OptimizationProblems::Problem const * problem)
{
    return
        BitSet::sizeOfStorage(problem->maxValue) + // selectedValuesMap
        BitSet::sizeOfStorage(problem->maxValue) + // admissibleValuesMap
        Vector<OptimizationProblems::ValueType>::sizeOfStorage(problem->variables.getCapacity()) + // selectedValues
        Memory::DefaultAlignmentPadding * 3; // alignment padding
}

__host__ __device__
void DynamicProgramming::State::swap(DynamicProgramming::State& s0, DynamicProgramming::State& s1)
{
    thrust::swap(s0.cost, s1.cost);
    BitSet::swap(s0.selectedValuesMap, s1.selectedValuesMap);
    BitSet::swap(s0.admissibleValuesMap, s1.admissibleValuesMap);
    Vector<OptimizationProblems::ValueType>::swap(s0.selectedValues, s1.selectedValues);
}

__host__ __device__
bool DynamicProgramming::State::operator<(DynamicProgramming::State const & other) const
{
    return cost < other.cost;
}