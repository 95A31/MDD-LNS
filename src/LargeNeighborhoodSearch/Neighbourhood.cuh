#pragma once

#include <Containers/Array.cuh>
#include <Containers/BitSet.cuh>
#include <OptimizationProblems/Problem.cuh>
#include <Utils/Random.cuh>

class Neighbourhood
{
    // Aliases, Enums, ...
    private:
    enum ConstraintType: u8 {None, Eq};

    // Members
    private:
    float probEq;
    Array<OptimizationProblems::ValueType> values;
    Array<ConstraintType> constraints;
    BitSet fixedValue;

    // Functions
    public:
    Neighbourhood(OptimizationProblems::Problem const * problem, float probEq, Memory::MallocType mallocType);
    __host__ __device__ void generate(RandomEngine* randomEngine, Vector<OptimizationProblems::ValueType>* solutionValues);
    __host__ __device__ bool constraintsCheck(u32 variableIdx, OptimizationProblems::ValueType value) const;
    __host__ __device__ void clear();
    void print(bool endLine = true);
};

Neighbourhood::Neighbourhood(OptimizationProblems::Problem const * problem, float probEq, Memory::MallocType mallocType) :
    probEq(probEq),
    values(problem->variables.getCapacity(), mallocType),
    constraints(problem->variables.getCapacity(), mallocType),
    fixedValue(problem->maxValue + 1, mallocType)
{
    clear();
}

__host__ __device__
void Neighbourhood::clear()
{
    u32 const constraintEndIdx = constraints.getCapacity();
    for(u32 constraintIdx = 0; constraintIdx < constraintEndIdx; constraintIdx +=1 )
    {
       *constraints[constraintIdx] = ConstraintType::None;
    }
    fixedValue.clear();
}

__host__ __device__
void Neighbourhood::generate(RandomEngine* randomEngine, Vector<OptimizationProblems::ValueType>* solutionValues)
{
    clear();
    for(u32 valueIdx = 0; valueIdx < solutionValues->getCapacity(); valueIdx += 1)
    {
        OptimizationProblems::ValueType const value = *solutionValues->at(valueIdx);
        float const random = randomEngine->getFloat01();

        if (random < probEq)
        {
            *constraints[valueIdx] = ConstraintType::Eq;
            *values[valueIdx] = value;
            fixedValue.insert(value);
        }
    }
}

__host__ __device__
bool Neighbourhood::constraintsCheck(unsigned int variableIdx, OptimizationProblems::ValueType value) const
{
    switch(*constraints[variableIdx])
    {
        case ConstraintType::Eq:
            return *values[variableIdx] == value;
        case ConstraintType::None:
            return not fixedValue.contains(value);
    }
    return false;
}

void Neighbourhood::print(bool endLine)
{
    auto printConstraint = [&](u32 variableIdx) -> void
    {
        switch (*constraints[variableIdx])
        {
            case ConstraintType::None:
                printf("*");
                break;
            case ConstraintType::Eq:
                printf("\033[30;42m%2d\033[0m", *values[variableIdx]);
                break;
        }
    };

    printf("[");
    printConstraint(0);
    for (u32 variableIdx = 1; variableIdx < constraints.getCapacity(); variableIdx += 1)
    {
        printf(",");
        printConstraint(variableIdx);
    }
    printf(endLine ? "]\n" : "]");
}