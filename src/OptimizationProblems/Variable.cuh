#pragma once

#include <OptimizationProblems/Context.h>

namespace OptimizationProblems
{
    class Variable
    {
        // Members
        public:
        ValueType minValue;
        ValueType maxValue;

        // Functions
        public:
        Variable(ValueType minValue, ValueType maxValue);
        __host__ __device__ inline bool boundsCheck(ValueType value);
    };
}

OptimizationProblems::Variable::Variable(ValueType minValue, ValueType maxValue) :
    minValue(minValue),
    maxValue(maxValue)
{
    assert(maxValue < OptimizationProblems::MaxValue);
}

__host__ __device__
bool OptimizationProblems::Variable::boundsCheck(OptimizationProblems::ValueType value)
{
    return minValue <= value and value <= maxValue;
}
