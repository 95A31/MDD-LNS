#pragma once

#include <thrust/remove.h>
#include <DecisionDiagram/StateMetadata.cuh>
#include <OptimizationProblems/CTWProblem.cuh>
#include <DynamicProgramming/CTWPState.cuh>
namespace DynamicProgramming
{
    __host__ __device__ inline void calcAdmissibleValues(OptimizationProblems::CTWProblem const * problem, CTWPState* state);
    __host__ __device__ inline DynamicProgramming::CostType calcCost(OptimizationProblems::CTWProblem const * problem, CTWPState const * currentState, OptimizationProblems::ValueType const value);
    __host__ __device__ inline OptimizationProblems::ValueType calcOtherEnd(OptimizationProblems::CTWProblem const * problem, OptimizationProblems::ValueType const value);
    __host__ __device__ inline bool closeInterruptedPair(OptimizationProblems::CTWProblem const * problem, CTWPState const * state, OptimizationProblems::ValueType const value);
    __host__ __device__ inline bool isClosingPair(OptimizationProblems::CTWProblem const * problem, CTWPState const * state);
    void makeRoot(OptimizationProblems::CTWProblem const * problem, CTWPState* root);
    __host__ __device__ inline bool interruptPair(OptimizationProblems::CTWProblem const * problem, CTWPState const * state, OptimizationProblems::ValueType const value);
    __host__ __device__ inline void makeState(OptimizationProblems::CTWProblem const * problem, CTWPState const * currentState, OptimizationProblems::ValueType value, DynamicProgramming::CostType cost, CTWPState* nextState);
    __host__ __device__ inline void updateInterruptedPairs(OptimizationProblems::CTWProblem const * problem, CTWPState* state, OptimizationProblems::ValueType value);
    __host__ __device__ inline void updatePrecedencesCount(OptimizationProblems::CTWProblem const * problem, CTWPState const * state, OptimizationProblems::ValueType value);
    __host__ __device__ inline bool checkDisjunctive1(OptimizationProblems::CTWProblem const * problem, CTWPState const * state, OptimizationProblems::ValueType value);
    __host__ __device__ inline bool checkDisjunctive2(OptimizationProblems::CTWProblem const * problem, CTWPState const * state, OptimizationProblems::ValueType value);
    __host__ __device__ inline u8 updatedS(OptimizationProblems::CTWProblem const * problem, CTWPState const * state, OptimizationProblems::ValueType value);
    __host__ __device__ inline u8 updatedM(OptimizationProblems::CTWProblem const * problem, CTWPState const * state, OptimizationProblems::ValueType value);
    __host__ __device__ inline u8 updatedL(OptimizationProblems::CTWProblem const * problem, CTWPState const * state, OptimizationProblems::ValueType value);
    __host__ __device__ inline u8 updatedN(OptimizationProblems::CTWProblem const * problem, CTWPState const * state, OptimizationProblems::ValueType value);
}

__host__ __device__
void DynamicProgramming::calcAdmissibleValues(OptimizationProblems::CTWProblem const * problem, DynamicProgramming::CTWPState* state)
{
    state->admissibleValuesMap.clear();
    for (u32 value = 0; value < state->precedencesCount.getCapacity(); value += 1)
    {
        if(not state->selectedValuesMap.contains(value))
        {
            if (*state->precedencesCount[value] == 0)
            {
                if (checkDisjunctive1(problem, state, value))
                {
                    if (checkDisjunctive2(problem, state, value))
                    {
                        state->admissibleValuesMap.insert(value);
                    }
                }
            }
        }
    }
}

__host__ __device__
DynamicProgramming::CostType DynamicProgramming::calcCost(OptimizationProblems::CTWProblem const * problem, CTWPState const * currentState, OptimizationProblems::ValueType const value)
{
    u32 s = 0;
    u32 m = 0;
    u32 l = 0;
    u32 n = 0;
    u32 const k = problem->k;

    if(not currentState->selectedValues.isEmpty())
    {
        s = updatedS(problem, currentState, value);
        m = updatedM(problem,currentState,value);
        l = updatedL(problem,currentState,value);
        n = updatedN(problem,currentState,value);
    }

    u32 const cost = (k * k * k * s) + (k * k * m) + (k * l) + n;
    return cost;
}

void DynamicProgramming::makeRoot(OptimizationProblems::CTWProblem const* problem, DynamicProgramming::CTWPState* root)
{
    thrust::fill(thrust::seq, root->precedencesCount.begin(), root->precedencesCount.end(), 0);
    for(Pair<OptimizationProblems::ValueType> const * atomicConstraint = problem->atomicConstraints.begin(); atomicConstraint != problem->atomicConstraints.end(); atomicConstraint += 1)
    {
        OptimizationProblems::ValueType const & i = atomicConstraint->first;
        *root->precedencesCount[i] += 1;
    }
    calcAdmissibleValues(problem, root);
}


__host__ __device__
void DynamicProgramming::makeState(OptimizationProblems::CTWProblem const * problem, CTWPState const * currentState, OptimizationProblems::ValueType value, DynamicProgramming::CostType cost, CTWPState* nextState)
{
    *nextState = *currentState;
    nextState->cost = cost;

    nextState->s = updatedS(problem, currentState, value);
    nextState->m = updatedM(problem, currentState, value);
    nextState->l = updatedL(problem, currentState, value);
    nextState->n = updatedN(problem, currentState, value);

    updateInterruptedPairs(problem, nextState, value);
    updatePrecedencesCount(problem, nextState, value);
    nextState->selectValue(value);
    calcAdmissibleValues(problem, nextState);
}

__host__ __device__
OptimizationProblems::ValueType DynamicProgramming::calcOtherEnd(OptimizationProblems::CTWProblem const * problem, OptimizationProblems::ValueType const value)
{
    if(value != 0)
    {
        return value <= problem->b ? value + problem->b : value - problem->b;
    }
    else
    {
        return 0;
    }
}

__host__ __device__
bool DynamicProgramming::closeInterruptedPair(OptimizationProblems::CTWProblem const * problem, DynamicProgramming::CTWPState const * state, OptimizationProblems::ValueType const value)
{
    if(not state->selectedValues.isEmpty())
    {
        // Selected values = i,...
        OptimizationProblems::ValueType const otherEnd = calcOtherEnd(problem, value);
        bool const isPresentOtherEnd = state->selectedValuesMap.contains(otherEnd);
        return isPresentOtherEnd and otherEnd != *state->selectedValues.back();
    }
    else
    {
        return false;
    }
}

__host__ __device__
bool DynamicProgramming::interruptPair(OptimizationProblems::CTWProblem const* problem, DynamicProgramming::CTWPState const* state, OptimizationProblems::ValueType const value)
{
    if(not state->selectedValues.isEmpty())
    {
        // Selected values = i,...
        OptimizationProblems::ValueType const i = *state->selectedValues.back();
        OptimizationProblems::ValueType const j = calcOtherEnd(problem, i);
        bool const isPresentJ = state->selectedValuesMap.contains(j);
        if ((not isPresentJ) and value != j)
        {
           return true;
        }
    }
    return false;
}

__host__ __device__
void DynamicProgramming::updatePrecedencesCount(OptimizationProblems::CTWProblem const* problem, DynamicProgramming::CTWPState const* state, OptimizationProblems::ValueType value)
{
    LightVector<u16> const * const atomicConstraintsMap = problem->atomicToCheck[value];
    for(u16 const * atomicConstraintIdx = atomicConstraintsMap->begin(); atomicConstraintIdx != atomicConstraintsMap->end(); atomicConstraintIdx += 1)
    {
        Pair<OptimizationProblems::ValueType> const * const atomicConstraint = problem->atomicConstraints.at(*atomicConstraintIdx);
        OptimizationProblems::ValueType const & i = atomicConstraint->first;
        OptimizationProblems::ValueType const & j = atomicConstraint->second;
        if (value == j)
        {
            *state->precedencesCount[i] -= 1;
        }
    }
}

__host__ __device__
bool DynamicProgramming::checkDisjunctive1(OptimizationProblems::CTWProblem const* problem, DynamicProgramming::CTWPState const* state, OptimizationProblems::ValueType value)
{
    LightVector<u16> const * const disjunctive1ToCheck = problem->disjunctive1ToCheck[value];
    for(u16 const * disjunctiveConstraint1Idx = disjunctive1ToCheck->begin(); disjunctiveConstraint1Idx != disjunctive1ToCheck->end(); disjunctiveConstraint1Idx += 1)
    {
        Triple<OptimizationProblems::ValueType> const * const disjunctiveConstraint1 = problem->disjunctiveConstraints.at(*disjunctiveConstraint1Idx);
        OptimizationProblems::ValueType const & i = disjunctiveConstraint1->second;
        OptimizationProblems::ValueType const & j = disjunctiveConstraint1->third;
        bool const isPresentI = state->selectedValuesMap.contains(i);
        bool const isPresentJ = state->selectedValuesMap.contains(j);
        if ((not isPresentI) and (not isPresentJ))
        {
            return false;
        }
    }
    return true;
}

__host__ __device__
bool DynamicProgramming::checkDisjunctive2(OptimizationProblems::CTWProblem const* problem, DynamicProgramming::CTWPState const* state, OptimizationProblems::ValueType value)
{
    LightVector<u16> const * const disjunctive2ToCheck = problem->disjunctive2ToCheck[value];
    for(u16 const * disjunctiveConstraint2Idx = disjunctive2ToCheck->begin(); disjunctiveConstraint2Idx != disjunctive2ToCheck->end(); disjunctiveConstraint2Idx += 1)
    {
        Triple<OptimizationProblems::ValueType> const * const disjunctiveConstraint2 = problem->disjunctiveConstraints.at(*disjunctiveConstraint2Idx);
        OptimizationProblems::ValueType const & i = disjunctiveConstraint2->second;
        OptimizationProblems::ValueType const & j = disjunctiveConstraint2->third;
        bool const isPresentI = state->selectedValuesMap.contains(i);
        bool const isPresentJ = state->selectedValuesMap.contains(j);
        if (isPresentJ and (not isPresentI))
        {
            return false;
        }
    }
    return true;
}

__host__ __device__
u8 DynamicProgramming::updatedS(OptimizationProblems::CTWProblem const* problem, DynamicProgramming::CTWPState const * state, OptimizationProblems::ValueType value)
{
    u8 s = state->s;
    if (interruptPair(problem,state,value))
    {
        s += 1;
    }
    return s;
}

__host__ __device__
u8 DynamicProgramming::updatedM(OptimizationProblems::CTWProblem const * problem, DynamicProgramming::CTWPState const * state, OptimizationProblems::ValueType value)
{
    u32 interruptedPairCount = state->interruptedPairs.getSize();
    if(closeInterruptedPair(problem,state,value)) // Closing interrupted pair
    {
        interruptedPairCount -= 1;
    }
    return static_cast<u8>(max(state->m, interruptedPairCount));

}

__host__ __device__
u8 DynamicProgramming::updatedL(OptimizationProblems::CTWProblem const * problem, DynamicProgramming::CTWPState const * state, OptimizationProblems::ValueType value)
{
    if (not state->interruptedPairs.isEmpty())
    {
        u32 oldestInterruptedPairAge = state->selectedValues.getSize() - 1 - state->interruptedPairs.front()->second;
        if(calcOtherEnd(problem, value) != state->interruptedPairs.front()->first) // Not closing oldest interrupted pair
        {
            oldestInterruptedPairAge +=  1;
        }
        return static_cast<u8>(max(oldestInterruptedPairAge, state->l));
    }
    else
    {
        return state->l;
    }
}

__host__ __device__
u8 DynamicProgramming::updatedN(OptimizationProblems::CTWProblem const* problem, DynamicProgramming::CTWPState const* state, OptimizationProblems::ValueType value)
{
    u8 n = state->n;
    LightVector<u16> const * const softAtomicConstraintsMap = problem->softAtomicToCheck[value];
    for(u16 const * softAtomicConstraintIdx = softAtomicConstraintsMap->begin(); softAtomicConstraintIdx != softAtomicConstraintsMap->end(); softAtomicConstraintIdx += 1)
    {
        Pair<OptimizationProblems::ValueType> const * const softAtomicConstraint = problem->softAtomicConstraints.at(*softAtomicConstraintIdx);
        OptimizationProblems::ValueType const & j = softAtomicConstraint->second;
        bool isPresentJ = state->selectedValuesMap.contains(j);
        if (not isPresentJ)
        {
            n += 1;
        }
    }
    return n;
}

__host__ __device__
void DynamicProgramming::updateInterruptedPairs(OptimizationProblems::CTWProblem const* problem, DynamicProgramming::CTWPState* state, OptimizationProblems::ValueType value)
{
    if(interruptPair(problem, state, value))
    {
        Pair<OptimizationProblems::ValueType> const openPair(*state->selectedValues.back(), static_cast<OptimizationProblems::ValueType>(state->selectedValues.getSize() - 1));
        state->interruptedPairs.pushBack(&openPair);
    }
    if(closeInterruptedPair(problem, state, value))
    {
        OptimizationProblems::ValueType const otherEnd = calcOtherEnd(problem, value);
        Pair<OptimizationProblems::ValueType> const * const openPairsEnd = thrust::remove_if(thrust::seq, state->interruptedPairs.begin(), state->interruptedPairs.end(), [=] __host__ __device__ (Pair<OptimizationProblems::ValueType> const & openPair) -> bool
        {
            return openPair.first == otherEnd;
        });
        state->interruptedPairs.resize(openPairsEnd);
    }
}
