#pragma once

#include <Containers/Array.cuh>
#include <Containers/Vector.cuh>
#include <Containers/Pair.cuh>
#include <Containers/Triple.cuh>
#include <External/Nlohmann/json.hpp>
#include <Utils/Algorithms.cuh>
#include <DynamicProgramming/Context.h>
#include <OptimizationProblems/Problem.cuh>

namespace OptimizationProblems
{
    class CTWProblem : public Problem
    {
        // Members
        public:
        ValueType const b;
        ValueType const k;
        Vector<Pair<ValueType>> atomicConstraints;
        Vector<Triple<ValueType>> disjunctiveConstraints;
        Vector<Pair<ValueType>> softAtomicConstraints;
        Array<LightVector<u16>> atomicToCheck;
        Array<LightVector<u16>> disjunctive1ToCheck;
        Array<LightVector<u16>> disjunctive2ToCheck;
        Array<LightVector<u16>> softAtomicToCheck;

        // Functions
        public:
        CTWProblem(u32 b, u32 k, Memory::MallocType mallocType);
    };

    template<>
    OptimizationProblems::CTWProblem* parseInstance<CTWProblem>(char const * problemFilename, Memory::MallocType mallocType);
}

OptimizationProblems::CTWProblem::CTWProblem(u32 b, u32 k, Memory::MallocType mallocType) :
    Problem(k + 1, mallocType),
    b(b),
    k(k),
    atomicConstraints(k * k, mallocType),
    disjunctiveConstraints(k * k * k, mallocType),
    softAtomicConstraints(k * k, mallocType),
    atomicToCheck((k + 1) , mallocType),
    disjunctive1ToCheck((k + 1), mallocType),
    disjunctive2ToCheck((k + 1), mallocType),
    softAtomicToCheck((k + 1), mallocType)
{
    u32 storagesSize = sizeof(u16) * k * atomicToCheck.getCapacity();
    u16* storages = reinterpret_cast<u16*>(Memory::safeMalloc(storagesSize, mallocType));
    for (u32 index = 0; index < atomicToCheck.getCapacity(); index += 1)
    {
        new (atomicToCheck[index]) LightVector<u16>(k, storages);
        storages = reinterpret_cast<u16*>(atomicToCheck[index]->endOfStorage());
    }

    storagesSize = sizeof(u16) * k * k * disjunctive1ToCheck.getCapacity();
    storages = reinterpret_cast<u16*>(Memory::safeMalloc(storagesSize, mallocType));
    for (u32 index = 0; index < disjunctive1ToCheck.getCapacity(); index += 1)
    {
        new (disjunctive1ToCheck[index]) LightVector<u16>(k * k, storages);
        storages = reinterpret_cast<u16*>(disjunctive1ToCheck[index]->endOfStorage());
    }

    storagesSize = sizeof(u16) * k * k * disjunctive2ToCheck.getCapacity();
    storages = reinterpret_cast<u16*>(Memory::safeMalloc(storagesSize, mallocType));
    for (u32 index = 0; index < disjunctive2ToCheck.getCapacity(); index += 1)
    {
        new (disjunctive2ToCheck[index]) LightVector<u16>(k * k, storages);
        storages = reinterpret_cast<u16*>(disjunctive2ToCheck[index]->endOfStorage());
    }

    storagesSize = sizeof(u16) * k * softAtomicToCheck.getCapacity();
    storages = reinterpret_cast<u16*>(Memory::safeMalloc(storagesSize, mallocType));
    for (u32 index = 0; index < softAtomicToCheck.getCapacity(); index += 1)
    {
        new (softAtomicToCheck[index]) LightVector<u16>(k, storages);
        storages = reinterpret_cast<u16*>(softAtomicToCheck[index]->endOfStorage());
    }
}

template<>
OptimizationProblems::CTWProblem* OptimizationProblems::parseInstance(char const * problemFilename, Memory::MallocType mallocType)
{
    // Parse json
    std::ifstream problemFile(problemFilename);
    nlohmann::json problemJson;
    problemFile >> problemJson;

    // Init problem
    u32 const problemSize = sizeof(OptimizationProblems::CTWProblem);
    OptimizationProblems::CTWProblem* const problem = reinterpret_cast<OptimizationProblems::CTWProblem*>(Memory::safeMalloc(problemSize, mallocType));
    u32 const b = problemJson["b"];
    u32 const k = problemJson["k"];
    new (problem) OptimizationProblems::CTWProblem(b, k, mallocType);

    // Init variables
    Variable variable(0,0);
    problem->add(&variable);
    variable.minValue = 1;
    variable.maxValue = k;
    for (u32 variableIdx = 1; variableIdx <= k; variableIdx += 1)
    {
        problem->add(&variable);
    }

    // Atomic constraints
    for (u16 atomicConstraintIdx = 0; atomicConstraintIdx < problemJson["AtomicConstraints"].size(); atomicConstraintIdx += 1)
    {
        auto& constraint = problemJson["AtomicConstraints"][atomicConstraintIdx];
        Pair<ValueType> atomicConstraint(constraint[0],constraint[1]);
        problem->atomicConstraints.pushBack(&atomicConstraint);
        problem->atomicToCheck[atomicConstraint.second]->pushBack(&atomicConstraintIdx);
    }

    // Disjunctive constraints
    for (u16 disjunctiveConstraintIdx = 0; disjunctiveConstraintIdx < problemJson["DisjunctiveConstraints"].size(); disjunctiveConstraintIdx += 1)
    {
        auto& constraint = problemJson["DisjunctiveConstraints"][disjunctiveConstraintIdx];
        if(constraint[0] == constraint[2])
        {
            Triple<ValueType> disjunctiveConstraint(constraint[0],constraint[1],constraint[3]);
            problem->disjunctiveConstraints.pushBack(&disjunctiveConstraint);
            problem->disjunctive1ToCheck[disjunctiveConstraint.first]->pushBack(&disjunctiveConstraintIdx);
        }
        else
        {
            Triple<ValueType> disjunctiveConstraint(constraint[0],constraint[1],constraint[2]);
            problem->disjunctiveConstraints.pushBack(&disjunctiveConstraint);
            problem->disjunctive2ToCheck[disjunctiveConstraint.first]->pushBack(&disjunctiveConstraintIdx);
        }
    }

    // Soft atomic constraints
    for (u16 softAtomicConstraintIdx = 0; softAtomicConstraintIdx < problemJson["SoftAtomicConstraints"].size(); softAtomicConstraintIdx += 1)
    {
        auto& constraint = problemJson["SoftAtomicConstraints"][softAtomicConstraintIdx];
        Pair<ValueType> softAtomicConstraint(constraint[0],constraint[1]);
        problem->softAtomicConstraints.pushBack(&softAtomicConstraint);
        problem->softAtomicToCheck[softAtomicConstraint.first]->pushBack(&softAtomicConstraintIdx);
    }

    return problem;
}