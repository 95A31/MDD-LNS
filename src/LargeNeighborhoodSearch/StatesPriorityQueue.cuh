#pragma once

#include <mutex>
#include <Containers/Buffer.cuh>
#include <Containers/LightMinMaxHeap.cuh>

template<typename StateType>
class StatesPriorityQueue
{
    // Members
    public:
    std::mutex mutex;
    private:
    Buffer<StateType> statesBuffer;
    LightMinMaxHeap<StateType> statesMinMaxHeap;

    // Functions
    public:
    template<typename ProblemType>
    StatesPriorityQueue(ProblemType const * problem, u32 capacity);
    bool isEmpty() const;
    StateType const * getMin() const;
    StateType const * getMax() const;
    void popMin();
    void popMax();
    void insert(StateType const * state);
    bool isFull() const;
    u32 getSize() const;
    void clear();


};

template<typename StateType>
template<typename ProblemType>
StatesPriorityQueue<StateType>::StatesPriorityQueue(ProblemType const * problem, u32 capacity) :
    statesBuffer(capacity, Memory::MallocType::Std),
    statesMinMaxHeap(capacity, Memory::MallocType::Std)
{
    // States
    std::byte* storage = StateType::mallocStorages(problem, capacity, Memory::MallocType::Std);
    for (u32 stateIdx = 0; stateIdx < statesBuffer.getCapacity(); stateIdx += 1)
    {
        new (statesBuffer[stateIdx]) StateType(problem, storage);
        storage = Memory::align(statesBuffer[stateIdx]->endOfStorage(), Memory::DefaultAlignment);
    }
}

template<typename StateType>
StateType const * StatesPriorityQueue<StateType>::getMin() const
{
    return statesMinMaxHeap.getMin();
}

template<typename StateType>
StateType const * StatesPriorityQueue<StateType>::getMax() const
{
    return statesMinMaxHeap.getMax();
}

template<typename StateType>
u32 StatesPriorityQueue<StateType>::getSize() const
{
    return statesBuffer.getSize();
}

template<typename StateType>
void StatesPriorityQueue<StateType>::insert(StateType const * state)
{
    StateType* bufferedState = statesBuffer.insert(state);
    statesMinMaxHeap.insert(bufferedState);
}

template<typename StateType>
bool StatesPriorityQueue<StateType>::isEmpty() const
{
    return statesMinMaxHeap.isEmpty();
}

template<typename StateType>
bool StatesPriorityQueue<StateType>::isFull() const
{
    return statesMinMaxHeap.isFull();
}

template<typename StateType>
void StatesPriorityQueue<StateType>::popMin()
{
    StateType const * const min = statesMinMaxHeap.getMin();
    statesBuffer.erase(min);
    statesMinMaxHeap.popMin();
}

template<typename StateType>
void StatesPriorityQueue<StateType>::popMax()
{
    StateType const * const max = statesMinMaxHeap.getMax();
    statesBuffer.erase(max);
    statesMinMaxHeap.popMax();
}
template<typename StateType>
void StatesPriorityQueue<StateType>::clear()
{
    statesBuffer.clear();
    statesMinMaxHeap.clear();
}


