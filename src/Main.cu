#include <thread>
#include <External/AnyOption/anyoption.h>
#include <Utils/Algorithms.cuh>
#include <Utils/Chrono.cuh>
#include <DynamicProgramming/CTWPModel.cuh>
#include <DynamicProgramming/OSSPModel.cuh>
#include <DynamicProgramming/SOPModel.cuh>
#include <LargeNeighborhoodSearch/SearchManagerCPU.cuh>
#include <LargeNeighborhoodSearch/SearchManagerGPU.cuh>
#include <LargeNeighborhoodSearch/StatesPriorityQueue.cuh>
#include <LargeNeighborhoodSearch/SyncState.cuh>
#include <Options.h>

using namespace std;
using namespace Memory;
using namespace Chrono;
using namespace DecisionDiagram;
using namespace DynamicProgramming;
using namespace OptimizationProblems;
using namespace LargeNeighborhoodSearch;
using ProblemType = OSSProblem;
using StateType = OSSPState;

// Auxiliary functions
AnyOption* parseOptions(int argc, char* argv[]);

void configGPU();

template<typename ProblemType, typename StateType>
void printStatistics(Options const * options, u64 startTime, SearchManagerCPU<ProblemType,StateType>* searchManagerCpu, SearchManagerGPU<ProblemType,StateType>* searchManagerGpu);

template<typename ProblemType, typename StateType>
void checkBetterSolutions(Options const * options, StateType* bestSolution, u64 startTime, SearchManagerCPU<ProblemType, StateType>* searchManagerCpu, SearchManagerGPU<ProblemType, StateType>* searchManagerGpu);

bool checkInitTimeout(u64 startTime);

bool checkTotalTimeout(u64 startTime, Options const * options);

// Debug
void printElapsedTime(u64 elapsedTimeMs);

void clearLine();

int main(int argc, char* argv[])
{
    u64 const startTime = now();
    
    // Options parsing
    byte* memory = nullptr;
    memory = safeMalloc(sizeof(Options), MallocType::Std);
    Options* optionsCpu = new (memory) Options();
    if (not optionsCpu->parseOptions(argc, argv))
    {
        optionsCpu->printUsage();
        return EXIT_FAILURE;
    }
    else
    {
        optionsCpu->printOptions();
    }

    printf("[INFO] Initializing data structures");
    printf(" | Time: ");
    printElapsedTime(now() - startTime);
    printf("\n");

    // Context initialization
    MallocType mallocTypeGpu = MallocType::Std;
    if (optionsCpu->mddsGpu > 0)
    {
        configGPU();
        mallocTypeGpu = MallocType::Managed;
    };

    memory = safeMalloc(sizeof(Options), mallocTypeGpu);
    Options* optionsGpu = new (memory) Options();
    optionsGpu->parseOptions(argc, argv);

    //Problem
    ProblemType* const problemCpu = parseInstance<ProblemType>(optionsCpu->inputFilename, MallocType::Std);
    ProblemType* const problemGpu = parseInstance<ProblemType>(optionsGpu->inputFilename, mallocTypeGpu);

    // Queue
    StatesPriorityQueue<StateType> statesPriorityQueue(problemCpu, optionsCpu->queueSize);

    // Search managers
    memory = nullptr;
    memory = safeMalloc(sizeof(SearchManagerCPU<ProblemType,StateType>), MallocType::Std);
    SearchManagerCPU<ProblemType,StateType>* searchManagerCpu = new (memory) SearchManagerCPU<ProblemType, StateType>(problemCpu, optionsCpu);
    memory = safeMalloc(sizeof(SearchManagerGPU<ProblemType,StateType>), mallocTypeGpu);
    SearchManagerGPU<ProblemType,StateType>* searchManagerGpu = new (memory) SearchManagerGPU<ProblemType, StateType>(problemGpu, optionsGpu, mallocTypeGpu);

    // Solutions
    memory = safeMalloc(sizeof(StateType), MallocType::Std);
    StateType* bestSolution = new (memory) StateType(problemCpu, MallocType::Std);
    bestSolution->invalidate();

    // Root
    memory = safeMalloc(sizeof(StateType), mallocTypeGpu);
    StateType* root = new (memory) StateType(problemCpu, mallocTypeGpu);
    makeRoot(problemCpu, root);
    statesPriorityQueue.insert(root);

    //Random
    printf("[INFO] Initializing random engines");
    printf(" | Time: ");
    printElapsedTime(now() - startTime);
    printf("\n");
    std::thread initRandomCpu(&SearchManagerCPU<ProblemType,StateType>::initializeRandomEngines, searchManagerCpu);
    std::thread initRandomGpu(&SearchManagerGPU<ProblemType,StateType>::initializeRandomEngines, searchManagerGpu);
    initRandomCpu.join();
    initRandomGpu.join();

    //Initial search
    bool timeout = false;
    printf("[INFO] Start initial search");
    printf(" | Time: ");
    printElapsedTime(now() - startTime);
    printf("\n");

    std::thread searchInitCpu(&SearchManagerCPU<ProblemType,StateType>::searchInitLoop, searchManagerCpu, &statesPriorityQueue, &timeout);
    searchInitCpu.detach();

    while(not searchManagerCpu->done)
    {
        printStatistics(optionsCpu, startTime, searchManagerCpu, searchManagerGpu);
        checkBetterSolutions(optionsCpu, bestSolution, startTime, searchManagerCpu, searchManagerGpu);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
        timeout = checkInitTimeout(startTime);
    }
    checkBetterSolutions(optionsCpu, bestSolution, startTime,  searchManagerCpu, searchManagerGpu);

    if(bestSolution->isValid())
    {
        //LargeNeighborhoodSearch search
        timeout = false;
        if(optionsCpu->statistics)
        {
            clearLine();
        }
        printf("[INFO] Switch to LargeNeighborhoodSearch");
        printf(" | Time: ");
        printElapsedTime(now() - startTime);
        printf("\n");

        std::thread searchLnsCpu(& SearchManagerCPU<ProblemType, StateType>::searchLnsLoop, searchManagerCpu, root, &timeout);
        searchLnsCpu.detach();
        std::thread searchLnsGpu(& SearchManagerGPU<ProblemType, StateType>::searchLnsLoop, searchManagerGpu, root, &timeout);
        searchLnsGpu.detach();

        while (not (searchManagerCpu->done and searchManagerGpu->done))
        {
            printStatistics(optionsCpu, startTime, searchManagerCpu, searchManagerGpu);
            checkBetterSolutions(optionsCpu, bestSolution, startTime, searchManagerCpu, searchManagerGpu);
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            timeout = checkTotalTimeout(startTime, optionsCpu);
        }
        checkBetterSolutions(optionsCpu, bestSolution, startTime, searchManagerCpu, searchManagerGpu);
    }

    if(optionsCpu->mddsGpu > 0)
    {
        cudaDeviceReset();
    }

    return EXIT_SUCCESS;
}

void configGPU()
{
    //Heap
    std::size_t const sizeHeap = 4ul * 1024ul * 1024ul * 1024ul;
    cudaDeviceSetLimit(cudaLimitMallocHeapSize, sizeHeap);

    //Stack
    size_t const sizeStackThread = 4 * 1024;
    cudaDeviceSetLimit(cudaLimitStackSize, sizeStackThread);

    //L1 Cache
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
}


void printElapsedTime(u64 elapsedTimeMs)
{
    u64 ms = elapsedTimeMs;

    u64 const h = ms / (1000 * 60 * 60);
    ms -= h * 1000 * 60 * 60;

    u64 const m = ms / (1000 * 60);
    ms -= m * 1000 * 60;
    
    u64 const s = ms / 1000;
    ms -= s * 1000;

    printf("%02lu:%02lu:%02lu.%03lu", h, m, s, ms);
}

void clearLine()
{
    // ANSI clear line escape code
    printf("\33[2K\r");
}

template<typename ProblemType, typename StateType>
void printStatistics(Options const * options, u64 startTime, SearchManagerCPU<ProblemType,StateType>* searchManagerCpu, SearchManagerGPU<ProblemType,StateType>* searchManagerGpu)
{
    if(options->statistics)
    {
        clearLine();
        printf("[INFO] Time: ");
        printElapsedTime(now() - startTime);

        if(options->mddsCpu > 0 and searchManagerCpu->bestSolution.state.isValid())
        {
            printf(" | CPU: %d - %lu - %lu MDD/s", searchManagerCpu->bestSolution.state.cost, searchManagerCpu->iteration, searchManagerCpu->speed);
        }
        else
        {
            printf(" | CPU: * - %lu - %lu MDD/s", searchManagerCpu->iteration, searchManagerCpu->speed);
        }

        if (options->mddsGpu > 0 and searchManagerGpu->bestSolution.state.isValid())
        {
            printf(" | GPU: %d - %lu - %lu MDD/s", searchManagerGpu->bestSolution.state.cost, searchManagerGpu->iteration, searchManagerGpu->speed);
        }
        else
        {
            printf(" | GPU: * - %lu - %lu MDD/s", searchManagerGpu->iteration, searchManagerGpu->speed);
        }

        printf("\r");
        fflush(stdout);
    }
}

template<typename ProblemType, typename StateType>
void checkBetterSolutions(Options const * options, StateType* bestSolution, u64 startTime, SearchManagerCPU<ProblemType, StateType>* searchManagerCpu, SearchManagerGPU<ProblemType, StateType>* searchManagerGpu)
{
    bool const betterSolutionFromCpu =
        searchManagerCpu->bestSolution.state.isValid() and
        searchManagerCpu->bestSolution.state.cost < bestSolution->cost and
        searchManagerCpu->bestSolution.state.cost < searchManagerGpu->bestSolution.state.cost;

    bool const betterSolutionFromGpu =
        searchManagerGpu->bestSolution.state.isValid() and
        searchManagerGpu->bestSolution.state.cost < bestSolution->cost and
        searchManagerGpu->bestSolution.state.cost < searchManagerCpu->bestSolution.state.cost;

    if (betterSolutionFromCpu or betterSolutionFromGpu)
    {
        if(betterSolutionFromCpu)
        {
            searchManagerCpu->bestSolution.mutex.lock();
            *bestSolution = searchManagerCpu->bestSolution.state;
            searchManagerCpu->bestSolution.mutex.unlock();
        }
        else
        {
            searchManagerGpu->bestSolution.mutex.lock();
            *bestSolution = searchManagerGpu->bestSolution.state;
            searchManagerGpu->bestSolution.mutex.unlock();
        }

        searchManagerCpu->neighborhoodSolution.mutex.lock();
        searchManagerCpu->neighborhoodSolution.state = *bestSolution;
        searchManagerCpu->neighborhoodSolution.mutex.unlock();

        searchManagerGpu->neighborhoodSolution.mutex.lock();
        searchManagerGpu->neighborhoodSolution.state = *bestSolution;
        searchManagerGpu->neighborhoodSolution.mutex.unlock();


        if (options->statistics)
        {
            clearLine();
        }
        printf("[SOLUTION] Source: ");
        if (options->statistics)
        {
            printf("%s", betterSolutionFromCpu ? "\033[30;44mCPU\033[0m" : "\033[30;42mGPU\033[0m");
        }
        else
        {
            printf("%s", betterSolutionFromCpu ? "CPU" : "GPU");
        }
        printf(" | Time: ");
        printElapsedTime(now() - startTime);
        if (options->statistics)
        {
            printf(" | Cost: \033[30;43m%u\033[0m", bestSolution->cost);
        }
        else
        {
            printf(" | Cost: %u", bestSolution->cost);
        }
        printf(" | Solution: ");
        bestSolution->print(true);
        fflush(stdout);
    }
}

bool checkInitTimeout(u64 startTime)
{
    // 10s timeout
    return  (now() - startTime) > ( 10 * 1000);
}

bool checkTotalTimeout(u64 startTime, Options const* options)
{
    return  (now() - startTime) > (options->timeout * 1000);
}