cmake_minimum_required(VERSION 3.18)

project(mdd-lns LANGUAGES CXX)

# CUDA
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 86)
string(APPEND CMAKE_CUDA_FLAGS " --expt-extended-lambda")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
string(APPEND CMAKE_CUDA_FLAGS " -Xcompiler -Wall")
string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -G")
string(APPEND CMAKE_CUDA_FLAGS_RELEASE " -lineinfo")

# Check CUDA and GCC version
set(CUDA_REQUIRED_VERSION 11)
set(GCC_REQUIRED_VERSION 9)
if (NOT ((CMAKE_CUDA_COMPILER_ID STREQUAL "NVIDIA") AND (CMAKE_CUDA_COMPILER_VERSION MATCHES "^${CUDA_REQUIRED_VERSION}*")))
    message(FATAL_ERROR "Required CUDA ${CUDA_REQUIRED_VERSION}")
endif ()
if (NOT ((CMAKE_CXX_COMPILER_ID STREQUAL "GNU") AND (CMAKE_CXX_COMPILER_VERSION MATCHES "^${GCC_REQUIRED_VERSION}*")))
    message(FATAL_ERROR "Required GCC ${GCC_REQUIRED_VERSION}")
endif ()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Add sources
set(SOURCES
    src/Main.cu
    src/OptimizationProblems/Variable.cuh
    src/DynamicProgramming/CTWPModel.cuh
    src/DynamicProgramming/SOPModel.cuh
    src/DecisionDiagram/MDD.cuh
    src/OptimizationProblems/CTWProblem.cuh
    src/OptimizationProblems/SOProblem.cuh
    src/OptimizationProblems/Problem.cuh
    src/DynamicProgramming/State.cuh
    src/LargeNeighborhoodSearch/OffloadBuffer.cuh
    src/LargeNeighborhoodSearch/SearchManagerCPU.cuh
    src/LargeNeighborhoodSearch/SearchManagerGPU.cuh
    src/LargeNeighborhoodSearch/Neighbourhood.cuh
    src/LargeNeighborhoodSearch/SyncState.cuh
    src/OptimizationProblems/Context.h
    src/DynamicProgramming/Context.h
    src/DecisionDiagram/StateMetadata.cuh
    src/LargeNeighborhoodSearch/StatesPriorityQueue.cuh
    src/DecisionDiagram/Context.h
    include/Utils/Algorithms.cuh
    include/Containers/Pair.cuh
    src/Options.h
    src/DynamicProgramming/CTWPState.cuh
    src/DynamicProgramming/SOPState.cuh
    src/OptimizationProblems/OSSProblem.cuh
    src/DynamicProgramming/OSSPState.cuh
    src/DynamicProgramming/OSSPModel.cuh)


# Add include path
include_directories(src)
include_directories(include)

# Add target executable
add_executable(mdd-lns ${SOURCES})