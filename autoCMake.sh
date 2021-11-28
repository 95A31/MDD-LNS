#!/bin/bash

function printHelp 
{
    echo "Usage: ./autoCMake.sh <Build type> <Build directory>"
    echo "  Build types:"
    echo "    -d  Debug"
    echo "    -r  Release"
    echo "Example: ./autoCMake.sh -d ./build"
}

if [ "$#" -ne 2 ];
then
    printHelp
    exit 1
fi

buildType=""
for arg in $1
do
    case "$arg" in

        -r) 
            buildType="Release"
            ;;
        -d) 
            buildType="Debug"
            ;;
        -h|--help) 
            printHelp
            exit 0
            ;;
        *)
            echo "[ERROR] Unrecognized argument: $arg"
            printHelp
            exit 1
            ;;
    esac
done

buildDir=$2
rm -rf $buildDir 2> /dev/null
mkdir -p $buildDir
cmake -B $buildDir -D CMAKE_BUILD_TYPE=$buildType
