#!/bin/bash
echo "fetching gSLICr fork submodule..."
cd ../
#git submodule init
#git submodule update
echo "compiling gSLICr fork..."
cd gSLICr || exit 1
mkdir -p build || exit 1
rm -rf build/*
cd build || exit 1
echo "running CMake..." || exit 1
cmake ../ || exit 1
echo "running make..." || exit 1
make -j4 || exit 1
echo "done"
