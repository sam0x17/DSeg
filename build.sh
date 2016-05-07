#!/bin/bash
echo "compiling..."
g++ src/delvr.cpp -o bin/delvr -std=c++11 -I bin/opt/boost_1_60_0/ -I bin/opencv/include -I bin/opt/vlfeat-0.9.20 -L bin/opencv/lib -L bin/opt/vlfeat-0.9.20/bin/glnxa64 -lvl -lm
echo "executable created in bin/delvr"
