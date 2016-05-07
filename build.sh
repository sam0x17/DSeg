#!/bin/bash
echo "compiling..."
g++ src/delvr.cpp -std=c++11 -I bin/opt/boost_1_60_0/ -I bin/vlfeat -L bin/vlfeat -I bin/opencv/include -L bin/opencv/bin -o bin/delvr
echo "executable created in bin/delvr"
