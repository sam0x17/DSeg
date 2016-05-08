#!/bin/bash
./build.sh
cd bin
rm contours*
time ./delvr
