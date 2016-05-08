#!/bin/bash
./build.sh
cd bin
rm contours*
rm patch*
time ./delvr
