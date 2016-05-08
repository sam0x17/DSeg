#!/bin/bash
./build.sh
cd bin
time ./delvr
gnome-open contours.png
