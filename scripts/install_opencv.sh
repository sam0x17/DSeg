#!/bin/bash

# make bin and bin/opt directories
cd ..
echo Making directories...
maindir=$(pwd)
mkdir -p bin
cd bin
bindir=$(pwd)
cd ..
mkdir -p bin/opt
mkdir -p bin/opt/download
mkdir -p bin/opt/opencv
mkdir -p bin/opencv

# wipe out existing opencv installation
rm bin/opt/opencv* -rf
rm bin/opt/download/opencv* -rf
rm bin/opencv -rf

mkdir -p bin/opencv

# download, build, and install opencv
echo "downloading OpenCV 3.1.0..."
cd bin/opt/download
wget https://github.com/Itseez/opencv/archive/3.1.0.zip
fname="3.1.0.zip"
sum=$(md5sum $fname)
opensum="6082ee2124d4066581a7386972bfd52a"
if [[ $sum = *$opensum* ]]
  then
    echo "md5 sum was valid ($sum)"
    echo "extracting archive..."
    unzip $fname -d ../ || exit 1
    echo "building OpenCV 3.1.0..."
    cd ../opencv-3.1.0 || exit 1
    cmake -D CMAKE_BUILD_TYPE=Release -D CMAKE_INSTALL_PREFIX=$bindir/opencv . || exit 1
    make -j12 || exit 1
    make install
  else
    echo "OpenCV download was corrupt (md5 hash mismatch) -- please try running again"
    exit 1
fi
echo "OpenCV installed in $bindir/opt/opencv"
