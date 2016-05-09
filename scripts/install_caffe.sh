#!/bin/bash

# make bin and bin/opt directories
cd ..
echo Making directories...
maindir=$(pwd)
mkdir -p bin
cd bin
bindir=$(pwd)
cd ..
mkdir -p bin/caffe
mkdir -p bin/opt/caffe
rm -rf bin/opt/caffe
rm -rf bin/caffe
mkdir -p bin/caffe

echo "downloading caffe..."
# download, build, and install caffe
cd bin/opt || exit 1
git clone git@github.com:BVLC/caffe.git || exit 1
echo "installing caffe..."
cd caffe || exit 1
cp $maindir/scripts/Makefile.config.caffe ./Makefile.config || exit 1
make all -j12 || exit 1
echo "done compiling caffee library, running make distribute..."
make distribute || exit 1
cp -r distribute/* ../../caffe/ || exit 1
echo "done"
