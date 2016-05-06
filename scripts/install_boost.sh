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
mkdir -p bin/opt/boost_1_60_0
mkdir -p bin/boost

# wipe out existing boost installation
rm bin/opt/boost* -rf
rm bin/opt/download/boost*
rm bin/boost -rf

mkdir -p bin/boost

# download, build, and install boost
echo "downloading Boost 1.60.0..."
cd bin/opt/download
wget http://iweb.dl.sourceforge.net/project/boost/boost/1.60.0/boost_1_60_0.tar.bz2
fname="boost_1_60_0.tar.bz2"
sum=$(md5sum $fname)
boostsum="65a840e1a0b13a558ff19eeb2c4f0cbe"
if [[ $sum = *$boostsum* ]]
  then
    echo "md5 sum was valid ($sum)"
    echo "extracting archive..."
    tar -xjf $fname -C ../
    echo "building Boost 1.60.0..."
    cd ../boost_1_60_0
    sed -e '1 i#ifndef Q_MOC_RUN' \
        -e '$ a#endif'            \
        -i boost/type_traits/detail/has_binary_operator.hpp &&
    ./bootstrap.sh --prefix=$bindir/boost &&
    ./b2 -j12 stage threading=multi link=shared --stagedir=$bindir/boost
  else
    echo "boost download was corrupt (md5 hash mismatch) -- please try running again"
    exit 1
fi
echo "Bost 1.60.0 installed in $bindir/opt/boost_1_60_0"
