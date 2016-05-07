#!/bin/bash
echo "initializing submodules (VLFeat)..."
cd ../
git submodule init
git submodule update
echo "compiling VLFeat.."
cd vlfeat || exit 1
make -j8 || exit 1
echo "done"
