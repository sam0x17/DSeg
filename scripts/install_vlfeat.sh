#!/bin/bash
echo "downloading and installing precompiled VLFeat binaries..."
cd ../
mkdir -p bin/opt/download
mkdir -p bin/opt/vlfeat-0.9.20
mkdir -p bin/vlfeat
rm -rf bin/vlfeat
rm -rf bin/opt/vlfeat-0.9.20
cd bin/opt/download
wget -N http://www.vlfeat.org/download/vlfeat-0.9.20-bin.tar.gz || exit 1
sum=$(md5sum vlfeat-0.9.20-bin.tar.gz)
vlsum="e22ada7ddd708d739ed9958b16642ba1"
if [[ $sum = *$vlsum* ]]
  then
    echo "md5 sum was valid ($sum)"
    echo "extracting archive..."
    tar -xzvf vlfeat-0.9.20-bin.tar.gz || exit 1
    mv vlfeat-0.9.20 ../ || exit 1
    echo "done"
  else
    echo "bad checksum: $sum  expected: $vlsum"
    echo "VLFeat download was corrupt (md5 hash mismatch) -- please try running again"
    exit 1
fi
