#!/bin/bash
echo "retrieving SUN2012 dataset..."
cd ../
mkdir -p bin/data
mkdir -p bin/opt/download
mkdir -p bin/data/SUN2012
echo "deleting any existing data..."
rm -rf bin/data/SUN2012
cd bin/opt/download
echo "downloading data"
wget -N http://groups.csail.mit.edu/vision/SUN/releases/SUN2012.tar.gz
echo "verifying download integrity..."
correct="5188e9503debd4511fa45cc856fb46ce"
sum=$(md5sum SUN2012.tar.gz)
if [[ $sum = *$correct* ]]
  then
    echo "md5 sum was valid ($sum)"
    echo "extracting data..."
    tar -xzvf SUN2012.tar.gz -C ../../data
    echo "dataset downloaded to bin/data/SUN2012"
  else
    echo "dataset download was corrupt (md5 hash mismatch) -- please try running again"
    exit 1
fi
