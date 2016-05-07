#!/bin/bash
echo "retrieving updated dataset..."
cd ../
mkdir -p bin/data
mkdir -p bin/opt/download
mkdir -p bin/data/imgs
mkdir -p bin/data/test_imgs
echo "deleting any existing data..."
rm -rf bin/data/imgs
rm -rf bin/data/test_imgs
cd bin/opt/download
echo "downloading data"
wget -N https://storage.googleapis.com/durosoft-shared/delvr/data.tar.gz
echo "verifying download integrity..."
correct="1478b832c1c00ec357842b8f81c32c8b"
sum=$(md5sum data.tar.gz)
if [[ $sum = *$correct* ]]
  then
    echo "md5 sum was valid ($sum)"
    echo "extracting data..."
    tar -xzvf data.tar.gz -C ../../data
    echo "dataset downloaded to bin/data"
  else
    echo "dataset download was corrupt (md5 hash mismatch) -- please try running again"
    exit 1
fi
