#!/bin/bash
echo "installing required packages..."
sudo apt-get update
echo "installing boost..."
sudo apt-get install --no-install-recommends libboost-all-dev
echo "installing VLFeat and OpenCV prerequisites..."
sudo apt-get install libgtk2.0-dev pkg-config build-essential pigz cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev p7zip-full
echo "installing Caffe prerequisites..."
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libgflags-dev libgoogle-glog-dev liblmdb-dev libatlas-base-dev
