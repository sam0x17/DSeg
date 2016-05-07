#!/bin/bash
echo "installing required packages..."
sudo apt-get update
sudo apt-get install build-essential pigz cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev p7zip-full nvidia-cuda-toolkit
