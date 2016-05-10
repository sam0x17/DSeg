#!/bin/bash
echo "installing prerequisites..."
cd scripts
./install_packages.sh
./install_opencv.sh
./install_vlfeat.sh
./install_caffe.sh
./download_data.sh
./download_sun.sh
echo "done."
