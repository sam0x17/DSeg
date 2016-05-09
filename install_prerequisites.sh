#!/bin/bash
echo "installing prerequisites..."
cd scripts
./install_packages.sh
./install_opencv.sh
./install_vlfeat.sh
./download_data.sh
echo "done."
