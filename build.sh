#!/bin/bash
echo "compiling..."
g++ src/delvr.cpp -o bin/delvr -std=c++11 -I bin/opt/boost_1_60_0/ -I bin/opencv/include -I bin/opt/vlfeat-0.9.20 -L bin/opencv/lib -L bin/opt/vlfeat-0.9.20/bin/glnxa64 -lvl -lm -lopencv_cudabgsegm -lopencv_cudaobjdetect -lopencv_cudastereo -lopencv_shape -lopencv_stitching -lopencv_cudafeatures2d -lopencv_superres -lopencv_cudacodec -lopencv_videostab -lopencv_cudaoptflow -lopencv_cudalegacy -lopencv_calib3d -lopencv_features2d -lopencv_objdetect -lopencv_highgui -lopencv_videoio -lopencv_photo -lopencv_imgcodecs -lopencv_cudawarping -lopencv_cudaimgproc -lopencv_cudafilters -lopencv_video -lopencv_ml -lopencv_imgproc -lopencv_flann -lopencv_cudaarithm -lopencv_core -lopencv_cudev
echo "executable created in bin/delvr"
