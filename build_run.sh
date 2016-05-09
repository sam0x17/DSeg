#!/bin/bash
./build.sh
cd bin
#rm contours*
#rm patch*
./delvr genfeats translucent data/imgs/hamina\ 128x128\ m-30\ crop-t hamina.dfeats
