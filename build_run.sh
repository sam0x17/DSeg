#!/bin/bash
./build.sh
cd bin

# feature generation (positive instances)
#./delvr genfeats translucent data/imgs/hamina\ 128x128\ m-30\ crop-t hamina.dfeats
#./delvr genfeats translucent data/imgs/m4a1_s\ 128x128\ m-30\ crop-t m4a1.dfeats
#./delvr genfeats translucent data/imgs/halifax\ 128x128\ m-30\ crop-t halifax.dfeats
#./delvr genfeats translucent data/imgs/kirov\ 128x128\ m-30\ crop-t kirov.dfeats
#./delvr genfeats translucent data/imgs/kuznet\ 128x128\ m-30\ crop-t kuznet.dfeats
#./delvr genfeats translucent data/imgs/udaloy\ 128x128\ m-30\ crop-t udaloy.dfeats
#./delvr genfeats translucent data/imgs/sovddg\ 128x128\ m-30\ crop-t sovddg.dfeats

# feature generation (negative instances)
./delvr genfeats opaque data/SUN2012/Images sun_negatives.dfeats

# detector training
#./delvr traindetector data/hamina.dfeats data/null data/hamina_detector.brain
