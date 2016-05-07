#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <boost/multi_array.hpp>

extern "C" {
    #include "generic.h"
    #include "slic.h"
}

typedef boost::multi_array<float, 3> delvr_img;
typedef delvr_img::index ind;
const ind R = 0;
const ind G = 1;
const ind B = 2;
const ind A = 3;

typedef boost::multi_array<vl_uint32, 2> delvr_seg;
typedef delvr_seg::index seg_ind;

float *img_data(delvr_img &img) {
  return (float *)img.data();
}

vl_uint32 *seg_data(delvr_seg &seg) {
  return (vl_uint32 *)seg.data();
}

void destroy_img(delvr_img &img) {
  img.resize(boost::extents[0][0][0]);
}

void destroy_seg(delvr_seg &seg) {
  seg.resize(boost::extents[0][0]);
}

int main(int argc, char** argv) {
  std::cout << "Delvr 1.0 Loaded." << std::endl;
  std::cout << "Copyright (C) Sam Kelly -- all rights reserved" << std::endl;
  std::cout << std::endl;
  int w = 1000;
  int h = 1000;
  delvr_img img(boost::extents[w][h][4]);
  for(ind x = 0; x < 1000; x++) {
    for(ind y = 0; y < 1000; y++) {
      img[x][y][R] = 0.4;
      img[x][y][G] = 0.6;
      img[x][y][B] = 0.8;
      img[x][y][A] = 0.01;
    }
  }
  delvr_seg seg(boost::extents[w][h]);
  vl_slic_segment (seg_data(seg),
                   img_data(img),
                   (vl_size)w,
                   (vl_size)h,
                   (vl_size)4,
                   (vl_size)16,
                   0.5,
                   (vl_size)5);
  destroy_img(img);
  destroy_seg(seg);
  return 0;
}

/*delvr_img delvr_img_init(unsigned int width, unsigned int height, unsigned int num_channels) {
  delvr_img4 img;
  img.width = width;
  img.height = height;
  img.data = malloc(sizeof(float) * width * height * num_channels);
  for(int x = 0; x < width; x++) {
    for(int y = 0; y < height; y++) {
    }
  }
}*/
