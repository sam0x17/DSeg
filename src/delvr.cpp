#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <limits>
#include <opencv2/opencv.hpp>
#include <boost/multi_array.hpp>

extern "C" {
  #include <vl/generic.h>
  #include <vl/slic.h>
}

const int DSEG_LINE_LENGTH = 24;

class DColor {
  public:
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

class DPos {
  public:
    unsigned int x;
    unsigned int y;
};

class BSegment {
  public:
    std::vector<DPos> positions;
    int r_sum = 0;
    int g_sum = 0;
    int b_sum = 0;
    int x_sum = 0;
    int y_sum = 0;

    void add_pixel(int x, int y, int r, int g, int b) {
      DPos pos;
      pos.x = x;
      pos.y = y;
      positions.push_back(pos);
      r_sum += r;
      g_sum += g;
      b_sum += b;
      x_sum += x;
      y_sum += y;
    }
};

class DSegment {
  public:
    DColor cta_color; // closest color to average
    float line[DSEG_LINE_LENGTH]; // 1D representation of outline, slope based
};

class DSegmentationResult {
  public:
    std::unordered_map<int, BSegment> bmap;
    cv::Mat segmented_image;
};

int seg_at(vl_uint32 *segmentation, int i, int j, int cols) {
  return (int)segmentation[j + cols * i];
}

DSegmentationResult perform_segmentation(cv::Mat mat, int region, int min_region, float regularization=1500.0, bool return_image=false) {
  cv::Mat matc;
  if (return_image)
    matc = mat.clone();
  vl_uint32* segmentation = new vl_uint32[mat.rows * mat.cols];
  vl_size height = mat.rows;
  vl_size width = mat.cols;
  vl_size channels = mat.channels();
  // convert to 1 dimensional array of floats
  float* image = new float[mat.rows * mat.cols * mat.channels()];
  for(int i = 0; i < mat.rows; ++i) {
    for(int j = 0; j < mat.cols; ++j) {
      // Assuming three channels ...
      image[j + mat.cols * i + mat.cols * mat.rows * 0] = mat.at<cv::Vec3b>(i, j)[0];
      image[j + mat.cols * i + mat.cols * mat.rows * 1] = mat.at<cv::Vec3b>(i, j)[1];
      image[j + mat.cols * i + mat.cols * mat.rows * 2] = mat.at<cv::Vec3b>(i, j)[2];
    }
  }
  std::unordered_map<int, BSegment> bmap;

  // do segmentation with VLFeat SLIC
  vl_slic_segment(segmentation, image, width, height, channels, region, regularization, min_region);

  int label = 0;
  int label_top = -1;
  int label_bottom = -1;
  int label_left = -1;
  int label_right = -1;

  // record segmentation result
  for(int i = 0; i < mat.rows; i++) {
    for(int j = 0; j < mat.cols; j++) {
      label = seg_at(segmentation, i, j, mat.cols);

      if (return_image) {
        label_top = label;
        if (i > 0)
          label_top = seg_at(segmentation, i - 1, j, matc.cols);
        label_bottom = label;
        if (i < matc.rows - 1)
          label_bottom = seg_at(segmentation, i + 1, j, matc.cols);
        label_left = label;
        if (j > 0)
          label_left = seg_at(segmentation, i, j - 1, matc.cols);
        label_right = label;
        if (j < matc.cols - 1)
          label_right = seg_at(segmentation, i, j + 1, matc.cols);
        if (label != label_top || label != label_bottom || label != label_left || label != label_right) {
          matc.at<cv::Vec3b>(i, j)[0] = 0;
          matc.at<cv::Vec3b>(i, j)[1] = 0;
          matc.at<cv::Vec3b>(i, j)[2] = 255;
        }
      }

      int b = mat.at<cv::Vec3b>(i, j)[0];
      int g = mat.at<cv::Vec3b>(i, j)[1];
      int r = mat.at<cv::Vec3b>(i, j)[2];
      std::unordered_map<int, BSegment>::iterator iter = bmap.find(label);
      if (iter == bmap.end()) {
        BSegment bseg;
        bseg.add_pixel(i, j, r, g, b);
        bmap[label] = bseg;
      } else {
        BSegment *bseg = &bmap[label];
        bseg->add_pixel(i, j, r, g, b);
      }
    }
  }
  DSegmentationResult res;
  res.bmap = bmap;
  if (return_image)
    res.segmented_image = matc;
  delete image;
  delete segmentation;
  return res;
}

inline int max(int a, int b) {
  if (a >= b)
    return a;
  return b;
}

inline int min(int a, int b) {
  if (a <= b)
    return a;
  return b;
}

DSegment generate_dseg(BSegment bseg) {
  int min_x = std::numeric_limits<int>::max();
  int min_y = min_x;
  int max_x = std::numeric_limits<int>::min();
  int max_y = max_x;
  for(DPos pos : bseg.positions) {
    min_x = min(min_x, pos.x);
    min_y = min(min_y, pos.y);
    max_x = max(max_x, pos.x);
    max_y = max(max_y, pos.y);
  }
  int grid_w = 2 + max_x - min_x;
  int grid_h = 2 + max_y - min_y;

  bool *grid = new bool[grid_w * grid_h];
  for(int i = 0; i < grid_w * grid_h; i++)
    grid[i] = false;
  for(DPos pos : bseg.positions) {
    int x = pos.x - min_x;
    int y = pos.y - min_y;
    grid[grid_w * y + x] = true;
  }
  for(int x = 1; x < grid_w - 1; x++) {
    for(int y = 1; y < grid_h - 1; y++) {

    }
  }
  DSegment dseg;
  return dseg;
}

std::vector<DSegment> generate_dsegs(std::unordered_map<int, BSegment> bmap) {
  std::vector<DSegment> dsegs;
  for(auto pair : bmap) {
    generate_dseg(pair.second);
  }
  return dsegs;
}

int main(int argc, char** argv) {
  cv::Mat mat = cv::imread("0000444.png", CV_LOAD_IMAGE_COLOR);
  //cv::Mat mat = cv::imread("../carrier.jpg", CV_LOAD_IMAGE_COLOR);
  std::cout << "loaded image" << std::endl;
  for(int scale = 1; scale <= 130;) {
    DSegmentationResult res = perform_segmentation(mat, 4 + scale, (4 + scale) * 0.25, 1500.0, false);
    std::stringstream ss;
    ss << "contours_" << scale << ".png";
    std::cout << ss.str() << std::endl;
    cv::imwrite(ss.str(), res.segmented_image);
    generate_dsegs(res.bmap);
    if (scale < 40)
      scale++;
    else if (scale < 60)
      scale += 2;
    else if (scale < 100)
      scale += 5;
    else
      scale += 10;
  }
  return 0;
}
