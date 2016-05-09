#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <limits>
#include <opencv2/opencv.hpp>
#include <boost/multi_array.hpp>
#include <math.h>

extern "C" {
  #include <vl/generic.h>
  #include <vl/slic.h>
}

const int DSEG_GRID_SIZE = 512;

inline float sq(float n) {
  return n * n;
}

float dist3d(float x1, float y1, float z1, float x2, float y2, float z2) {
  return sqrt(sq(x1 - x2) + sq(y1 - y2) + sq(z1 - z2));
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

void fill_magic_pink(cv::Mat &mat) {
  for(int x = 0; x < mat.cols; x++) {
    for(int y = 0; y < mat.rows; y++) {
      if(mat.at<cv::Vec3b>(x, y)[0] == 0 &&
         mat.at<cv::Vec3b>(x, y)[1] == 0 &&
         mat.at<cv::Vec3b>(x, y)[2] == 0)
      {
        mat.at<cv::Vec3b>(x, y)[0] = 255;
        mat.at<cv::Vec3b>(x, y)[1] = 0;
        mat.at<cv::Vec3b>(x, y)[2] = 255;
      }
    }
  }
}

class DColor {
  public:
    unsigned char r;
    unsigned char g;
    unsigned char b;
};

inline bool is_magic_pink(DColor color) {
  return color.r == 255 && color.g == 0 && color.b == 255;
}

class DPos {
  public:
    unsigned int x;
    unsigned int y;
};

DPos make_pos(int x, int y) {
  DPos pos;
  pos.x = x;
  pos.y = y;
  return pos;
}

class BSegment {
  public:
    int id;
    int r_sum = 0;
    int g_sum = 0;
    int b_sum = 0;
    int x_sum = 0;
    int y_sum = 0;
    DPos center_pos;
    DColor main_color;
    DColor avg_color;
    std::vector<DPos> positions;
    std::vector<DColor> colors;

    void add_pixel(int x, int y, int r, int g, int b) {
      DColor color;
      color.r = r;
      color.g = g;
      color.b = b;
      if (is_magic_pink(color))
        return;
      DPos pos;
      pos.x = x;
      pos.y = y;
      positions.push_back(pos);
      r_sum += r;
      g_sum += g;
      b_sum += b;
      x_sum += x;
      y_sum += y;
      colors.push_back(color);
    }

    void compute_averages() {
      if (positions.size() == 0)
        return;
      // calculate center_pos
      center_pos.x = x_sum / positions.size();
      center_pos.y = y_sum / positions.size();
      // find color closest to average color
      avg_color.r = r_sum / positions.size();
      avg_color.g = g_sum / positions.size();
      avg_color.b = b_sum / positions.size();
      float closest_dist = std::numeric_limits<float>::max();
      for(DColor color : colors) {
        float dist = dist3d(color.r, color.g, color.b, avg_color.r, avg_color.g, avg_color.b);
        if (dist < closest_dist) {
          closest_dist = dist;
          main_color = color;
        }
      }
    }
};

class DSegment {
  public:
    DColor cta_color; // closest color to average
    cv::Mat patch_orig;
    cv::Mat patch_resized;
};

class DSegmentationResult {
  public:
    std::unordered_map<int, BSegment> bmap;
    cv::Mat segmented_image;
};

int seg_at(vl_uint32 *segmentation, int i, int j, int cols) {
  return (int)segmentation[j + cols * i];
}

inline DPos project_point(int dest_width, int dest_height, int x, int y, float xmod, float ymod) {
  DPos pos;
  pos.x = min(max(round((float)x * xmod), 0), dest_width - 1);
  pos.y = min(max(round((float)y * ymod), 0), dest_height - 1);
  return pos;
}

inline bool in_bounds(int w, int h, int x, int y) {
  return x >= 0 && y >= 0 && x < w && y < h;
}

cv::Mat resize_contour(cv::Mat &src, int dest_width, int dest_height) {
  cv::Mat dest = cv::Mat(dest_width, dest_height, cv::DataType<unsigned char>::type);
  //cv::Mat src = shift_image(sr);
  float xmod = ((float)dest_width) / (float)src.cols;
  float ymod = ((float)dest_height) / (float)src.rows;
  for(int x = 0; x < dest_width; x++)
    for(int y = 0; y < dest_height; y++)
      dest.at<unsigned char>(x, y) = 255;
  for(int x = 0; x < src.cols; x++) {
    for(int y = 0; y < src.rows; y++) {
      if (src.at<unsigned char>(x, y) != 0)
        continue;
      DPos dest_pos = project_point(dest_width, dest_height, x, y, xmod, ymod);
      DPos border_cands[8] = {make_pos(x    , y - 1),  // TOP
                              make_pos(x + 1, y - 1),  // TOP RIGHT
                              make_pos(x + 1, y    ),  // RIGHT
                              make_pos(x + 1, y + 1),  // BOTTOM RIGHT
                              make_pos(x    , y + 1),  // BOTTOM
                              make_pos(x - 1, y + 1),  // BOTTOM LEFT
                              make_pos(x - 1, y    ),  // LEFT
                              make_pos(x - 1, y - 1)}; // TOP LEFT
      std::vector<DPos> border;
      for(int i = 0; i < 8; i++) {
        DPos pos = border_cands[i];
        if (in_bounds(src.cols, src.rows, pos.x, pos.y) && src.at<unsigned char>(pos.x, pos.y) == 0)
          border.push_back(project_point(dest_width, dest_height, pos.x, pos.y, xmod, ymod));
      }
      if (border.size() > 0)
        border.push_back(border[0]); // enclose
      std::vector<cv::Point> pts;
      for(DPos pos : border) {
        pts.push_back(cv::Point(pos.x, pos.y));
      }
      if (border.size() > 1) {
        std::vector<std::vector<cv::Point>> polys;
        polys.push_back(pts);
        cv::fillPoly(dest, polys, 120);
      }
    }
  }

  for(int x = 0; x < src.cols; x++) {
    for(int y = 0; y < src.rows; y++) {
      if (src.at<unsigned char>(x, y) == 0) {
        DPos proj = project_point(dest_width, dest_height, x, y, xmod, ymod);
        //dest.at<unsigned char>(proj.x, proj.y) = 0;
        cv::drawMarker(dest, cv::Point(proj.x, proj.y), 0);
      }
    }
  }
  return dest;
}

DSegmentationResult perform_segmentation(cv::Mat mat, int region, int min_region, float regularization=800.0, bool return_image=false) {
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
      // create bsegs
      int b = mat.at<cv::Vec3b>(i, j)[0];
      int g = mat.at<cv::Vec3b>(i, j)[1];
      int r = mat.at<cv::Vec3b>(i, j)[2];
      std::unordered_map<int, BSegment>::iterator iter = bmap.find(label);
      if (iter == bmap.end()) {
        BSegment bseg;
        bseg.add_pixel(i, j, r, g, b);
        bseg.id = label;
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
  delete[] image;
  delete[] segmentation;
  return res;
}

DSegment generate_dseg(BSegment bseg, int num=0) {
  DSegment dseg;
  if (bseg.positions.size() == 0)
    return dseg;
  DColor main_color = bseg.main_color;
  DColor avg_color = bseg.avg_color;
  std::cout << "main color: " << (int)main_color.r << ", " << (int)main_color.g << ", " << (int)main_color.b << std::endl;
  std::cout << " avg color: " << (int)avg_color.r << ", " << (int)avg_color.g << ", " << (int)avg_color.b << std::endl;
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
  int w = max_x - min_x + 1;
  int h = max_y - min_y + 1;
  cv::Mat patch = cv::Mat(w, h, cv::DataType<unsigned char>::type);
  for(int x = 0; x < w; x++)
    for(int y = 0; y < h; y++)
      patch.at<unsigned char>(x, y) = 255;
  std::cout << "w: " << w << "  h: " << h << std::endl;
  for(DPos pos : bseg.positions) {
    int x = pos.x - min_x;
    int y = pos.y - min_y;
    patch.at<unsigned char>(x, y) = 0;
  }
  std::stringstream ss;
  ss << "patch_" << num << "_" << bseg.id << ".png";
  //cv::imwrite(ss.str(), patch);
  std::cout << ss.str() << std::endl;
  dseg.patch_orig = patch;
  dseg.patch_resized = resize_contour(patch, DSEG_GRID_SIZE, DSEG_GRID_SIZE);
  //dseg.patch_resized = cv::Mat(DSEG_GRID_SIZE, DSEG_GRID_SIZE, cv::DataType<unsigned char>::type);
  //cv::resize(patch, dseg.patch_resized, dseg.patch_resized.size(), cv::INTER_AREA);
  cv::imwrite(ss.str(), dseg.patch_resized);
  return dseg;
}

std::vector<DSegment> generate_dsegs(std::unordered_map<int, BSegment> bmap, int num=0) {
  std::vector<DSegment> dsegs;
  for(auto pair : bmap) {
    pair.second.compute_averages();
    DColor main_color = pair.second.main_color;
    if (!is_magic_pink(main_color) && pair.second.positions.size() > 0) {
      generate_dseg(pair.second, num);
    }
  }
  return dsegs;
}

int main(int argc, char** argv) {
  cv::Mat mat = cv::imread("0000444.png", CV_LOAD_IMAGE_COLOR);
  //cv::Mat mat = cv::imread("../carrier.jpg", CV_LOAD_IMAGE_COLOR);
  fill_magic_pink(mat);
  std::cout << "loaded image" << std::endl;
  for(int scale = 1; scale <= 85;) {
    DSegmentationResult res = perform_segmentation(mat, 4 + scale, (4 + scale) * 0.25, 2500.0, true);
    std::stringstream ss;
    ss << "contours_" << scale << ".png";
    std::cout << ss.str() << std::endl;
    cv::imwrite(ss.str(), res.segmented_image);
    generate_dsegs(res.bmap, scale);
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
