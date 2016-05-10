#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <limits>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <boost/multi_array.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <thread>
#include <chrono>
#include <mutex>
#include <math.h>
#include <glob.h>

extern "C" {
  #include <vl/generic.h>
  #include <vl/slic.h>
}

const int ANN_EPOC_INCREMENT = 10;
const int DSEG_GRID_SIZE = 16;
const int DSEG_POINTS_THRESHOLD = 8;
const int DSEG_MAX_IMG_SIZE = 800;
const int DSEG_POSITIVE_SCALE_START = 1;
const int DSEG_POSITIVE_SCALE_END = 60;
const int DSEG_NEGATIVE_SCALE_START = 5;
const int DSEG_NEGATIVE_SCALE_END = 45;
const int DSEG_MAX_OPAQUE_IMAGES = 2000;
const int DSEG_DATA_SIZE = DSEG_GRID_SIZE * DSEG_GRID_SIZE + 4;
const float DSEG_REGION_RATIO = 0.25;
const float DSEG_REGULARIZATION = 4000.0;
std::random_device rd;
std::mt19937 g(rd());

// vector addition
template <typename T>
std::vector<T>& operator+=(std::vector<T>& a, const std::vector<T>& b)
{
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

inline float sq(float n) {
  return n * n;
}

inline int sq(int n) {
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

    void add_pixel(int x, int y, unsigned char r, unsigned char g, unsigned char b) {
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
      r_sum += (int)r;
      g_sum += (int)g;
      b_sum += (int)b;
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
      avg_color.r = (unsigned char)(r_sum / positions.size());
      avg_color.g = (unsigned char)(g_sum / positions.size());
      avg_color.b = (unsigned char)(b_sum / positions.size());
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

class DFeatVect {
  public:
    char data[DSEG_GRID_SIZE * DSEG_GRID_SIZE + 4];

    void set_color(DColor color) {
      data[DSEG_GRID_SIZE * DSEG_GRID_SIZE + 1] = (char)color.r;
      data[DSEG_GRID_SIZE * DSEG_GRID_SIZE + 2] = (char)color.g;
      data[DSEG_GRID_SIZE * DSEG_GRID_SIZE + 3] = (char)color.b;
    }

    void set_grid(cv::Mat &mat) {
      for(int x = 0; x < DSEG_GRID_SIZE; x++)
        for(int y = 0; y < DSEG_GRID_SIZE; y++)
          data[y * DSEG_GRID_SIZE + x] = 255 - mat.at<unsigned char>(x, y);
    }
};

DFeatVect make_feature_vector(DSegment dseg) {
  DFeatVect feat;
  feat.set_grid(dseg.patch_resized);
  feat.set_color(dseg.cta_color);
  return feat;
}

class DFeatFile {
  public:
    std::vector<char> buffer;
    long num_features;
    long file_size;
    bool positive; // whether these are positive or negative examples

    void load(std::string file_path, bool ispositive) {
      positive = ispositive;
      std::cout << "loading " << file_path << "..." << std::endl;
      std::ifstream file(file_path, std::ios::binary | std::ios::ate);
      std::streamsize size = file.tellg();
      file_size = size;
      num_features = size / DSEG_DATA_SIZE;
      std::cout << "loading " << size << " bytes into memory..." << std::endl;
      file.seekg(0, std::ios::beg);
      buffer.reserve(size);
      if (!file.read(buffer.data(), size)) {
        std::cout << "could not read file " << file_path << std::endl;
        exit(1);
      }
      std::cout << "done loading" << std::endl;
    }

    unsigned char *block(int index) {
      return (unsigned char *)(buffer.data() + DSEG_DATA_SIZE * index);
    }

    cv::Mat get_ANN_training_blob(int cutoff, int offset=0) {
      if (cutoff > num_features) {
        std::cout << "error: cutoff cannot exceed num_features!" << std::endl;
      }
      std::cout << "converting to cv::Mat blob for ANN..." << std::endl;
      cv::Mat inputs(cutoff - offset, DSEG_DATA_SIZE, CV_32F);
      for(int i = offset; i < cutoff; i++) {
        for(int j = 0; j < DSEG_DATA_SIZE; j++) {
          inputs.at<float>(i, j) = (float)(block(i)[j]);
        }
      }
      std::cout << "done converting." << std::endl;
      return inputs;
    }
};

class DSegmentationResult {
  public:
    std::unordered_map<int, BSegment> bmap;
    cv::Mat segmented_image;
};

class DPair {
  public:
    unsigned char *block;
    bool positive;
};

class ANNDataset {
  public:
    cv::Mat training_inputs;
    cv::Mat training_outputs;
    cv::Mat validation_inputs;
    cv::Mat validation_outputs;

    void load(DFeatFile &positives, DFeatFile &negatives, DFeatFile &validation_positives) {
      std::vector<DPair> positive_pairs;
      std::vector<DPair> negative_pairs;
      std::vector<DPair> validation_positive_pairs;
      std::vector<DPair> training_negative_pairs;
      std::vector<DPair> validation_negative_pairs;
      // add positive examples
      for(int i = 0; i < positives.num_features; i++) {
        DPair pair;
        pair.positive = true;
        pair.block = positives.block(i);
        positive_pairs.push_back(pair);
      }
      // add validation positive examples
      for(int i = 0; i < validation_positives.num_features; i++) {
        DPair pair;
        pair.positive = true;
        pair.block = positives.block(i);
        validation_positive_pairs.push_back(pair);
      }
      // add negative examples
      for(int i = 0; i < negatives.num_features; i++) {
        DPair pair;
        pair.positive = false;
        pair.block = negatives.block(i);
        negative_pairs.push_back(pair);
      }
      // shuffle pairs
      std::shuffle(std::begin(positive_pairs), std::end(positive_pairs), rd);
      std::shuffle(std::begin(negative_pairs), std::end(negative_pairs), rd);
      std::shuffle(std::begin(validation_positive_pairs), std::end(validation_positive_pairs), rd);
      // divide training and validation
      for(DPair pair : negative_pairs) {
        if (validation_negative_pairs.size() < validation_positive_pairs.size()) {
          validation_negative_pairs.push_back(pair);
        } else {
          if (training_negative_pairs.size() < 500000) {
            training_negative_pairs.push_back(pair);
          } else {
            break;
          }
        }
      }
      std::cout << negative_pairs.size() << " negative features available" << std::endl;
      std::cout << validation_negative_pairs.size() << " negative features allocated to validation set" << std::endl;
      std::cout << training_negative_pairs.size() << " negative features available for training set" << std::endl;
      negative_pairs.clear();
      std::cout << positive_pairs.size() << " positive features available for training set" << std::endl;
      std::vector<DPair> training_pairs;
      for(DPair pair : positive_pairs) {
        if (training_pairs.size() < training_negative_pairs.size()) {
          training_pairs.push_back(pair);
        } else {
          break;
        }
      }
      training_pairs += training_negative_pairs;
      std::cout << "final size of training set:  " << training_pairs.size() << std::endl;
      std::vector<DPair> validation_pairs;
      validation_pairs += validation_positive_pairs;
      validation_pairs += validation_negative_pairs;
      std::cout << "final size of validation set: " << validation_pairs.size() << std::endl;
      // final shuffle
      std::shuffle(std::begin(validation_pairs), std::end(validation_pairs), rd);
      std::shuffle(std::begin(training_pairs), std::end(training_pairs), rd);
      positive_pairs.clear();
      negative_pairs.clear();
      validation_positive_pairs.clear();
      validation_negative_pairs.clear();
      training_negative_pairs.clear();
      std::cout << "done shuffling" << std::endl;

      // create images
      std::cout << "generating training inputs image..." << std::endl;
      training_inputs = cv::Mat(training_pairs.size(), DSEG_DATA_SIZE, CV_32F);
      for(int i = 0; i < training_pairs.size(); i++)
        for(int j = 0; j < DSEG_DATA_SIZE; j++)
          training_inputs.at<float>(i, j) = ((float)(training_pairs[i].block[j])) / 255.0;
      positives.buffer.clear();

      std::cout << "generating validation inputs image..." << std::endl;
      validation_inputs = cv::Mat(validation_pairs.size(), DSEG_DATA_SIZE, CV_32F);
      for(int i = 0; i < validation_pairs.size(); i++)
        for(int j = 0; j < DSEG_DATA_SIZE; j++)
          validation_inputs.at<float>(i, j) = ((float)(validation_pairs[i].block[j])) / 255.0;
      validation_positives.buffer.clear();
      negatives.buffer.clear();

      std::cout << "generating training outputs image..." << std::endl;
      training_outputs = cv::Mat(training_pairs.size(), 1, CV_32F);
      for(int i = 0; i < training_pairs.size(); i++) {
        float val;
        if (training_pairs[i].positive) {
          val = 1.0;
        } else {
          val = 0.0;
        }
        training_outputs.at<float>(i, 0) = val;
      }

      std::cout << "generating validation outputs image..." << std::endl;
      validation_outputs = cv::Mat(validation_pairs.size(), 1, CV_32F);
      for(int i = 0; i < validation_pairs.size(); i++) {
        float val;
        if (validation_pairs[i].positive) {
          val = 1.0;
        } else {
          val = 0.0;
        }
        validation_outputs.at<float>(i, 0) = val;
      }

      std::cout << "done." << std::endl;
    }
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

// novel algorithm for vectorizing and resizing solid-color images
// preserves softness of edges
cv::Mat resize_contour(cv::Mat &src, int dest_width, int dest_height) {
  cv::Mat dest = cv::Mat(dest_width, dest_height, cv::DataType<unsigned char>::type);
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
        cv::fillPoly(dest, polys, 0);
      }
    }
  }
  /* commented out code for drawing markers
  for(int x = 0; x < src.cols; x++) {
    for(int y = 0; y < src.rows; y++) {
      if (src.at<unsigned char>(x, y) == 0) {
        DPos proj = project_point(dest_width, dest_height, x, y, xmod, ymod);
        //dest.at<unsigned char>(proj.x, proj.y) = 0;
        cv::drawMarker(dest, cv::Point(proj.x, proj.y), 0);
      }
    }
  }*/
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
      unsigned char b = (unsigned char)mat.at<cv::Vec3b>(i, j)[0];
      unsigned char g = (unsigned char)mat.at<cv::Vec3b>(i, j)[1];
      unsigned char r = (unsigned char)mat.at<cv::Vec3b>(i, j)[2];
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
  dseg.cta_color = bseg.main_color;
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
  for(DPos pos : bseg.positions) {
    int x = pos.x - min_x;
    int y = pos.y - min_y;
    patch.at<unsigned char>(x, y) = 0;
  }
  //std::stringstream ss;
  //ss << "patch_" << num << "_" << bseg.id << ".png";
  //cv::imwrite(ss.str(), patch);
  //dseg.patch_orig = patch;
  dseg.patch_resized = resize_contour(patch, DSEG_GRID_SIZE, DSEG_GRID_SIZE);
  //dseg.patch_resized = cv::Mat(DSEG_GRID_SIZE, DSEG_GRID_SIZE, cv::DataType<unsigned char>::type);
  //cv::resize(patch, dseg.patch_resized, dseg.patch_resized.size(), cv::INTER_AREA);
  //cv::imwrite(ss.str(), dseg.patch_resized);
  return dseg;
}

std::vector<DSegment> generate_dsegs(std::unordered_map<int, BSegment> bmap, int num=0) {
  std::vector<DSegment> dsegs;
  for(auto pair : bmap) {
    pair.second.compute_averages();
    DColor main_color = pair.second.main_color;
    if (!is_magic_pink(main_color) && pair.second.positions.size() > DSEG_POINTS_THRESHOLD) {
      dsegs.push_back(generate_dseg(pair.second, num));
    }
  }
  return dsegs;
}

std::vector<DFeatVect> frame_to_feature_vectors(cv::Mat &mat, bool translucent=true, bool output_images=false) {
  if (translucent)
    fill_magic_pink(mat);
  if (mat.rows > DSEG_MAX_IMG_SIZE || mat.cols > DSEG_MAX_IMG_SIZE) {
    int new_rows, new_cols;
    if (mat.rows > mat.cols) {
      new_rows = DSEG_MAX_IMG_SIZE;
      new_cols = min(round(((float)DSEG_MAX_IMG_SIZE / (float)mat.rows) * mat.cols), DSEG_MAX_IMG_SIZE);
    } else {
      new_cols = DSEG_MAX_IMG_SIZE;
      new_rows = min(round(((float)DSEG_MAX_IMG_SIZE / (float)mat.cols) * mat.rows), DSEG_MAX_IMG_SIZE);
    }
    cv::Mat tmp;
    //std::cout << "old: " << mat.cols << ", " << mat.rows << " => " << new_cols << ", " << new_rows << std::endl;
    cv::resize(mat, tmp, cv::Size(new_cols, new_rows));
    mat = tmp.clone();
  }
  std::vector<DSegment> all_dsegs;
  int scale_start, scale_end;
  if (translucent) {
    scale_start = DSEG_POSITIVE_SCALE_START;
    scale_end = DSEG_POSITIVE_SCALE_END;
  } else {
    scale_start = DSEG_NEGATIVE_SCALE_START;
    scale_end = DSEG_NEGATIVE_SCALE_END;
  }
  for(int scale = scale_start; scale <= scale_end;) {
    DSegmentationResult res = perform_segmentation(mat, 4 + scale, (4 + scale) * DSEG_REGION_RATIO, DSEG_REGULARIZATION, output_images);
    if (output_images) {
      std::stringstream ss;
      ss << "contours_" << scale << ".png";
      std::cout << ss.str() << std::endl;
      cv::imwrite(ss.str(), res.segmented_image);
    }
    std::vector<DSegment> dsegs = generate_dsegs(res.bmap, scale);
    all_dsegs.insert(all_dsegs.end(), dsegs.begin(), dsegs.end());
    if (translucent) {
      if (scale < 10)
        scale += 1;
      else if (scale < 20)
        scale += 2;
      else if (scale < 30)
        scale += 4;
      else if (scale < 40)
        scale += 8;
      else if (scale < 50)
        scale += 10;
      else
        scale += 15;
    } else {
      scale += 10;
    }
  }
  std::vector<DFeatVect> feats;
  for(DSegment dseg : all_dsegs)
    feats.push_back(make_feature_vector(dseg));
  return feats;
}

// adapted from http://stackoverflow.com/questions/612097/how-can-i-get-the-list-of-files-in-a-directory-using-c-or-c
std::vector<std::string> match_files(const std::string &pattern) {
  glob_t glob_result;
  glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
  std::vector<std::string> files;
  for(unsigned int i = 0; i < glob_result.gl_pathc; ++i) {
    files.push_back(std::string(glob_result.gl_pathv[i]));
  }
  globfree(&glob_result);
  return files;
}

int num_done = 0;
std::mutex print_mutex;
void thread_print(int thread_num, std::string msg) {
  print_mutex.lock();
  num_done++;
  std::cout << "#" << thread_num << "(" << num_done << "): " << msg << std::endl;
  print_mutex.unlock();
}

std::mutex file_mutex;
std::ofstream out_file;

void genfeats_multithreaded(int thread_num, std::vector<std::string> img_paths, std::string outfile, bool translucent) {
  for(std::string path : img_paths) {
    cv::Mat mat = cv::imread(path, CV_LOAD_IMAGE_COLOR);
    std::vector<DFeatVect> feats = frame_to_feature_vectors(mat, translucent, false);
    std::stringstream ss;
    ss << "found " << feats.size() << " features in " << path;
    thread_print(thread_num, ss.str());
    file_mutex.lock();
    for(DFeatVect feat : feats) {
      out_file.write(feat.data, sizeof(feat.data));
    }
    file_mutex.unlock();
  }
}

void train_ANN(cv::Ptr<cv::ml::ANN_MLP> &net, ANNDataset &dataset, double target_error=0.2) {
  std::cout << "training..." << std::endl;
  double err = 1.0;
  int epoch = 0;
  do {
    net->train(dataset.training_inputs, cv::ml::ROW_SAMPLE, dataset.training_outputs);
    epoch += ANN_EPOC_INCREMENT;
    std::cout << "epoch: " << epoch << std::endl;
    int num_correct = 0;
    int num_wrong = 0;
    cv::Mat output;
    net->predict(dataset.validation_inputs, output);
    for(int i = 0; i < output.rows; i++) {
      float actual = output.at<float>(0, i);
      if (std::isnan(actual))
        actual = 0.0;
      float expected = dataset.validation_outputs.at<float>(0, i);
      bool correct = round(max(0, actual)) == round(expected);
      if (correct)
        num_correct += 1;
      else
        num_wrong += 1;
    }
    float accuracy = (float)num_correct / (float)(num_wrong + num_correct);
    err = 1.0 - accuracy;
    std::cout << "num_correct: " << num_correct << std::endl;
    std::cout << "num_wrong:   " << num_wrong << std::endl;
    std::cout << "accuracy: " << accuracy << std::endl;
  } while(err > target_error);
  std::cout << "training stopped (target error threshold reached)" << std::endl;

}

int main(int argc, char** argv) {
  std::cout << "===========================================================" << std::endl;
  std::cout << " Delvr 1.0 - copyright (c) Sam Kelly - all rights reserved " << std::endl;
  std::cout << "===========================================================" << std::endl;
  std::cout << std::endl;
  if (argc == 1) {
    std::cout << "usage: delvr genfeats [translucent|opaque] [path/to/images] [path/to/outfile]" << std::endl;
    std::cout << "       delvr traindetector [path/to/positive/features] [path/to/negative/features] [output/path]" << std::endl;
    return 0;
  }
  unsigned num_threads = std::thread::hardware_concurrency();

  // BEGIN GENFEATS
  if (std::string(argv[1]) == "genfeats") {
    if (argc != 5) {
      std::cout << "wrong number of arguments!" << std::endl;
      return 1;
    }
    bool translucent = false;
    if (std::string(argv[2]) == "opaque") {
      translucent = false;
    } else if (std::string(argv[2]) == "translucent") {
      translucent = true;
    } else {
      std::cout << "error: translucent / opaque not specified!" << std::endl;
      return 1;
    }
    std::string images_path = std::string(argv[3]);
    if (!boost::algorithm::ends_with(images_path, "/"))
      images_path = images_path + "/";
    std::string outpath = std::string(argv[4]);
    std::cout << "Feature generation routine started." << std::endl;
    std::cout << std::endl;

    std::vector<std::string> imgs;
    if (translucent) {
      imgs = match_files(images_path + "*.png");
      std::cout << "found " << imgs.size() << " PNG files in " << images_path << std::endl;
    } else {
      // hack for SUN since glob does not work recursively
      imgs = match_files(images_path + "*.jpg");
      imgs += match_files(images_path + "**/*.jpg");
      imgs += match_files(images_path + "**/**/*.jpg");
      imgs += match_files(images_path + "**/**/**/*.jpg");
      imgs += match_files(images_path + "**/**/**/**/*.jpg");
      imgs += match_files(images_path + "**/**/**/**/**/*.jpg");
      std::cout << "found " << imgs.size() << " JPG files in " << images_path << std::endl;
    }
    std::cout << "randomly shuffling images..." << std::endl;
    std::shuffle(std::begin(imgs), std::end(imgs), rd);
    std::cout << "done shuffling." << std::endl;

    std::cout << "will use " << num_threads << " threads" << std::endl;
    std::cout << "creating threads..." << std::endl;
    std::vector<std::thread> threads;
    // divy up workload among available threads
    int num_added = 0;
    std::vector<std::vector<std::string>> workload;
    for(int i = 0; i < num_threads; i++)
      workload.push_back(std::vector<std::string>());
    for(int i = 0, j = 0; i < imgs.size(); i++, j++) {
      num_added++;
      if (!translucent && num_added >= DSEG_MAX_OPAQUE_IMAGES)
        break;
      workload[j].push_back(imgs[i]);
      if (j == num_threads - 1)
        j = -1;
    }
    std::cout << "will use " << num_added << " images" << std::endl;
    // set up io
    out_file.open(outpath, std::ios::out | std::ios::app | std::ios::binary);
    // start threads
    for(int i = 0; i < num_threads; i++) {
      threads.push_back(std::thread(genfeats_multithreaded, i, workload[i], outpath, translucent));
    }
    for(int i = 0; i < num_threads; i++) {
      threads[i].join();
    }
    std::cout << "all threads finished" << std::endl;
    std::cout << "finalizing output file..." << std::endl;
    out_file.close();
    std::cout << "done." << std::endl;
    return 0;

  // BEGIN TRAIN DETECTOR
  } else if(std::string(argv[1]) == "traindetector") {
    if (argc != 6) {
      std::cout << "error: wrong number of arguments!" << std::endl;
      exit(1);
    }
    std::string positive_features_path = std::string(argv[2]);
    std::string negative_features_path = std::string(argv[3]);
    std::string positive_features_test_path = std::string(argv[4]);
    std::string output_path = std::string(argv[5]);
    DFeatFile positive_features;
    positive_features.load(positive_features_path, true);
    DFeatFile positive_features_test;
    positive_features_test.load(positive_features_test_path, true);
    DFeatFile negative_features;
    negative_features.load(negative_features_path, false);
    ANNDataset dataset;
    dataset.load(positive_features, negative_features, positive_features_test);

    int num_samples = dataset.training_inputs.rows;

    std::vector<int> layer_sizes = {DSEG_DATA_SIZE, 84, 1};
    cv::Ptr<cv::ml::ANN_MLP> net = cv::ml::ANN_MLP::create();
    net->setLayerSizes(layer_sizes);
    net->setActivationFunction(cv::ml::ANN_MLP::SIGMOID_SYM);
    net->setTrainMethod(cv::ml::ANN_MLP::RPROP);
    net->setTermCriteria(cv::TermCriteria(cv::TermCriteria::Type::MAX_ITER, ANN_EPOC_INCREMENT, 0.0));

    train_ANN(net, dataset, 0.2);
  }
  return 0;
}
