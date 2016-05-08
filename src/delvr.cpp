#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <boost/multi_array.hpp>

extern "C" {
    #include <vl/generic.h>
    #include <vl/slic.h>
}

int main(int argc, char** argv) {
  // loading routine based on https://github.com/davidstutz/vlfeat-slic-example
  cv::Mat mat = cv::imread("../carrier.jpg", CV_LOAD_IMAGE_COLOR);

  // convert to 1 dimensional
  float* image = new float[mat.rows*mat.cols*mat.channels()];
  for (int i = 0; i < mat.rows; ++i) {
      for (int j = 0; j < mat.cols; ++j) {
          // Assuming three channels ...
          image[j + mat.cols * i + mat.cols * mat.rows * 0] = mat.at<cv::Vec3b>(i, j)[0];
          image[j + mat.cols * i + mat.cols * mat.rows * 1] = mat.at<cv::Vec3b>(i, j)[1];
          image[j + mat.cols * i + mat.cols * mat.rows * 2] = mat.at<cv::Vec3b>(i, j)[2];
          if(mat.channels() == 4)
          image[j + mat.cols * i + mat.cols * mat.rows * 3] = mat.at<cv::Vec3b>(i, j)[3];
      }
  }

  // The algorithm will store the final segmentation in a one-dimensional array.
  vl_uint32* segmentation = new vl_uint32[mat.rows*mat.cols];
  vl_size height = mat.rows;
  vl_size width = mat.cols;
  vl_size channels = mat.channels();

  // The region size defines the number of superpixels obtained.
  // Regularization describes a trade-off between the color term and the
  // spatial term.
  vl_size region = 30;
  float regularization = 1000.;
  vl_size minRegion = 10;

  vl_slic_segment(segmentation, image, width, height, channels, region, regularization, minRegion);

  // Convert segmentation.
  int** labels = new int*[mat.rows];
  for (int i = 0; i < mat.rows; ++i) {
      labels[i] = new int[mat.cols];

      for (int j = 0; j < mat.cols; ++j) {
          labels[i][j] = (int) segmentation[j + mat.cols*i];
      }
  }

  int label = 0;
  int labelTop = -1;
  int labelBottom = -1;
  int labelLeft = -1;
  int labelRight = -1;

  for (int i = 0; i < mat.rows; i++) {
      for (int j = 0; j < mat.cols; j++) {

          label = labels[i][j];

          labelTop = label;
          if (i > 0) {
              labelTop = labels[i - 1][j];
          }

          labelBottom = label;
          if (i < mat.rows - 1) {
              labelBottom = labels[i + 1][j];
          }

          labelLeft = label;
          if (j > 0) {
              labelLeft = labels[i][j - 1];
          }

          labelRight = label;
          if (j < mat.cols - 1) {
              labelRight = labels[i][j + 1];
          }

          if (label != labelTop || label != labelBottom || label!= labelLeft || label != labelRight) {
              mat.at<cv::Vec3b>(i, j)[0] = 0;
              mat.at<cv::Vec3b>(i, j)[1] = 0;
              mat.at<cv::Vec3b>(i, j)[2] = 255;
          }
      }
  }

  cv::imwrite("contours.png", mat);

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
