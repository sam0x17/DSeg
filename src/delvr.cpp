#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
#include <boost/multi_array.hpp>

extern "C" {
  #include <vl/generic.h>
  #include <vl/slic.h>
}

int seg_at(vl_uint32 *segmentation, int i, int j, int cols) {
  return (int)segmentation[j + cols * i];
}

int main(int argc, char** argv) {
  // loading routine based on https://github.com/davidstutz/vlfeat-slic-example
  cv::Mat mat = cv::imread("0000444.png", CV_LOAD_IMAGE_COLOR);
  //cv::Mat mat = cv::imread("../carrier.jpg", CV_LOAD_IMAGE_COLOR);
  std::cout << "loaded image" << std::endl;

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
  vl_uint32* segmentation = new vl_uint32[mat.rows * mat.cols];
  vl_size height = mat.rows;
  vl_size width = mat.cols;
  vl_size channels = mat.channels();

  // The region size defines the number of superpixels obtained.
  // Regularization describes a trade-off between the color term and the
  // spatial term.

  for(int scale = 1; scale <= 100; scale++) {
    cv::Mat matc = mat.clone();
    for(int i = 0; i < mat.rows * mat.cols; i++) {
      segmentation[i] = 0.0;
    }
    vl_size region = 4 + scale;
    float regularization = 1500.0;
    vl_size minRegion = region * 0.25;
    std::cout << "performing segmentation" << std::endl;
    std::cout << "region: " << region << std::endl;
    std::cout << "minregion: " << minRegion << std::endl;
    vl_slic_segment(segmentation, image, width, height, channels, region, regularization, minRegion);

    int label = 0;
    int label_top = -1;
    int label_bottom = -1;
    int label_left = -1;
    int label_right = -1;

    for (int i = 0; i < matc.rows; i++) {
      for (int j = 0; j < matc.cols; j++) {
        label = seg_at(segmentation, i, j, matc.cols);

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
    }

    std::stringstream ss;
    ss << "contours" << scale << ".png";
    std::cout << ss.str() << std::endl;
    cv::imwrite(ss.str(), matc);
  }
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
