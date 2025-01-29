#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <stdexcept>
#include <vector>

class Preprocessor {
public:
    static const int CLIP_INPUT_SIZE = 224;
    static const std::vector<float> NORM_MEAN;
    static const std::vector<float> NORM_STD;

    static torch::Tensor encode_image(const cv::Mat& img);

private:
    static cv::Mat _crop_and_resize(const cv::Mat& img);
    static cv::Mat _image_to_float_array(const cv::Mat& img);
};

#endif // PREPROCESSOR_H