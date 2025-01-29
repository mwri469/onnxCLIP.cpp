#include "preprocessor.hpp"

const std::vector<float> Preprocessor::NORM_MEAN = {0.48145466f, 0.4578275f, 0.40821073f};
const std::vector<float> Preprocessor::NORM_STD = {0.26862954f, 0.26130258f, 0.27577711f};

torch::Tensor Preprocessor::encode_image(const cv::Mat& img) {
    cv::Mat float_img = _image_to_float_array(img);
    cv::Mat resized_img = _crop_and_resize(float_img);

    // Normalize channels
    cv::Mat normalized_img;
    cv::subtract(resized_img, cv::Scalar(NORM_MEAN[0], NORM_MEAN[1], NORM_MEAN[2]), normalized_img);
    cv::divide(normalized_img, cv::Scalar(NORM_STD[0], NORM_STD[1], NORM_STD[2]), normalized_img);

    // Convert to torch tensor
    torch::Tensor tensor_img = torch::from_blob(normalized_img.data, {1, resized_img.rows, resized_img.cols, 3}, torch::kFloat32);
    tensor_img = tensor_img.permute({0, 3, 1, 2}); // Change to (C, H, W)

    return tensor_img;
}

cv::Mat Preprocessor::_crop_and_resize(const cv::Mat& img) {
    int h = img.rows;
    int w = img.cols;

    if (h * w == 0) {
        throw std::invalid_argument("Height and width of the image should both be non-zero.");
    }

    int target_size = CLIP_INPUT_SIZE;
    int resized_h, resized_w;

    if (h < w) {
        resized_h = target_size;
        resized_w = static_cast<int>(resized_h * static_cast<float>(w) / h);
    } else {
        resized_w = target_size;
        resized_h = static_cast<int>(resized_w * static_cast<float>(h) / w);
    }

    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_CUBIC);

    // Crop to a square
    int y_from = (resized_h - target_size) / 2;
    int x_from = (resized_w - target_size) / 2;
    cv::Rect roi(x_from, y_from, target_size, target_size);
    cv::Mat cropped_img = resized_img(roi);

    return cropped_img;
}

cv::Mat Preprocessor::_image_to_float_array(const cv::Mat& img) {
    if (img.channels() != 3 && img.channels() != 1) {
        throw std::invalid_argument("The image should have 1 or 3 channels.");
    }

    cv::Mat float_img;
    if (img.channels() == 1) {
        cv::cvtColor(img, float_img, cv::COLOR_GRAY2RGB);
    } else {
        img.convertTo(float_img, CV_32FC3, 1.0 / 255.0);
    }

    if (cv::sum(float_img < 0.0)[0] > 0) {
        throw std::invalid_argument("Images should have non-negative pixel values.");
    }

    if (cv::sum(float_img > 1.0)[0] > 0) {
        throw std::invalid_argument("Images should have values in [0, 1].");
    }

    return float_img;
}