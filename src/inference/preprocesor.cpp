#include "preprocessor.hpp"
#include <stdexcept>

// Initialize static constants
const std::array<float, 3> Preprocessor::NORM_MEAN = {0.48145466f, 0.4578275f, 0.40821073f};
const std::array<float, 3> Preprocessor::NORM_STD = {0.26862954f, 0.26130258f, 0.27577711f};

cv::Mat Preprocessor::encodeImage(const cv::Mat& img) {
    // Convert input image to standardized float array
    cv::Mat processedImg = imageToFloatArray(img);
    
    // Perform crop and resize operations
    processedImg = cropAndResize(processedImg);
    
    // normalise channels
    processedImg = normaliseChannels(processedImg);
    
    // Convert to NCHW format (1, 3, 224, 224)
    std::vector<cv::Mat> channels;
    cv::split(processedImg, channels);
    
    // Create output matrix with shape (1, 3, 224, 224)
    cv::Mat output(1, 3 * CLIP_INPUT_SIZE * CLIP_INPUT_SIZE, CV_32F);
    float* outputPtr = output.ptr<float>();
    
    // Fill output in NCHW format
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < CLIP_INPUT_SIZE; ++h) {
            for (int w = 0; w < CLIP_INPUT_SIZE; ++w) {
                outputPtr[c * CLIP_INPUT_SIZE * CLIP_INPUT_SIZE + h * CLIP_INPUT_SIZE + w] = 
                    channels[c].at<float>(h, w);
            }
        }
    }
    
    return output;
}

cv::Mat Preprocessor::_cropAndResize(const cv::Mat& img) {
    int h = img.rows;
    int w = img.cols;
    
    if (h * w == 0) {
        throw std::invalid_argument("Height and width of the image should both be non-zero");
    }
    
    // Calculate resize dimensions while maintaining aspect ratio
    int resized_h, resized_w;
    if (h < w) {
        resized_h = CLIP_INPUT_SIZE;
        resized_w = static_cast<int>(CLIP_INPUT_SIZE * (static_cast<float>(w) / h));
    } else {
        resized_w = CLIP_INPUT_SIZE;
        resized_h = static_cast<int>(CLIP_INPUT_SIZE * (static_cast<float>(h) / w));
    }
    
    // Resize image
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(resized_w, resized_h), 0, 0, cv::INTER_CUBIC);
    
    // Center crop
    int y_from = (resized_h - CLIP_INPUT_SIZE) / 2;
    int x_from = (resized_w - CLIP_INPUT_SIZE) / 2;
    cv::Rect roi(x_from, y_from, CLIP_INPUT_SIZE, CLIP_INPUT_SIZE);
    
    return resized(roi).clone();
}

cv::Mat Preprocessor::_imageToFloatArray(const cv::Mat& img) {
    validateImage(img);
    
    cv::Mat float_img;
    
    // Convert to float32 and scale to [0,1] if necessary
    if (img.depth() == CV_32F) {
        float_img = img.clone();
    } else {
        img.convertTo(float_img, CV_32F, 1.0/255.0);
    }
    
    // Handle grayscale images
    if (float_img.channels() == 1) {
        cv::cvtColor(float_img, float_img, cv::COLOR_GRAY2RGB);
    }
    
    cv::Mat output;
    float_img.convertTo(output, CV_32F);
    
    return output;
}

void Preprocessor::_validateImage(const cv::Mat& img) {
    if (img.empty()) {
        throw std::invalid_argument("Input image is empty");
    }
    
    if (img.dims > 2 && img.channels() != 3) {
        throw std::invalid_argument("Expected 3-channel RGB image or single-channel grayscale image");
    }
    
    if (img.depth() == CV_32F || img.depth() == CV_64F) {
        double minVal, maxVal;
        cv::minMaxLoc(img, &minVal, &maxVal);
        if (minVal < 0 || maxVal > 1) {
            throw std::invalid_argument("Floating point images should have values in [0,1]");
        }
    }
}

cv::Mat Preprocessor::_normaliseChannels(const cv::Mat& img) {
    cv::Mat normalised;
    img.convertTo(normalised, CV_32F);
    
    std::vector<cv::Mat> channels;
    cv::split(normalised, channels);
    
    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - NORM_MEAN[i]) / NORM_STD[i];
    }
    
    cv::merge(channels, normalised);
    return normalised;
}
