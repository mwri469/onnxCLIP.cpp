#ifndef CLIP_PREPROCESSOR_H
#define CLIP_PREPROCESSOR_H

#pragma once

#include <vector>
#include <array>
#include <opencv2/opencv.hpp>

#define CLIP_INPUT_SIZE 224

class CLIPpreprocessor {
public:
	// Constructor/destructor
	CLIPpreprocessor();
	~CLIPpreprocessor();
	
	cv::Mat encodeImage(const cv::Mat& img);
public:
	// Fixed variables that ensure the correct output shapes and values for the `Model` class.
	static constexpr int CLIP_INPUT_SIZE = 224;

	// Normalization constants taken from original CLIP:
	// https://github.com/openai/CLIP/blob/3702849800aa56e2223035bccd1c6ef91c704ca8/clip/clip.py#L85
	static const std::array<float, 3> NORM_MEAN;
    static const std::array<float, 3> NORM_STD;

private:
	static cv::Mat _cropAndResize(const cv::Mat& img);
    static cv::Mat _imageToFloatArray(const cv::Mat& img);
    static void _validateImage(const cv::Mat& img);
    static cv::Mat _normalizeChannels(const cv::Mat& img);
}

#endif // CLIP_PREPROCESSOR_H
