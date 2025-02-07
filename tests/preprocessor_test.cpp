#include <iostream>
#include <opencv2/opencv.hpp>
#include "../src/inference/preprocessor.hpp"
#include <torch/torch.h>

const std::string ASSETS_PATH = "../assets/";

// Helper function to check if Mat contains values in valid range
bool checkMatRange(const cv::Mat& mat, float min_val, float max_val) {
    double minVal, maxVal;
    cv::minMaxLoc(mat, &minVal, &maxVal);
    return (minVal >= min_val && maxVal <= max_val);
}

bool test_matches_original_clip() {
    std::cout << "=== Running test: MatchesOriginalCLIP ===" << std::endl;
    Preprocessor preprocessor;
    
    // Load test image
    cv::Mat img = cv::imread(ASSETS_PATH + "franz-kafka.jpg");
    if (img.empty()) {
        std::cerr << "Error: Could not load test image." << std::endl;
        return false;
    }

    // Process image
    torch::Tensor processed = preprocessor.encode_image(img);

    // Verify output dimensions
    auto sizes = processed.sizes();
    if (sizes.size() != 4 || 
        sizes[0] != 1 || 
        sizes[1] != 3 || 
        sizes[2] != Preprocessor::CLIP_INPUT_SIZE || 
        sizes[3] != Preprocessor::CLIP_INPUT_SIZE) {
        std::cerr << "Error: Output dimensions incorrect. Expected [1, 3, "
                  << Preprocessor::CLIP_INPUT_SIZE << ", " << Preprocessor::CLIP_INPUT_SIZE
                  << "], got [" << sizes[0] << ", " << sizes[1] << ", "
                  << sizes[2] << ", " << sizes[3] << "]" << std::endl;
        return false;
    }

    // Load expected values from text file
    torch::Tensor expected;
    try {
        expected = load_tensor_from_txt(ASSETS_PATH + "expected_preprocessed_image.txt");
    } catch (const std::exception& e) {
        std::cerr << "Error loading expected values: " << e.what() << std::endl;
        return false;
    }

    // Verify numerical similarity
    if (!torch::allclose(processed, expected, /*rtol=*/1e-5, /*atol=*/1e-6)) {
        std::cerr << "Error: Processed values do not match expected values." << std::endl;
        return false;
    }

    std::cout << "CLIP preprocessing matches reference implementation." << std::endl;
    return true;
}

torch::Tensor load_tensor_from_txt(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    // Read dimensions
    int num_dims;
    file >> num_dims;
    
    std::vector<int64_t> dims(num_dims);
    for (int i = 0; i < num_dims; ++i) {
        file >> dims[i];
    }

    // Read flattened data
    std::vector<float> data;
    float value;
    while (file >> value) {
        data.push_back(value);
    }

    // Verify data size matches dimensions
    int64_t expected_size = 1;
    for (auto dim : dims) {
        expected_size *= dim;
    }
    
    if (data.size() != expected_size) {
        throw std::runtime_error("Data size mismatch in file: " + file_path);
    }

    // Create and reshape tensor
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    return torch::from_blob(data.data(), dims, options).clone();
}

bool test_basic_preprocessing() {
    std::cout << "=== Running test: BasicPreprocessing ===" << std::endl;
    Preprocessor preprocessor;
    cv::Mat img = cv::imread(ASSETS_PATH + "franz-kafka.jpg");
    if (img.empty()) {
        std::cerr << "Error: Could not load test image." << std::endl;
        return false;
    }

    torch::Tensor processed = preprocessor.encode_image(img);

    // Check output dimensions (1, 3, 224, 224)
    if (processed.rows != 1 || processed.cols != 3 * Preprocessor::CLIP_INPUT_SIZE * Preprocessor::CLIP_INPUT_SIZE || processed.type() != CV_32F) {
        std::cerr << "Error: Output dimensions or type are incorrect." << std::endl;
        return false;
    }

    std::cout << "Basic preprocessing passed." << std::endl;
    return true;
}

bool test_different_input_sizes() {
    std::cout << "=== Running test: DifferentInputSizes ===" << std::endl;
    Preprocessor preprocessor;

    // Test with a tall image
    cv::Mat tall_img(480, 320, CV_8UC3, cv::Scalar(255, 255, 255));
    torch::Tensor processed_tall = preprocessor.encode_image(tall_img);

    // Test with a wide image
    cv::Mat wide_img(320, 480, CV_8UC3, cv::Scalar(255, 255, 255));
    torch::Tensor processed_wide = preprocessor.encode_image(wide_img);

    // Both should result in the same output dimensions
    if (processed_tall.size() != processed_wide.size()) {
        std::cerr << "Error: Output sizes differ for tall and wide images." << std::endl;
        return false;
    }

    std::cout << "Different input sizes handled correctly." << std::endl;
    return true;
}

bool test_grayscale_input() {
    std::cout << "=== Running test: GrayscaleInput ===" << std::endl;
    Preprocessor preprocessor;
    cv::Mat gray_img(224, 224, CV_8UC1, cv::Scalar(128));
    torch::Tensor processed = preprocessor.encode_image(gray_img);

    // Check that output has correct dimensions for RGB
    if (processed.cols != 3 * Preprocessor::CLIP_INPUT_SIZE * Preprocessor::CLIP_INPUT_SIZE) {
        std::cerr << "Error: Output dimensions are incorrect for grayscale input." << std::endl;
        return false;
    }

    std::cout << "Grayscale input handled correctly." << std::endl;
    return true;
}

bool test_normalization() {
    std::cout << "=== Running test: Normalization ===" << std::endl;
    Preprocessor preprocessor;
    cv::Mat img = cv::imread(ASSETS_PATH + "franz-kafka.jpg");
    if (img.empty()) {
        std::cerr << "Error: Could not load test image." << std::endl;
        return false;
    }

    torch::Tensor processed = preprocessor.encode_image(img);

    // Reshape to extract channels
    torch::Tensor reshaped = processed.reshape(3, Preprocessor::CLIP_INPUT_SIZE);
    std::vector<cv::Mat> channels;
    cv::split(reshaped, channels);

    // Check that values are normalized (should be roughly between -2 and 2 after normalization)
    for (const auto& channel : channels) {
        if (!checkMatRange(channel, -3.0f, 3.0f)) {
            std::cerr << "Error: Normalization range is incorrect." << std::endl;
            return false;
        }
    }

    std::cout << "Normalization passed." << std::endl;
    return true;
}

bool test_invalid_inputs() {
    std::cout << "=== Running test: InvalidInputs ===" << std::endl;
    Preprocessor preprocessor;

    // Empty image
    cv::Mat empty_img;
    try {
        preprocessor.encode_image(empty_img);
        std::cerr << "Error: Empty image did not throw an exception." << std::endl;
        return false;
    } catch (const std::invalid_argument&) {
        // Expected
    }

    // Image with invalid dimensions
    cv::Mat invalid_img(0, 224, CV_8UC3);
    try {
        preprocessor.encode_image(invalid_img);
        std::cerr << "Error: Invalid image dimensions did not throw an exception." << std::endl;
        return false;
    } catch (const std::invalid_argument&) {
        // Expected
    }

    std::cout << "Invalid inputs handled correctly." << std::endl;
    return true;
}

bool test_float_image_input() {
    std::cout << "=== Running test: FloatImageInput ===" << std::endl;
    Preprocessor preprocessor;
    cv::Mat float_img(224, 224, CV_32FC3, cv::Scalar(0.5, 0.5, 0.5));
    torch::Tensor processed = preprocessor.encode_image(float_img);

    // Check output type
    if (processed.type() != CV_32F) {
        std::cerr << "Error: Output type is incorrect for float image input." << std::endl;
        return false;
    }

    std::cout << "Float image input handled correctly." << std::endl;
    return true;
}

bool test_aspect_ratio_preservation() {
    std::cout << "=== Running test: AspectRatioPreservation ===" << std::endl;
    Preprocessor preprocessor;
    cv::Mat rect_img(300, 600, CV_8UC3, cv::Scalar(255, 255, 255));
    torch::Tensor processed = preprocessor.encode_image(rect_img);
    cv::Mat reshaped = processed.reshape(3, Preprocessor::CLIP_INPUT_SIZE);

    // The center crop should be square
    if (reshaped.rows != Preprocessor::CLIP_INPUT_SIZE || reshaped.cols != Preprocessor::CLIP_INPUT_SIZE) {
        std::cerr << "Error: Aspect ratio was not preserved." << std::endl;
        return false;
    }

    std::cout << "Aspect ratio preserved correctly." << std::endl;
    return true;
}

bool test_output_range() {
    std::cout << "=== Running test: OutputRange ===" << std::endl;
    Preprocessor preprocessor;
    cv::Mat img = cv::imread(ASSETS_PATH + "isiahthomas.jpg");
    if (img.empty()) {
        std::cerr << "Error: Could not load test image." << std::endl;
        return false;
    }

    torch::Tensor processed = preprocessor.encode_image(img);

    // Check that values are in a reasonable range after normalization
    if (!checkMatRange(processed, -5.0f, 5.0f)) {
        std::cerr << "Error: Output range is incorrect." << std::endl;
        return false;
    }

    std::cout << "Output range is valid." << std::endl;
    return true;
}

int main() {
    int passed = 0;
    int failed = 0;

    auto run_test = [&](bool (*test_func)(), const std::string& name) {
        bool result = test_func();
        if (result) {
            std::cout << "+++ PASSED +++\n" << std::endl;
            passed++;
        } else {
            std::cout << "--- FAILED ---\n" << std::endl;
            failed++;
        }
    };

    run_test(test_basic_preprocessing, "BasicPreprocessing");
    run_test(test_different_input_sizes, "DifferentInputSizes");
    run_test(test_grayscale_input, "GrayscaleInput");
    run_test(test_normalization, "Normalization");
    run_test(test_invalid_inputs, "InvalidInputs");
    run_test(test_float_image_input, "FloatImageInput");
    run_test(test_aspect_ratio_preservation, "AspectRatioPreservation");
    run_test(test_output_range, "OutputRange");
    run_test(test_matches_original_clip, "MatchesOriginalCLIP");

    std::cout << "Test Summary:" << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;

    return failed == 0 ? 0 : 1;
}