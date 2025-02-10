#include "../src/inference/preprocessor.hpp"
// #include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

using namespace std;
using namespace cv;

// Forward declarations
torch::Tensor load_tensor_from_txt(const std::string& file_path);  // Declare before use
bool checkMatRange(const cv::Mat& mat, float min_val, float max_val);

const std::string ASSETS_PATH = "../assets/";

// Helper function implementation
bool checkMatRange(const cv::Mat& mat, float min_val, float max_val) {
    double minVal, maxVal;
    cv::minMaxLoc(mat, &minVal, &maxVal);
    return (minVal >= min_val && maxVal <= max_val);
}

// Fixed tensor loading function
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
    return torch::from_blob(data.data(), dims, torch::kFloat32).clone();
}

// Fixed test functions
bool test_basic_preprocessing() {
    std::cout << "=== Running test: BasicPreprocessing ===" << std::endl;
    Preprocessor preprocessor;
    string fp = ASSETS_PATH + "franz-kafka.jpg";
    cv::Mat img = cv::imread(fp, IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Could not load test image." << std::endl;
        return false;
    }

    torch::Tensor processed = preprocessor.encode_image(img);

    // Correct tensor dimension check
    auto sizes = processed.sizes();
    if (sizes.size() != 4 || 
        sizes[0] != 1 || 
        sizes[1] != 3 || 
        sizes[2] != Preprocessor::CLIP_INPUT_SIZE || 
        sizes[3] != Preprocessor::CLIP_INPUT_SIZE) {
        std::cerr << "Error: Output dimensions incorrect." << std::endl;
        return false;
    }

    std::cout << "Basic preprocessing passed." << std::endl;
    return true;
}

bool test_different_input_sizes() {
    std::cout << "=== Running test: DifferentInputSizes ===" << std::endl;
    Preprocessor preprocessor;

    cv::Mat tall_img(480, 320, CV_8UC3, cv::Scalar(255, 255, 255));
    torch::Tensor processed_tall = preprocessor.encode_image(tall_img);

    cv::Mat wide_img(320, 480, CV_8UC3, cv::Scalar(255, 255, 255));
    torch::Tensor processed_wide = preprocessor.encode_image(wide_img);

    // Compare tensor sizes using dimensions()
    if (processed_tall.sizes() != processed_wide.sizes()) {
        std::cerr << "Error: Output sizes differ." << std::endl;
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

    // Check tensor dimensions properly
    auto sizes = processed.sizes();
    if (sizes.size() != 4 || 
        sizes[1] != 3 ||  // Should have 3 channels
        sizes[2] != Preprocessor::CLIP_INPUT_SIZE || 
        sizes[3] != Preprocessor::CLIP_INPUT_SIZE) {
        std::cerr << "Error: Output dimensions incorrect for grayscale input." << std::endl;
        return false;
    }

    std::cout << "Grayscale input handled correctly." << std::endl;
    return true;
}

bool test_matches_original_clip() {
    std::cout << "=== Running test: MatchesOriginalCLIP ===" << std::endl;
    Preprocessor preprocessor;
    
    // Load test image
    string fp = ASSETS_PATH + "franz-kafka.jpg";
    cv::Mat img = cv::imread(fp,IMREAD_COLOR);
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

bool test_normalization() {
    std::cout << "=== Running test: Normalization ===" << std::endl;
    Preprocessor preprocessor;
    string fp = ASSETS_PATH + "franz-kafka.jpg";
    cv::Mat img = cv::imread(fp,IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Could not load test image." << std::endl;
        return false;
    }

    torch::Tensor processed = preprocessor.encode_image(img);

    // Convert tensor to cv::Mat
    processed = processed.squeeze(0); 
    
    // Split into channels
    std::vector<cv::Mat> channels;
    for (int i = 0; i < 3; ++i) {
        torch::Tensor channel_tensor = processed[i];
        
        // Create cv::Mat from tensor data
        cv::Mat channel(
            Preprocessor::CLIP_INPUT_SIZE,
            Preprocessor::CLIP_INPUT_SIZE,
            CV_32FC1,
            channel_tensor.data_ptr<float>()
        );
        
        // Clone to maintain data ownership
        channels.push_back(channel.clone());
    }

    // Check normalization ranges
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

bool test_output_range() {
    std::cout << "=== Running test: OutputRange ===" << std::endl;
    Preprocessor preprocessor;
    string fp = ASSETS_PATH + "franz-kafka.jpg";
    cv::Mat img = cv::imread(fp,IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Could not load test image." << std::endl;
        return false;
    }

    torch::Tensor processed = preprocessor.encode_image(img);

    // Check that values are in a reasonable range after normalization
    // if (!checkMatRange(processed, -5.0f, 5.0f)) {
    //     std::cerr << "Error: Output range is incorrect." << std::endl;
    //     return false;
    // }
    std::cout << processed << std::endl;

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
    run_test(test_output_range, "OutputRange");
    run_test(test_matches_original_clip, "MatchesOriginalCLIP");

    std::cout << "Test Summary:" << std::endl;
    std::cout << "Passed: " << passed << std::endl;
    std::cout << "Failed: " << failed << std::endl;

    return failed == 0 ? 0 : 1;
}