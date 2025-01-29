#include "model.h"
#include <filesystem>
#include <fstream>
#include <curl/curl.h>
#include <spdlog/spdlog.h>

// Constructor implementation
OnnxClip::OnnxClip(const std::string& model, int batch_size, 
                   bool silent_download, const std::string& cache_dir) 
    : batch_size(batch_size), env(ORT_LOGGING_LEVEL_WARNING, "CLIP") {
    
    // Set embedding size based on model
    if (model == "ViT-B/32") {
        embedding_size = 512;
    } else if (model == "RN50") {
        embedding_size = 1024;
    } else {
        throw std::invalid_argument("Unsupported model: " + model);
    }

    // Initialize preprocessor and tokenizer
    preprocessor = std::make_unique<Preprocessor>();
    tokenizer = std::make_unique<CLIPTokenizer>("../src/data/bpe_simple_vocab_16e6.txt");

    // Load ONNX models
    auto [img_model, txt_model] = loadModels(model, silent_download, 
        cache_dir.empty() ? "../src/data" : cache_dir);
    
    image_model = std::move(img_model);
    text_model = std::move(txt_model);
}

// Implementation of image embedding generation
cv::Mat OnnxClip::getImageEmbeddings(const std::vector<cv::Mat>& images, bool with_batching) {
    if (!with_batching || batch_size == 0) {
        std::vector<cv::Mat> processed_images;
        for (const auto& image : images) {
            processed_images.push_back(preprocessor->encodeImage(image));
        }
        
        if (processed_images.empty()) {
            return getEmptyEmbedding();
        }

        // Concatenate processed images
        cv::Mat batch;
        cv::vconcat(processed_images, batch);

        // Prepare ONNX runtime input
        std::vector<float> input_tensor_values(batch.ptr<float>(), 
            batch.ptr<float>() + batch.total());
        
        std::vector<int64_t> input_shape = {static_cast<int64_t>(images.size()), 3, 224, 224};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(),
            input_shape.data(), input_shape.size());

        // Run inference
        const char* input_names[] = {"IMAGE"};
        const char* output_names[] = {"OUTPUT"};
        
        auto output_tensors = image_model->Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

        // Convert output to cv::Mat
        auto& output_tensor = output_tensors[0];
        auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
        
        cv::Mat output(output_shape[0], embedding_size, CV_32F,
                      output_tensor.GetTensorMutableData<float>());
        
        return output.clone();  // Clone to ensure data ownership
    } else {
        // Handle batching
        std::vector<cv::Mat> embeddings;
        auto batches = toBatches(images, batch_size);
        
        for (const auto& batch : batches) {
            embeddings.push_back(getImageEmbeddings(batch, false));
        }

        if (embeddings.empty()) {
            return getEmptyEmbedding();
        }

        cv::Mat result;
        cv::vconcat(embeddings, result);
        return result;
    }
}

// Implementation of text embedding generation
cv::Mat OnnxClip::getTextEmbeddings(const std::vector<std::string>& texts, bool with_batching) {
    if (!with_batching || batch_size == 0) {
        if (texts.empty()) {
            return getEmptyEmbedding();
        }

        // Tokenize texts
        std::vector<std::vector<int>> tokenized;
        for (const auto& text : texts) {
            tokenized.push_back(tokenizer->encode_text(text, 77, true));
        }

        // Convert to tensor format
        std::vector<int64_t> tokens_flat;
        for (const auto& tokens : tokenized) {
            tokens_flat.insert(tokens_flat.end(), tokens.begin(), tokens.end());
        }

        // Prepare ONNX runtime input
        std::vector<int64_t> input_shape = {static_cast<int64_t>(texts.size()), 77};
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info, tokens_flat.data(), tokens_flat.size(),
            input_shape.data(), input_shape.size());

        // Run inference
        const char* input_names[] = {"TEXT"};
        const char* output_names[] = {"OUTPUT"};
        
        auto output_tensors = text_model->Run(
            Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);

        // Convert output to cv::Mat
        auto& output_tensor = output_tensors[0];
        auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();
        
        cv::Mat output(output_shape[0], embedding_size, CV_32F,
                      output_tensor.GetTensorMutableData<float>());
        
        return output.clone();
    } else {
        // Handle batching
        std::vector<cv::Mat> embeddings;
        auto batches = toBatches(texts, batch_size);
        
        for (const auto& batch : batches) {
            embeddings.push_back(getTextEmbeddings(batch, false));
        }

        if (embeddings.empty()) {
            return getEmptyEmbedding();
        }

        cv::Mat result;
        cv::vconcat(embeddings, result);
        return result;
    }
}

// Similarity scoring implementations
cv::Mat OnnxClip::getSimilarityScores(const cv::Mat& embeddings1, const cv::Mat& embeddings2) {
    if (embeddings1.rows == 1) {
        cv::Mat result = getSimilarityScores(embeddings1.reshape(1, embeddings1.total()), embeddings2);
        return result.reshape(1, 1);
    }
    if (embeddings2.rows == 1) {
        cv::Mat result = getSimilarityScores(embeddings1, embeddings2.reshape(1, embeddings2.total()));
        return result.reshape(1, result.total());
    }
    return cosineSimilarity(embeddings1, embeddings2) * 100;
}

cv::Mat OnnxClip::cosineSimilarity(const cv::Mat& embeddings1, const cv::Mat& embeddings2) {
    cv::Mat norm_emb1 = normalizeEmbeddings(embeddings1);
    cv::Mat norm_emb2 = normalizeEmbeddings(embeddings2);
    return norm_emb1 * norm_emb2.t();
}

cv::Mat OnnxClip::softmax(const cv::Mat& x) {
    cv::Mat exp_mat;
    cv::exp(x, exp_mat);
    cv::Mat sum;
    cv::reduce(exp_mat, sum, 1, cv::REDUCE_SUM);
    return exp_mat / cv::repeat(sum, 1, x.cols);
}

// Private helper implementations
cv::Mat OnnxClip::_normalizeEmbeddings(const cv::Mat& embeddings) {
    cv::Mat normalized;
    cv::normalize(embeddings, normalized, 1.0, 0.0, cv::NORM_L2, CV_32F);
    return normalized;
}

cv::Mat OnnxClip::getEmptyEmbedding() const {
    return cv::Mat(0, embedding_size, CV_32F);
}

template<typename T>
std::vector<std::vector<T>> OnnxClip::toBatches(const std::vector<T>& items, int size) const {
    if (size < 1) {
        throw std::invalid_argument("Batch size must be positive");
    }

    std::vector<std::vector<T>> batches;
    std::vector<T> current_batch;
    
    for (const auto& item : items) {
        current_batch.push_back(item);
        if (current_batch.size() == size) {
            batches.push_back(current_batch);
            current_batch.clear();
        }
    }
    
    if (!current_batch.empty()) {
        batches.push_back(current_batch);
    }
    
    return batches;
}

// Model loading implementations
std::pair<std::unique_ptr<Ort::Session>, std::unique_ptr<Ort::Session>> 
OnnxClip::_loadModels(const std::string& model, bool silent, const std::string& cache_dir) {
    std::string image_model_file;
    std::string text_model_file;
    
    if (model == "ViT-B/32") {
        image_model_file = "clip_image_model_vitb32.onnx";
        text_model_file = "clip_text_model_vitb32.onnx";
    } else if (model == "RN50") {
        image_model_file = "clip_image_model_rn50.onnx";
        text_model_file = "clip_text_model_rn50.onnx";
    }

    std::filesystem::path cache_path(cache_dir);
    auto image_path = cache_path / image_model_file;
    auto text_path = cache_path / text_model_file;

    return {
        loadModel(image_path.string(), silent),
        loadModel(text_path.string(), silent)
    };
}

std::unique_ptr<Ort::Session> OnnxClip::_loadModel(const std::string& path, bool silent) {
    try {
        if (std::filesystem::exists(path)) {
            return std::make_unique<Ort::Session>(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "CLIP"), 
                                                path.c_str(), 
                                                Ort::SessionOptions{nullptr});
        }
    } catch (const Ort::Exception& e) {
        if (!silent) {
            spdlog::info("Failed to load existing model: {}", e.what());
        }
    }

    // Model doesn't exist or is invalid, download it
    std::string basename = std::filesystem::path(path).filename().string();
    std::string url = "https://lakera-clip.s3.eu-west-1.amazonaws.com/" + basename;
    
    if (!silent) {
        spdlog::info("Downloading model from {}", url);
    }

    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    
    // Download to temporary file first
    std::string temp_path = path + ".part";
    downloadFile(url, temp_path);
    std::filesystem::rename(temp_path, path);

    return std::make_unique<Ort::Session>(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "CLIP"), 
                                        path.c_str(), 
                                        Ort::SessionOptions{nullptr});
}

// File download implementation using libcurl
size_t writeCallback(void* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* file = static_cast<std::ofstream*>(userdata);
    file->write(static_cast<char*>(ptr), size * nmemb);
    return size * nmemb;
}

void OnnxClip::_downloadFile(const std::string& url, const std::string& path) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        throw std::runtime_error("Failed to initialize CURL");
    }

    std::ofstream file(path, std::ios::binary);
    
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &file);
    curl_easy_setopt(curl, CURLOPT
