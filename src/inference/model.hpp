#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "preprocessor.h"
#include "tokenizer.h"

class OnnxClip {
public:
    // Constructor
    OnnxClip(const std::string& model = "ViT-B/32", 
             int batch_size = 0,
             bool silent_download = false,
             const std::string& cache_dir = "");

    // Main public interface
    cv::Mat getImageEmbeddings(const std::vector<cv::Mat>& images, bool with_batching = true);
    cv::Mat getTextEmbeddings(const std::vector<std::string>& texts, bool with_batching = true);

    // Helper functions for similarity scoring
    static cv::Mat getSimilarityScores(const cv::Mat& embeddings1, const cv::Mat& embeddings2);
    static cv::Mat cosineSimilarity(const cv::Mat& embeddings1, const cv::Mat& embeddings2);
    static cv::Mat softmax(const cv::Mat& x);

    // Getters
    int getEmbeddingSize() const { return embedding_size; }

private:
    // Private helper functions
    static std::pair<std::unique_ptr<Ort::Session>, std::unique_ptr<Ort::Session>> 
    _loadModels(const std::string& model, bool silent, const std::string& cache_dir);
    
    static std::unique_ptr<Ort::Session> 
    _loadModel(const std::string& path, bool silent);
    
    cv::Mat 
	getEmptyEmbedding() const;
    
    template<typename T>
    std::vector<std::vector<T>> 
	_toBatches(const std::vector<T>& items, int size) const;
    
    static void 
	_downloadFile(const std::string& url, const std::string& path);
    static cv::Mat 
	_normalizeEmbeddings(const cv::Mat& embeddings);
private:
	private:
	int 							embedding_size;
    int 							batch_size;
    std::unique_ptr<Preprocessor> 	preprocessor;
    std::unique_ptr<CLIPTokenizer> 	tokenizer;
    std::unique_ptr<Ort::Session> 	image_model;
    std::unique_ptr<Ort::Session> 	text_model;
    Ort::Env 						env;
};
