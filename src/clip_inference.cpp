#include <iostream>
#include <vector>
#include <torch/torch.h>
#include <torch/script.h>
#include <onnxruntime_cxx_api.h>

class CLIPInference {
private:
    Ort::Env env;
    Ort::Session session;
    Ort::MemoryInfo memory_info;

    // Helper function to convert vector to Ort::Value
    Ort::Value vectorToOrtValue(const std::vector<float>& tensor, 
                                 const std::vector<int64_t>& shape) {
        return Ort::Value::CreateTensor<float>(
            memory_info, 
            const_cast<float*>(tensor.data()), 
            tensor.size(), 
            shape.data(), 
            shape.size()
        );
    }

public:
    CLIPInference(const std::string& model_path) 
        : env(ORT_LOGGING_LEVEL_WARNING),
          session(env, model_path.c_str(), Ort::SessionOptions{}),
          memory_info(Ort::MemoryInfo::CreateCpu(
              OrtArenaAllocator, OrtMemTypeDefault)) {
        
        // Print input and output node information
        for (int i = 0; i < session.GetInputCount(); ++i) {
            Ort::AllocatedStringPtr name(session.GetInputNameAllocated(i, allocator));
            std::cout << "Input " << i << " : " << name.get() << std::endl;
        }
        
        for (int i = 0; i < session.GetOutputCount(); ++i) {
            Ort::AllocatedStringPtr name(session.GetOutputNameAllocated(i, allocator));
            std::cout << "Output " << i << " : " << name.get() << std::endl;
        }
    }

    // Perform inference on text input
    std::vector<float> textInference(const std::string& text) {
        // Preprocess text (you'll need to implement tokenization specific to your CLIP model)
        std::vector<float> text_tensor = preprocessText(text);
        std::vector<int64_t> input_shape = {1, text_tensor.size()}; // Adjust based on your model

        // Prepare input
        std::vector<Ort::Value> inputs;
        inputs.push_back(vectorToOrtValue(text_tensor, input_shape));

        // Prepare output
        std::vector<Ort::Value> outputs;
        std::vector<int64_t> output_shape = {1, 512}; // Adjust to match your model's embedding size
        float* output_tensor = new float[512]; // Adjust size as needed

        // Run inference
        outputs.push_back(Ort::Value::CreateTensor<float>(
            memory_info, 
            output_tensor, 
            512, 
            output_shape.data(), 
            output_shape.size()
        ));

        // Run session
        session.Run(
            Ort::RunOptions{}, 
            session.GetInputNames().data(), 
            inputs.data(), 
            inputs.size(),
            session.GetOutputNames().data(), 
            outputs.data(), 
            outputs.size()
        );

        // Convert output to vector
        std::vector<float> embedding(output_tensor, output_tensor + 512);
        delete[] output_tensor;

        return embedding;
    }
};

int main() {
    try {
        // Path to your ONNX model
        std::string model_path = "/home/teknique/Documents/clipx/CLIP/onnx_models/clip_text_vitb32_224x224.onnx";
        
        // Create inference instance
        CLIPInference clip_inference(model_path);

        // Example usage
        std::string text_query = "A photo of a cat";
        auto text_embedding = clip_inference.textInference(text_query);

        // Print embedding
        std::cout << "Text Embedding (first 10 values): ";
        for (int i = 0; i < 10; ++i) {
            std::cout << text_embedding[i] << " ";
        }
        std::cout << std::endl;

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
