#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <torchvision/vision.h>
#include <torchvision/ops/nms.h>

torch::Tensor bin_to_tensor(const std::string &fp, int rows, int cols) {
	// Open file and read in binary
	std::ifstream file(fp, std::ios::binary);
	if (!file.is_open()) {
		throw std::runtime_error("Failed to open file: " + fp);
	}

	// Determine size based on file and expected dims
	std::vector<float> buffer(rows * cols);
	file.read(reinterpret_cast<char*>(buffer.data()), rows*cols*sizeof(float));
	file.close();
	
	// Convert to tensor and reshape
	return torch::from_blob(buffer.data(), {rows,cols}).clone();
}

int main(int argc, char const *argv[]) {
	if (argc != 4) {
		std::cerr << "Usage: " << argv[0] << "<detections> <features> <output_binary_filepath>";
		return 1;
	}

	// Get inputs
	int detections = std::stoi(argv[1]);
	int features = std::stoi(argv[2]);
	std::string filepath = argv[3];

	// Load in binary
	torch::Tensor output_tensor = bin_to_tensor(filepath, detections, features);

	// Extract bounding boxes & confidence scores
	std::cout << "Tensor size: " << output_tensor.sizes() <<std::endl;
	torch::Tensor boxes = output_tensor.slice(1,0,4); // cols [0->3]
	torch::Tensor conf = output_tensor.slice(1, -1, output_tensor.size(1)).squeeze(); 
	
	// Verification of tensor size
	// int row_indx = 0;
	// torch::Tensor row = output_tensor[row_indx];
	// std::cout << "Row " << row_indx << ": " << row << std::endl;
	// std::cout << output_tensor.sizes() << std::endl;
	// std::cout << boxes.sizes() << std::endl;
	// std::cout << conf.sizes() << std::endl;
	double iou_threshold = 0.5;
		
	// Apply NMS
	torch::Tensor selected_indices = vision::ops::nms(boxes, conf, iou_threshold);
	// std::cout << "Selected indices: " << selected_indices << std::endl;
	return 0;
}
