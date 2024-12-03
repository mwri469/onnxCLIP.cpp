cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(CLIP_inference)

# Set paths
set(LIBTORCH_DIR "${CMAKE_SOURCE_DIR}/libtorch")
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-1.20.1")

# Find LibTorch
list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_DIR}")
find_package(Torch REQUIRED)

# Include ONNX Runtime directories
include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

# Add your executable
add_executable(CLIP_inference clip_inference.cpp)

# Link libraries
target_link_libraries(CLIP_inference 
    ${TORCH_LIBRARIES}
    onnxruntime
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
