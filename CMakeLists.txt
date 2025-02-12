cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
set(project_name clip_cpp)
project(${project_name})

# Set paths
set(LIBTORCH_DIR "${CMAKE_SOURCE_DIR}/libtorch")
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-1.20.1")
set(OpenCV_DIR "${CMAKE_SOURCE_DIR}/opencv/build")

list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_DIR}")
find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(src/inference)

# Library
add_library(${project_name}-lib
        src/inference/tokenizer.cpp
        src/inference/tokenizer.hpp
        src/inference/preprocessor.hpp
        src/inference/preprocessor.cpp)

target_link_libraries(${project_name}-lib
        PUBLIC ${OpenCV_LIBS}
        PUBLIC ${TORCH_LIBRARIES})

###############################################################################
#### GENERATE OUTPUT ##########################################################
###############################################################################

# Executable to add that will be compiled and run
add_executable(clip_cpp 
                src/main.cpp)
target_link_libraries(clip_cpp
                ${project_name}-lib)

###############################################################################
#### TESTING ##################################################################
###############################################################################
add_executable(tokenizer_test
                tests/tokenizer_test.cpp)
target_link_libraries(tokenizer_test
                ${project_name}-lib
                pthread)
                

add_executable(preprocessor_test
                tests/preprocessor_test.cpp)
target_link_libraries(preprocessor_test
                ${project_name}-lib
                pthread)                

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
