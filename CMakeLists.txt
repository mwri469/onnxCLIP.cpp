cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(project_name print_vocab)
project(${project_name})

# Define path to LibTorch Instillation
# set(CMAKE_PREFIX_PATH /home/teknique/torchinstall/libtorch)

# Find torch library and define all required files
# find_package(Torch REQUIRED)
# find_library(Torchvision_LIBRARIES torchvision REQUIRED PATHS /usr/local/lib)
# set(CMAKE_CXX_FLAGS, "${CMAKE_CXX_FLAGS}, ${TORCH_CXX_FLAGS}")

# Include directories for Torchvision
# include_directories(/usr/local/include)
# Include directories for headers
include_directories(src/inference)

add_library(${project_name}-lib
        src/inference/tokenizer.cpp
        src/inference/tokeniser.h)

# Executable to add that will be compiled and run
add_executable(print_vocab 
                src/main.cpp
                src/inference/tokenizer.cpp)

# Link torch libs to the executable
# target_link_libraries(print_vocab "${TORCH_LIBRARIES}" "${Torchvision_LIBRARIES}")

