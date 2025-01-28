cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(project_name print_vocab)
project(${project_name})

# Define path to LibTorch Instillation
# set(CMAKE_PREFIX_PATH /home/teknique/torchinstall/libtorch)

# Find torch library and define all required files
# find_package(Torch REQUIRED)
# find_library(Torchvision_LIBRARIES torchvision REQUIRED PATHS /usr/local/lib)
# set(CMAKE_CXX_FLAGS, "${CMAKE_CXX_FLAGS}, ${TORCH_CXX_FLAGS}")

find_package(GTest REQUIRED)

# Include directories for Torchvision
# include_directories(/usr/local/include)
# Include directories for headers
include_directories(src/inference)

add_library(${project_name}-lib
        src/inference/tokenizer.cpp
        src/inference/tokeniszer.hpp)

# Executable to add that will be compiled and run
add_executable(print_vocab 
                src/main.cpp
                src/inference/tokenizer.cpp)

add_executable(tokenizer_test
                tests/tokenizer_test.cpp)
target_link_libraries(tokenizer_test
                        PRIVATE
                        GTest::GTest
                        GTest::Main
                        ${project_name}-lib)      

# Enable testing
enable_testing()
add_test(NAME tokenizer_test COMMAND tokenizer_test)
# Link torch libs to the executable
# target_link_libraries(print_vocab "${TORCH_LIBRARIES}" "${Torchvision_LIBRARIES}")

