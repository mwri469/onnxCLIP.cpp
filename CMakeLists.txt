cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(project_name clip_cpp)
project(${project_name})

###############################################################################
#### INCLUDES #################################################################
###############################################################################

# Set paths
set(LIBTORCH_DIR "${CMAKE_SOURCE_DIR}/libtorch")
set(ONNXRUNTIME_DIR "${CMAKE_SOURCE_DIR}/onnxruntime-linux-x64-1.20.1")
set(GTEST_DIR "${CMAKE_SOURCE_DIR}/googletest/build") 

list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_DIR}")
find_package(Torch REQUIRED)
find_package(GTest REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(src/inference)

# Manually specify gtest libraries
set(GTEST_LIBRARY "${GTEST_DIR}/lib/libgtest.a")
set(GTEST_MAIN_LIBRARY "${GTEST_DIR}/lib/libgtest_main.a")

###############################################################################
#### DEPENDENCIES #############################################################
###############################################################################

# None

###############################################################################
#### SOURCES ##################################################################
###############################################################################

add_library(${project_name}-lib
        src/inference/tokenizer.cpp
        src/inference/tokenizer.hpp
        src/inference/preprocessor.hpp
        src/inference/preprocessor.cpp)

target_link_libraries(${project_name}-lib
        PUBLIC ${OpenCV_LIBS}
        PUBLIC ${TORCH_LIBRARIES})

# add_library(${project_name}-testing)
# target_link_libraries(${project_name}-testing
#         GTest::GTest
#         GTest::Main
#         ${project_name}-lib)

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
                ${GTEST_LIBRARY}
                ${GTEST_MAIN_LIBRARY}
                ${project_name}-lib)

add_executable(preprocessor_test
                tests/preprocessor_test.cpp)
target_link_libraries(preprocessor_test
                ${GTEST_LIBRARY}
                ${GTEST_MAIN_LIBRARY}
                ${project_name}-lib)                   

# Enable testing
enable_testing()
add_test(NAME tokenizer_test COMMAND tokenizer_test)
add_test(NAME preprocessor_test COMMAND preprocessor_test)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")