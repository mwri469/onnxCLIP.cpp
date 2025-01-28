cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(project_name clip_cpp)
project(${project_name})

###############################################################################
#### INCLUDES #################################################################
###############################################################################

find_package(GTest REQUIRED)

include_directories(src/inference)

###############################################################################
#### DEPENDENCIES #############################################################
###############################################################################

# None

###############################################################################
#### SOURCES ##################################################################
###############################################################################

add_library(${project_name}-lib
        src/inference/tokenizer.cpp
        src/inference/tokeniszer.hpp
        src/inference/preprocessor.hpp
        src/inference/preprocessor.cpp)

###############################################################################
#### GENERATE OUTPUT ##########################################################
###############################################################################

# Executable to add that will be compiled and run
add_executable(clip_cpp 
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

