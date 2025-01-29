cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

set(project_name clip_cpp)
project(${project_name})

###############################################################################
#### INCLUDES #################################################################
###############################################################################

find_package(GTest REQUIRED)
find_package(OpenCV REQUIRED)

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
        src/inference/tokenizer.hpp
        src/inference/preprocessor.hpp
        src/inference/preprocessor.cpp)

target_link_libraries(${project_name}-lib
        PUBLIC ${OpenCV_LIBS})

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

###############################################################################
#### TESTING ##################################################################
###############################################################################

add_executable(tokenizer_test
                tests/tokenizer_test.cpp)
target_link_libraries(tokenizer_test
                PRIVATE
                GTest::GTest
                GTest::Main
                ${project_name}-lib)

add_executable(preprocessor_test
                tests/preprocessor_test.cpp)
target_link_libraries(preprocessor_test
                PRIVATE
                GTest::GTest
                GTest::Main
                ${project_name}-lib)                   

# Enable testing
enable_testing()
add_test(NAME tokenizer_test COMMAND tokenizer_test)
add_test(NAME preprocessor_test COMMAND preprocessor_test)

