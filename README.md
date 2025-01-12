# C++ implementation for inference of ONNX CLIP models

The aim of this project is for a C++ library capble of inferencing CLIP models under the ONNX umbrella.

To learn more, visit [onnx.ai](https://onnx.ai)

```bash
$ mkdir build && cd ./build
$ cmake ..
$ make
```

## Requirements

First, install [libtorch](https://pytorch.org/get-started/locally/) and torchvision, which are required dependencies of this project.

```bash
$ wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-cxx11-abi-shared-with-deps-latest.zip
$ unzip libtorch-cxx11-abi-shared-with-deps-latest.zip
$ git clone git@github.com:pytorch/vision.git
$ cd vision
$ mkdir build
```

Edit ./vision/CMakeLists.txt to add CMAKE_PREFIX_PATH and turn off all options:

```cmake
cmake_minimum_required(VERSION 3.18)
project(torchvision)
set(CMAKE_CXX_STANDARD 17)
file(STRINGS version.txt TORCHVISION_VERSION)

set(CMAKE_PREFIX_PATH /path/to/clip_onnx_cpp_inference/libtorch)

option(WITH_CUDA "Enable CUDA support" OFF)
option(WITH_MPS "Enable MPS support" OFF)
option(WITH_PNG "Enable features requiring LibPNG." OFF)
option(WITH_JPEG "Enable features requiring LibJPEG." OFF)
# Libwebp is disabled by default, which means enabling it from cmake is largely
# untested. Since building from cmake is very low pri anyway, this is OK. If
# you're a user and you need this, please open an issue (and a PR!).
option(WITH_WEBP "Enable features requiring LibWEBP." OFF)
# Same here
option(WITH_AVIF "Enable features requiring LibAVIF." OFF)

# ... rest of file ...
```

If you have libtorch/vision installed locally, ./clip_onnx_cpp_inference/CMakeLists.txt can be modified to instead use your local installation.

Further, for ./clip_onnx_cpp_inference/tests, an install of Google Test is required.