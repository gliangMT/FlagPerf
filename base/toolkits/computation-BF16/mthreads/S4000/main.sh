#!/bin/bash

# 设置编译器和参数
CXX=g++
MCC=mcc
CXXFLAGS="-std=c++17 -I./include -I/usr/local/musa/include -fPIC"
MCCFLAGS="-std=c++17 --offload-arch=mp_22 -I../include -mtgpu -fPIC -I./include -I/usr/local/musa/include"
LDFLAGS="-lmusart -L/usr/local/musa/lib"

# 输出目录
SRC_DIR=src
BUILD_DIR=build
# BIN_DIR=bin
EXECUTABLE=gemm

# 创建输出目录
mkdir -p $BUILD_DIR
# mkdir -p $BIN_DIR

# 编译 .cpp 文件
# echo "Compiling benchmark.cpp, common.cpp and logger.cpp..."
$CXX $CXXFLAGS -c $SRC_DIR/common.cpp -o $BUILD_DIR/common.o
$CXX $CXXFLAGS -c $SRC_DIR/logger.cpp -o $BUILD_DIR/logger.o
$CXX $CXXFLAGS -c $SRC_DIR/benchmark.cpp -o $BUILD_DIR/benchmark.o

# 编译 .cu 文件
# echo "Compiling compute_mma_bf16.cu..."
$MCC $MCCFLAGS -c $SRC_DIR/compute_mma_bf16.mu -o $BUILD_DIR/compute_mma_bf16.o

# 编译主文件
# echo "Compiling main.cpp..."
$CXX $CXXFLAGS -c $SRC_DIR/main.cpp -o $BUILD_DIR/main.o

# 链接所有目标文件
# echo "Linking objects..."
$CXX $CXXFLAGS $BUILD_DIR/*.o -o $EXECUTABLE $LDFLAGS

# 编译完成
# echo "Build successful! Executable created at $EXECUTABLE"

./gemm
