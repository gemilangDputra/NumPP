cmake_minimum_required(VERSION 3.10)

project(NUMPP VERSION 1.0 LANGUAGES CXX)
add_library(NUMPP INTERFACE)

target_include_directories(NUMPP INTERFACE include)

find_package(OpenBLAS REQUIRED)

add_executable(main main.cpp)

target_compile_definitions(main PRIVATE
    NUMPP_USE_BLAS=1
    NUMPP_USE_LAPACK=1
    NUMPP_BLAS_HEADER='<openblas/cblas.h'
    NUMPP_LAPACK_HEADER='<openblas/lapacke.h'
)

target_link_libraries(main PRIVATE
    OpenBLAS::OpenBLAS
    NUMPP
)