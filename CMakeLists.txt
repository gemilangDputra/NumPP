cmake_minimum_required(VERSION 3.10)
project(NUMPP VERSION 1.0 LANGUAGES CXX)

# ============================================================
# NUMPP
# ============================================================

add_library(NUMPP INTERFACE)

target_include_directories(NUMPP INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)


# ============================================================
# BLAS configuration
# ============================================================

set(NUMPP_BLAS "none"
    CACHE STRING "BLAS backend"
)

set(NUMPP_BLAS_HEADER ""
    CACHE STRING "BLAS C header"
)

set_property(CACHE NUMPP_BLAS PROPERTY STRINGS
    none
    openblas
)


# Default: BLAS disabled
set(NUMPP_USE_BLAS 0)


if(NUMPP_BLAS STREQUAL "openblas")

    if(NUMPP_BLAS_HEADER STREQUAL "")
        message(FATAL_ERROR
            "NUMPP_BLAS_HEADER must be specified when "
            "NUMPP_BLAS=openblas"
        )
    endif()

    find_package(BLAS REQUIRED)

    set(NUMPP_USE_BLAS 1)

    target_link_libraries(NUMPP INTERFACE
        BLAS::BLAS
    )

elseif(NOT NUMPP_BLAS STREQUAL "none")

    message(FATAL_ERROR
        "Unsupported NUMPP_BLAS='${NUMPP_BLAS}'"
    )

endif()


# ============================================================
# Generated configuration header
# ============================================================

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/numpp_blas_config.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/numpp/numpp_blas_config.hpp
    @ONLY
)

target_include_directories(NUMPP INTERFACE
    ${CMAKE_CURRENT_BINARY_DIR}/generated
)


# ============================================================
# Test
# ============================================================

add_executable(TEST
    test/test.cpp
)

target_link_libraries(TEST PRIVATE
    NUMPP
)