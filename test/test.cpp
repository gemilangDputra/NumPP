#include <numpp/matrix.hpp>
#include <numpp/matrix/linalg/matmul.hpp>

#include <eigen3/Eigen/Dense>

#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <chrono>

template<numpp::numpp_matrix MAT, typename Derived>
bool compare_benchmark(
    const MAT& numpp,
    double numpp_time,
    const Eigen::MatrixBase<Derived>& eigen,
    double eigen_time,
    double abs_tolerance = 1e-3,
    double rel_tolerance = 1e-5
) {
    // ============================================================
    // Shape
    // ============================================================

    if (numpp.row() != static_cast<std::size_t>(eigen.rows()) ||
        numpp.col() != static_cast<std::size_t>(eigen.cols())) {

        std::cerr
            << "[MISMATCH] Shape\n"
            << "  NUMPP : "
            << numpp.row() << "x" << numpp.col() << '\n'
            << "  Eigen : "
            << eigen.rows() << "x" << eigen.cols() << '\n';

        return false;
    }

    // ============================================================
    // Values
    // ============================================================

    for (std::size_t i = 0; i < numpp.row(); ++i) {
        for (std::size_t j = 0; j < numpp.col(); ++j) {

            const double numpp_value =
                static_cast<double>(numpp(i, j));

            const double eigen_value =
                static_cast<double>(
                    eigen(
                        static_cast<Eigen::Index>(i),
                        static_cast<Eigen::Index>(j)
                    )
                );

            const double diff =
                std::abs(numpp_value - eigen_value);

            // Combined absolute + relative tolerance:
            //
            // |a - b| <= atol + rtol * max(|a|, |b|)
            //
            const double scale =
                std::max(
                    std::abs(numpp_value),
                    std::abs(eigen_value)
                );

            const double allowed_error =
                abs_tolerance + rel_tolerance * scale;

            if (diff > allowed_error) {
                std::cerr
                    << "[MISMATCH] Value\n"
                    << "  position       : ("
                    << i << ", " << j << ")\n"
                    << "  NUMPP          : " << numpp_value << '\n'
                    << "  Eigen          : " << eigen_value << '\n'
                    << "  difference     : " << diff << '\n'
                    << "  allowed error  : " << allowed_error << '\n'
                    << "  abs tolerance  : " << abs_tolerance << '\n'
                    << "  rel tolerance  : " << rel_tolerance << '\n'
                    << "  NUMPP time     : " << numpp_time << " ms\n"
                    << "  Eigen time     : " << eigen_time << " ms\n";

                return false;
            }
        }
    }

    // ============================================================
    // Match + Benchmark
    // ============================================================

    std::cout
        << "[MATCH]"
        << "  NUMPP = " << numpp_time << " ms"
        << "  Eigen = " << eigen_time << " ms";

    if (numpp_time > 0.0 && eigen_time > 0.0) {

        if (numpp_time < eigen_time) {
            std::cout
                << "  NUMPP speedup = "
                << eigen_time / numpp_time
                << "x";
        }
        else if (eigen_time < numpp_time) {
            std::cout
                << "  Eigen speedup = "
                << numpp_time / eigen_time
                << "x";
        }
        else {
            std::cout
                << "  speedup = 1x";
        }
    }

    std::cout << '\n';

    return true;
}

template<typename A, typename B, typename Op>
auto benchmark(const A& a, const B& b, Op op) {
    using clock = std::chrono::steady_clock;
    const auto start = clock::now();
    auto result = op(a, b);
    const auto end = clock::now();
    const double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    return std::pair{
        std::move(result),
        elapsed
    };
}

int main() {
    using EigenMatrixViewRow = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;
    using EigenMatrixViewCol = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>>;

    std::vector<float> vector512x512(512*512);
    std::vector<float> vector512w1024(1024*512);
    std::vector<float> vector1024x1024(1024*1024);

    for (size_t i = 0; i < vector512x512.size(); ++i)   vector512x512[i] = static_cast<float>(i+1);
    for (size_t i = 0; i < vector512w1024.size(); ++i)  vector512w1024[i] = static_cast<float>(i+1);
    for (size_t i = 0; i < vector1024x1024.size(); ++i) vector1024x1024[i] = static_cast<float>(i+1);

    numpp::matrix_view numpp_matrix512x512Row   = numpp::reshape(vector512x512,    512,  512, numpp::layout::rowmajor);
    numpp::matrix_view numpp_matrix1024x1024Row = numpp::reshape(vector1024x1024, 1024, 1024, numpp::layout::rowmajor);
    numpp::matrix_view numpp_matrix512x1024Row  = numpp::reshape(vector512w1024,   512, 1024, numpp::layout::rowmajor);
    numpp::matrix_view numpp_matrix1024x512Row  = numpp::reshape(vector512w1024,  1024,  512, numpp::layout::rowmajor);
    
    numpp::matrix_view numpp_matrix512x512Col   = numpp::reshape(vector512x512,    512,  512, numpp::layout::colmajor);
    numpp::matrix_view numpp_matrix1024x1024Col = numpp::reshape(vector1024x1024, 1024, 1024, numpp::layout::colmajor);
    numpp::matrix_view numpp_matrix512x1024Col  = numpp::reshape(vector512w1024,   512, 1024, numpp::layout::colmajor);
    numpp::matrix_view numpp_matrix1024x512Col  = numpp::reshape(vector512w1024,  1024,  512, numpp::layout::colmajor);

    EigenMatrixViewRow eigen_matrix512x512Row(vector512x512.data(),     512, 512);
    EigenMatrixViewRow eigen_matrix1024x1024Row(vector1024x1024.data(), 1024, 1024);
    EigenMatrixViewRow eigen_matrix512x1024Row(vector512w1024.data(),   512, 1024);
    EigenMatrixViewRow eigen_matrix1024x512Row(vector512w1024.data(),   1024, 512);
    
    EigenMatrixViewCol eigen_matrix512x512Col(vector512x512.data(),     512, 512);
    EigenMatrixViewCol eigen_matrix1024x1024Col(vector1024x1024.data(), 1024, 1024);
    EigenMatrixViewCol eigen_matrix512x1024Col(vector512w1024.data(),   512, 1024);
    EigenMatrixViewCol eigen_matrix1024x512Col(vector512w1024.data(),   1024, 512);

    std::cout << "mulai!\n";

    auto [numpp_c1, numpp_c1_time] = benchmark(
        numpp_matrix512x512Row,
        numpp_matrix512x512Row,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c2, numpp_c2_time] = benchmark(
        numpp_matrix1024x1024Row,
        numpp_matrix1024x1024Row,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c3, numpp_c3_time] = benchmark(
        numpp_matrix1024x512Row,
        numpp_matrix512x1024Row,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c4, numpp_c4_time] = benchmark(
        numpp_matrix512x1024Row,
        numpp_matrix1024x512Row,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c5, numpp_c5_time] = benchmark(
        numpp_matrix512x512Row,
        numpp_matrix512x512Row.transpose(),
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c6, numpp_c6_time] = benchmark(
        numpp_matrix512x512Row.transpose(),
        numpp_matrix512x512Row,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c7, numpp_c7_time] = benchmark(
        numpp_matrix1024x512Row,
        numpp_matrix1024x512Row.transpose(),
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c8, numpp_c8_time] = benchmark(
        numpp_matrix512x1024Row.transpose(),
        numpp_matrix512x1024Row,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c9, numpp_c9_time] = benchmark(
        numpp_matrix1024x512Row,
        numpp_matrix1024x512Row.transpose(),
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c10, numpp_c10_time] = benchmark(
        numpp_matrix1024x512Row.transpose(),
        numpp_matrix1024x512Row,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c11, numpp_c11_time] = benchmark(
        numpp_matrix512x512Row,
        numpp_matrix1024x1024Row.slice(
            numpp::slice_range(0, 512),
            numpp::slice_range(0, 512)
        ),
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c12, numpp_c12_time] = benchmark(
        numpp_matrix512x512Row,
        numpp_matrix1024x1024Row.slice(
            numpp::slice_range(0, 512),
            numpp::slice_range(512, 1024)
        ),
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c13, numpp_c13_time] = benchmark(
        numpp_matrix512x512Row,
        numpp_matrix1024x1024Row.slice(
            numpp::slice_range(512, 1024),
            numpp::slice_range(0, 512)
        ),
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c14, numpp_c14_time] = benchmark(
        numpp_matrix512x512Row,
        numpp_matrix1024x1024Row.slice(
            numpp::slice_range(0, 512),
            numpp::slice_range(0, 1024, 2)
        ),
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c15, numpp_c15_time] = benchmark(
        numpp_matrix512x512Row,
        numpp_matrix1024x1024Row.slice(
            numpp::slice_range(0, 1024, 2),
            numpp::slice_range(0, 512)
        ),
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c16, numpp_c16_time] = benchmark(
        numpp_matrix512x512Row,
        numpp_matrix1024x1024Row.slice(
            numpp::slice_range(0, 1024, 2),
            numpp::slice_range(0, 1024, 2)
        ),
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c17, numpp_c17_time] = benchmark(
        numpp_matrix1024x1024Row.slice(
            numpp::slice_range(0, 512),
            numpp::slice_range(0, 512)
        ),
        numpp_matrix512x512Row,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c18, numpp_c18_time] = benchmark(
        numpp_matrix1024x1024Row.slice(
            numpp::slice_range(0, 512),
            numpp::slice_range(512, 1024)
        ),
        numpp_matrix512x512Row,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c19, numpp_c19_time] = benchmark(
        numpp_matrix1024x1024Row.slice(
            numpp::slice_range(512, 1024),
            numpp::slice_range(0, 512)
        ),
        numpp_matrix512x512Row,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c20, numpp_c20_time] = benchmark(
        numpp_matrix1024x1024Row.slice(
            numpp::slice_range(0, 512),
            numpp::slice_range(0, 1024, 2)
        ),
        numpp_matrix512x512Row,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c21, numpp_c21_time] = benchmark(
        numpp_matrix1024x1024Row.slice(
            numpp::slice_range(0, 1024, 2),
            numpp::slice_range(0, 512)
        ),
        numpp_matrix512x512Row,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c22, numpp_c22_time] = benchmark(
        numpp_matrix1024x1024Row.slice(
            numpp::slice_range(0, 1024, 2),
            numpp::slice_range(0, 1024, 2)
        ),
        numpp_matrix512x512Row,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c23, numpp_c23_time] = benchmark(
        numpp_matrix512x512Col,
        numpp_matrix512x512Col,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c24, numpp_c24_time] = benchmark(
        numpp_matrix1024x1024Col,
        numpp_matrix1024x1024Col,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c25, numpp_c25_time] = benchmark(
        numpp_matrix1024x512Col,
        numpp_matrix512x1024Col,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c26, numpp_c26_time] = benchmark(
        numpp_matrix512x1024Col,
        numpp_matrix1024x512Col,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c27, numpp_c27_time] = benchmark(
        numpp_matrix512x512Col,
        numpp_matrix512x512Col.transpose(),
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c28, numpp_c28_time] = benchmark(
        numpp_matrix512x512Col.transpose(),
        numpp_matrix512x512Col,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c29, numpp_c29_time] = benchmark(
        numpp_matrix1024x512Col,
        numpp_matrix1024x512Col.transpose(),
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c30, numpp_c30_time] = benchmark(
        numpp_matrix512x1024Col.transpose(),
        numpp_matrix512x1024Col,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    std::cout << "numpp 1:2\n";

    auto [numpp_c31, numpp_c31_time] = benchmark(
        numpp_matrix1024x512Col,
        numpp_matrix1024x512Col.transpose(),
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c32, numpp_c32_time] = benchmark(
        numpp_matrix1024x512Col.transpose(),
        numpp_matrix1024x512Col,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c33, numpp_c33_time] = benchmark(
        numpp_matrix512x512Col,
        numpp_matrix1024x1024Col.slice(
            numpp::slice_range(0, 512),
            numpp::slice_range(0, 512)
        ),
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c34, numpp_c34_time] = benchmark(
        numpp_matrix512x512Col,
        numpp_matrix1024x1024Col.slice(
            numpp::slice_range(0, 512),
            numpp::slice_range(512, 1024)
        ),
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c35, numpp_c35_time] = benchmark(
        numpp_matrix512x512Col,
        numpp_matrix1024x1024Col.slice(
            numpp::slice_range(512, 1024),
            numpp::slice_range(0, 512)
        ),
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c36, numpp_c36_time] = benchmark(
        numpp_matrix512x512Col,
        numpp_matrix1024x1024Col.slice(
            numpp::slice_range(0, 512),
            numpp::slice_range(0, 1024, 2)
        ),
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c37, numpp_c37_time] = benchmark(
        numpp_matrix512x512Col,
        numpp_matrix1024x1024Col.slice(
            numpp::slice_range(0, 1024, 2),
            numpp::slice_range(0, 512)
        ),
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c38, numpp_c38_time] = benchmark(
        numpp_matrix512x512Col,
        numpp_matrix1024x1024Col.slice(
            numpp::slice_range(0, 1024, 2),
            numpp::slice_range(0, 1024, 2)
        ),
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c39, numpp_c39_time] = benchmark(
        numpp_matrix1024x1024Col.slice(
            numpp::slice_range(0, 512),
            numpp::slice_range(0, 512)
        ),
        numpp_matrix512x512Col,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c40, numpp_c40_time] = benchmark(
        numpp_matrix1024x1024Col.slice(
            numpp::slice_range(0, 512),
            numpp::slice_range(512, 1024)
        ),
        numpp_matrix512x512Col,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c41, numpp_c41_time] = benchmark(
        numpp_matrix1024x1024Col.slice(
            numpp::slice_range(512, 1024),
            numpp::slice_range(0, 512)
        ),
        numpp_matrix512x512Col,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c42, numpp_c42_time] = benchmark(
        numpp_matrix1024x1024Col.slice(
            numpp::slice_range(0, 512),
            numpp::slice_range(0, 1024, 2)
        ),
        numpp_matrix512x512Col,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c43, numpp_c43_time] = benchmark(
        numpp_matrix1024x1024Col.slice(
            numpp::slice_range(0, 1024, 2),
            numpp::slice_range(0, 512)
        ),
        numpp_matrix512x512Col,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c44, numpp_c44_time] = benchmark(
        numpp_matrix1024x1024Col.slice(
            numpp::slice_range(0, 1024, 2),
            numpp::slice_range(0, 1024, 2)
        ),
        numpp_matrix512x512Col,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c45, numpp_c45_time] = benchmark(
        numpp_matrix512x512Row,
        numpp_matrix512x512Col,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c46, numpp_c46_time] = benchmark(
        numpp_matrix1024x1024Row,
        numpp_matrix1024x1024Col,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    // 512x1024 * 1024x512 -> 512x512
    auto [numpp_c47, numpp_c47_time] = benchmark(
        numpp_matrix512x1024Row,
        numpp_matrix1024x512Col,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    // 1024x512 * 512x1024 -> 1024x1024
    auto [numpp_c48, numpp_c48_time] = benchmark(
        numpp_matrix1024x512Row,
        numpp_matrix512x1024Col,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c49, numpp_c49_time] = benchmark(
        numpp_matrix512x512Col,
        numpp_matrix512x512Row,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c50, numpp_c50_time] = benchmark(
        numpp_matrix1024x1024Col,
        numpp_matrix1024x1024Row,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    // 512x1024 * 1024x512 -> 512x512
    auto [numpp_c51, numpp_c51_time] = benchmark(
        numpp_matrix512x1024Col,
        numpp_matrix1024x512Row,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    // 1024x512 * 512x1024 -> 1024x1024
    auto [numpp_c52, numpp_c52_time] = benchmark(
        numpp_matrix1024x512Col,
        numpp_matrix512x1024Row,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c53, numpp_c53_time] = benchmark(
        numpp_matrix512x512Row,
        numpp_matrix512x512Col.transpose(),
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c54, numpp_c54_time] = benchmark(
        numpp_matrix512x512Row.transpose(),
        numpp_matrix512x512Col,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c55, numpp_c55_time] = benchmark(
        numpp_matrix512x512Col,
        numpp_matrix512x512Row.transpose(),
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c56, numpp_c56_time] = benchmark(
        numpp_matrix512x512Col.transpose(),
        numpp_matrix512x512Row,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c57, numpp_c57_time] = benchmark(
        numpp_matrix1024x1024Row,
        numpp_matrix1024x1024Col.transpose(),
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c58, numpp_c58_time] = benchmark(
        numpp_matrix1024x1024Row.transpose(),
        numpp_matrix1024x1024Col,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c59, numpp_c59_time] = benchmark(
        numpp_matrix1024x1024Col,
        numpp_matrix1024x1024Row.transpose(),
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    auto [numpp_c60, numpp_c60_time] = benchmark(
        numpp_matrix1024x1024Col.transpose(),
        numpp_matrix1024x1024Row,
        [](const auto& a, const auto& b) {
            return numpp::matmul(a, b);
        }
    );

    std::cout << "numpp selesai!\n";

auto [eigen_c1, eigen_c1_time] =
    benchmark(
        eigen_matrix512x512Row,
        eigen_matrix512x512Row,
        [](const auto& a, const auto& b) {
            return (a * b).eval();
        }
    );

auto [eigen_c5, eigen_c5_time] =
    benchmark(
        eigen_matrix512x512Row,
        eigen_matrix512x512Row,
        [](const auto& a, const auto& b) {
            return (a * b.transpose()).eval();
        }
    );

auto [eigen_c6, eigen_c6_time] =
    benchmark(
        eigen_matrix512x512Row,
        eigen_matrix512x512Row,
        [](const auto& a, const auto& b) {
            return (a.transpose() * b).eval();
        }
    );

    auto [eigen_c2, eigen_c2_time] =
        benchmark(
            eigen_matrix1024x1024Row,
            eigen_matrix1024x1024Row,
            [](const auto& a, const auto& b) {
                return (a * b).eval();
            }
        );

    auto [eigen_c3, eigen_c3_time] =
        benchmark(
            eigen_matrix1024x512Row,
            eigen_matrix512x1024Row,
            [](const auto& a, const auto& b) {
                return (a * b).eval();
            }
        );

    auto [eigen_c4, eigen_c4_time] =
        benchmark(
            eigen_matrix512x1024Row,
            eigen_matrix1024x512Row,
            [](const auto& a, const auto& b) {
                return (a * b).eval();
            }
        );

    auto [eigen_c7, eigen_c7_time] =
        benchmark(
            eigen_matrix1024x512Row,
            eigen_matrix1024x512Row,
            [](const auto& a, const auto& b) {
                return (a * b.transpose()).eval();
            }
        );

    auto [eigen_c8, eigen_c8_time] =
        benchmark(
            eigen_matrix512x1024Row,
            eigen_matrix512x1024Row,
            [](const auto& a, const auto& b) {
                return (a.transpose() * b).eval();
            }
        );

    auto [eigen_c9, eigen_c9_time] =
        benchmark(
            eigen_matrix1024x512Row,
            eigen_matrix1024x512Row,
            [](const auto& a, const auto& b) {
                return (a * b.transpose()).eval();
            }
        );

    auto [eigen_c10, eigen_c10_time] =
        benchmark(
            eigen_matrix1024x512Row,
            eigen_matrix1024x512Row,
            [](const auto& a, const auto& b) {
                return (a.transpose() * b).eval();
            }
        );


    // ------------------------------------------------------------
    // C11 - C16
    // ------------------------------------------------------------

    auto [eigen_c11, eigen_c11_time] =
        benchmark(
            eigen_matrix512x512Row,
            eigen_matrix1024x1024Row.block(0, 0, 512, 512),
            [](const auto& a, const auto& b) {
                return (a * b).eval();
            }
        );

    auto [eigen_c12, eigen_c12_time] =
        benchmark(
            eigen_matrix512x512Row,
            eigen_matrix1024x1024Row.block(0, 512, 512, 512),
            [](const auto& a, const auto& b) {
                return (a * b).eval();
            }
        );
// ============================================================
// C14 - C16
// ============================================================

// ============================================================
// C13
// ============================================================

auto [eigen_c13, eigen_c13_time] =
    benchmark(
        eigen_matrix512x512Row,
        eigen_matrix1024x1024Row.block(
            512, 0, 512, 512
        ),
        [](const auto& a, const auto& b) {
            return (a * b).eval();
        }
    );

using EigenRowStride =
    Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;

using EigenRowStridedMap =
    Eigen::Map<
        Eigen::Matrix<
            float,
            Eigen::Dynamic,
            Eigen::Dynamic,
            Eigen::RowMajor
        >,
        0,
        EigenRowStride
    >;


// ------------------------------------------------------------
// C14: rows 0:512, cols 0:1024 step 2
//
// NUMPP:
// B.slice(
//     slice_range(0, 512),
//     slice_range(0, 1024, 2)
// )
//
// Logical shape : 512 x 512
// Row stride     : 1
// Col stride     : 2
//
// RowMajor Eigen:
// inner stride = column stride = 2
// outer stride = row stride * source columns
//              = 1 * 1024
// ------------------------------------------------------------

// ============================================================
// C14 - C16
// ============================================================

using EigenRowStride =
    Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;

using EigenRowStridedMap =
    Eigen::Map<
        Eigen::Matrix<
            float,
            Eigen::Dynamic,
            Eigen::Dynamic,
            Eigen::RowMajor
        >,
        0,
        EigenRowStride
    >;


// ------------------------------------------------------------
// C14
//
// NUMPP:
//
// rows: slice_range(0, 512)
// cols: slice_range(0, 1024, 2)
//
// rows = 0,1,2,...,511
// cols = 0,2,4,...,1022
//
// row step = 1
// col step = 2
// ------------------------------------------------------------

EigenRowStridedMap eigen_row_c14(
    vector1024x1024.data(),
    512,
    512,
    EigenRowStride(
        1024,  // outer stride
        2      // inner stride
    )
);

auto [eigen_c14, eigen_c14_time] =
    benchmark(
        eigen_matrix512x512Row,
        eigen_row_c14,
        [](const auto& a, const auto& b) {
            return (a * b).eval();
        }
    );


// ------------------------------------------------------------
// C15
//
// NUMPP:
//
// rows: slice_range(0, 1024, 2)
// cols: slice_range(0, 512)
//
// rows = 0,2,4,...,1022
// cols = 0,1,2,...,511
//
// row step = 2
// col step = 1
// ------------------------------------------------------------

EigenRowStridedMap eigen_row_c15(
    vector1024x1024.data(),
    512,
    512,
    EigenRowStride(
        2048,  // outer stride = 2 * 1024
        1      // inner stride
    )
);

auto [eigen_c15, eigen_c15_time] =
    benchmark(
        eigen_matrix512x512Row,
        eigen_row_c15,
        [](const auto& a, const auto& b) {
            return (a * b).eval();
        }
    );


// ------------------------------------------------------------
// C16
//
// NUMPP:
//
// rows: slice_range(0, 1024, 2)
// cols: slice_range(0, 1024, 2)
//
// rows = 0,2,4,...,1022
// cols = 0,2,4,...,1022
//
// row step = 2
// col step = 2
// ------------------------------------------------------------

EigenRowStridedMap eigen_row_c16(
    vector1024x1024.data(),
    512,
    512,
    EigenRowStride(
        2048,  // outer stride = 2 * 1024
        2      // inner stride
    )
);

auto [eigen_c16, eigen_c16_time] =
    benchmark(
        eigen_matrix512x512Row,
        eigen_row_c16,
        [](const auto& a, const auto& b) {
            return (a * b).eval();
        }
    );


// ============================================================
// C17 - C22
// ============================================================


// ------------------------------------------------------------
// C17
//
// NUMPP:
// B.slice(
//     slice_range(0, 512),
//     slice_range(0, 512)
// )
// ------------------------------------------------------------

auto [eigen_c17, eigen_c17_time] =
    benchmark(
        eigen_matrix1024x1024Row.block(
            0, 0, 512, 512
        ),
        eigen_matrix512x512Row,
        [](const auto& a, const auto& b) {
            return (a * b).eval();
        }
    );


// ------------------------------------------------------------
// C18
//
// NUMPP:
// B.slice(
//     slice_range(0, 512),
//     slice_range(512, 1024)
// )
// ------------------------------------------------------------

auto [eigen_c18, eigen_c18_time] =
    benchmark(
        eigen_matrix1024x1024Row.block(
            0, 512, 512, 512
        ),
        eigen_matrix512x512Row,
        [](const auto& a, const auto& b) {
            return (a * b).eval();
        }
    );


// ------------------------------------------------------------
// C19
//
// NUMPP:
// B.slice(
//     slice_range(512, 1024),
//     slice_range(0, 512)
// )
// ------------------------------------------------------------

auto [eigen_c19, eigen_c19_time] =
    benchmark(
        eigen_matrix1024x1024Row.block(
            512, 0, 512, 512
        ),
        eigen_matrix512x512Row,
        [](const auto& a, const auto& b) {
            return (a * b).eval();
        }
    );


// ------------------------------------------------------------
// C20
//
// NUMPP:
// A = B.slice(
//     slice_range(0, 512),
//     slice_range(0, 1024, 2)
// )
//
// row step = 1
// col step = 2
//
// Jadi sama dengan C14.
// ------------------------------------------------------------

auto [eigen_c20, eigen_c20_time] =
    benchmark(
        eigen_row_c14,
        eigen_matrix512x512Row,
        [](const auto& a, const auto& b) {
            return (a * b).eval();
        }
    );


// ------------------------------------------------------------
// C21
//
// NUMPP:
// A = B.slice(
//     slice_range(0, 1024, 2),
//     slice_range(0, 512)
// )
//
// row step = 2
// col step = 1
//
// Sama dengan C15.
// ------------------------------------------------------------

auto [eigen_c21, eigen_c21_time] =
    benchmark(
        eigen_row_c15,
        eigen_matrix512x512Row,
        [](const auto& a, const auto& b) {
            return (a * b).eval();
        }
    );


// ------------------------------------------------------------
// C22
//
// NUMPP:
// A = B.slice(
//     slice_range(0, 1024, 2),
//     slice_range(0, 1024, 2)
// )
//
// row step = 2
// col step = 2
//
// Sama dengan C16.
// ------------------------------------------------------------

auto [eigen_c22, eigen_c22_time] =
    benchmark(
        eigen_row_c16,
        eigen_matrix512x512Row,
        [](const auto& a, const auto& b) {
            return (a * b).eval();
        }
    );


// ============================================================
// C23 - C32
// ============================================================


// ------------------------------------------------------------
// C23
// ------------------------------------------------------------

auto [eigen_c23, eigen_c23_time] =
    benchmark(
        eigen_matrix512x512Col,
        eigen_matrix512x512Col,
        [](const auto& a, const auto& b) {
            return (a * b).eval();
        }
    );


// ------------------------------------------------------------
// C24
// ------------------------------------------------------------

auto [eigen_c24, eigen_c24_time] =
    benchmark(
        eigen_matrix1024x1024Col,
        eigen_matrix1024x1024Col,
        [](const auto& a, const auto& b) {
            return (a * b).eval();
        }
    );


// ------------------------------------------------------------
// C25
// ------------------------------------------------------------

auto [eigen_c25, eigen_c25_time] =
    benchmark(
        eigen_matrix1024x512Col,
        eigen_matrix512x1024Col,
        [](const auto& a, const auto& b) {
            return (a * b).eval();
        }
    );


// ------------------------------------------------------------
// C26
// ------------------------------------------------------------

auto [eigen_c26, eigen_c26_time] =
    benchmark(
        eigen_matrix512x1024Col,
        eigen_matrix1024x512Col,
        [](const auto& a, const auto& b) {
            return (a * b).eval();
        }
    );


// ------------------------------------------------------------
// C27
//
// A * B.transpose()
// ------------------------------------------------------------

auto [eigen_c27, eigen_c27_time] =
    benchmark(
        eigen_matrix512x512Col,
        eigen_matrix512x512Col,
        [](const auto& a, const auto& b) {
            return (a * b.transpose()).eval();
        }
    );


// ------------------------------------------------------------
// C28
//
// A.transpose() * B
// ------------------------------------------------------------

auto [eigen_c28, eigen_c28_time] =
    benchmark(
        eigen_matrix512x512Col,
        eigen_matrix512x512Col,
        [](const auto& a, const auto& b) {
            return (a.transpose() * b).eval();
        }
    );


// ------------------------------------------------------------
// C29
//
// A * B.transpose()
// ------------------------------------------------------------

auto [eigen_c29, eigen_c29_time] =
    benchmark(
        eigen_matrix1024x512Col,
        eigen_matrix1024x512Col,
        [](const auto& a, const auto& b) {
            return (a * b.transpose()).eval();
        }
    );


// ------------------------------------------------------------
// C30
//
// A.transpose() * B
// ------------------------------------------------------------

auto [eigen_c30, eigen_c30_time] =
    benchmark(
        eigen_matrix512x1024Col,
        eigen_matrix512x1024Col,
        [](const auto& a, const auto& b) {
            return (a.transpose() * b).eval();
        }
    );


// ------------------------------------------------------------
// C31
//
// A * B.transpose()
// ------------------------------------------------------------

auto [eigen_c31, eigen_c31_time] =
    benchmark(
        eigen_matrix1024x512Col,
        eigen_matrix1024x512Col,
        [](const auto& a, const auto& b) {
            return (a * b.transpose()).eval();
        }
    );


// ------------------------------------------------------------
// C32
//
// A.transpose() * B
// ------------------------------------------------------------

auto [eigen_c32, eigen_c32_time] =
    benchmark(
        eigen_matrix1024x512Col,
        eigen_matrix1024x512Col,
        [](const auto& a, const auto& b) {
            return (a.transpose() * b).eval();
        }
    );


// ============================================================
// C33 - C35
// ============================================================


// ------------------------------------------------------------
// C33
//
// NUMPP:
// A = 512x512
// B = B1024x1024.slice(
//         rows 0:512,
//         cols 0:512
//     )
// ------------------------------------------------------------

auto [eigen_c33, eigen_c33_time] =
    benchmark(
        eigen_matrix512x512Col,
        eigen_matrix1024x1024Col.block(
            0, 0, 512, 512
        ),
        [](const auto& a, const auto& b) {
            return (a * b).eval();
        }
    );


// ------------------------------------------------------------
// C34
//
// NUMPP:
// rows 0:512
// cols 512:1024
// ------------------------------------------------------------

auto [eigen_c34, eigen_c34_time] =
    benchmark(
        eigen_matrix512x512Col,
        eigen_matrix1024x1024Col.block(
            0, 512, 512, 512
        ),
        [](const auto& a, const auto& b) {
            return (a * b).eval();
        }
    );


// ------------------------------------------------------------
// C35
//
// NUMPP:
// rows 512:1024
// cols 0:512
// ------------------------------------------------------------

auto [eigen_c35, eigen_c35_time] =
    benchmark(
        eigen_matrix512x512Col,
        eigen_matrix1024x1024Col.block(
            512, 0, 512, 512
        ),
        [](const auto& a, const auto& b) {
            return (a * b).eval();
        }
    );


// ============================================================
// C36 - C37
// ============================================================

using EigenColStride =
    Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;

using EigenColStridedMap =
    Eigen::Map<
        Eigen::Matrix<
            float,
            Eigen::Dynamic,
            Eigen::Dynamic,
            Eigen::ColMajor
        >,
        0,
        EigenColStride
    >;


// ------------------------------------------------------------
// C36
//
// NUMPP:
//
// rows: slice_range(0, 512)
// cols: slice_range(0, 1024, 2)
//
// row step = 1
// col step = 2
//
// ColumnMajor:
// inner = row step = 1
// outer = col step * 1024 = 2048
// ------------------------------------------------------------

EigenColStridedMap eigen_col_c36(
    vector1024x1024.data(),
    512,
    512,
    EigenColStride(
        2048,  // outer stride
        1      // inner stride
    )
);

auto [eigen_c36, eigen_c36_time] =
    benchmark(
        eigen_matrix512x512Col,
        eigen_col_c36,
        [](const auto& a, const auto& b) {
            return (a * b).eval();
        }
    );


// ------------------------------------------------------------
// C37
//
// NUMPP:
//
// rows: slice_range(0, 1024, 2)
// cols: slice_range(0, 512)
//
// row step = 2
// col step = 1
//
// ColumnMajor:
// inner = row step = 2
// outer = col step * 1024 = 1024
// ------------------------------------------------------------

EigenColStridedMap eigen_col_c37(
    vector1024x1024.data(),
    512,
    512,
    EigenColStride(
        1024,  // outer stride
        2      // inner stride
    )
);

auto [eigen_c37, eigen_c37_time] =
    benchmark(
        eigen_matrix512x512Col,
        eigen_col_c37,
        [](const auto& a, const auto& b) {
            return (a * b).eval();
        }
    );
    EigenColStridedMap eigen_col_c38(
        vector1024x1024.data(),
        512,
        512,
        EigenColStride(2 * 1024, 2)
    );

    auto [eigen_c38, eigen_c38_time] =
        benchmark(
            eigen_matrix512x512Col,
            eigen_col_c38,
            [](const auto& a, const auto& b) {
                return (a * b).eval();
            }
        );


    // ============================================================
    // C39 - C44
    // ============================================================

    auto [eigen_c39, eigen_c39_time] =
        benchmark(
            eigen_matrix1024x1024Col.block(0, 0, 512, 512),
            eigen_matrix512x512Col,
            [](const auto& a, const auto& b) {
                return (a * b).eval();
            }
        );

    auto [eigen_c40, eigen_c40_time] =
        benchmark(
            eigen_matrix1024x1024Col.block(0, 512, 512, 512),
            eigen_matrix512x512Col,
            [](const auto& a, const auto& b) {
                return (a * b).eval();
            }
        );

    auto [eigen_c41, eigen_c41_time] =
        benchmark(
            eigen_matrix1024x1024Col.block(512, 0, 512, 512),
            eigen_matrix512x512Col,
            [](const auto& a, const auto& b) {
                return (a * b).eval();
            }
        );

    auto [eigen_c42, eigen_c42_time] =
        benchmark(
            eigen_col_c36,
            eigen_matrix512x512Col,
            [](const auto& a, const auto& b) {
                return (a * b).eval();
            }
        );

    auto [eigen_c43, eigen_c43_time] =
        benchmark(
            eigen_col_c37,
            eigen_matrix512x512Col,
            [](const auto& a, const auto& b) {
                return (a * b).eval();
            }
        );

    auto [eigen_c44, eigen_c44_time] =
        benchmark(
            eigen_col_c38,
            eigen_matrix512x512Col,
            [](const auto& a, const auto& b) {
                return (a * b).eval();
            }
        );


// ============================================================
// C45 - C52
// ============================================================

auto [eigen_c45, eigen_c45_time] =
    benchmark(
        eigen_matrix512x512Row,
        eigen_matrix512x512Col,
        [](const auto& a, const auto& b) {
            return (a * b).eval();
        }
    );

auto [eigen_c46, eigen_c46_time] =
    benchmark(
        eigen_matrix1024x1024Row,
        eigen_matrix1024x1024Col,
        [](const auto& a, const auto& b) {
            return (a * b).eval();
        }
    );

// 512x1024 * 1024x512 -> 512x512
auto [eigen_c47, eigen_c47_time] =
    benchmark(
        eigen_matrix512x1024Row,
        eigen_matrix1024x512Col,
        [](const auto& a, const auto& b) {
            return (a * b).eval();
        }
    );

// 1024x512 * 512x1024 -> 1024x1024
auto [eigen_c48, eigen_c48_time] =
    benchmark(
        eigen_matrix1024x512Row,
        eigen_matrix512x1024Col,
        [](const auto& a, const auto& b) {
            return (a * b).eval();
        }
    );

auto [eigen_c49, eigen_c49_time] =
    benchmark(
        eigen_matrix512x512Col,
        eigen_matrix512x512Row,
        [](const auto& a, const auto& b) {
            return (a * b).eval();
        }
    );

auto [eigen_c50, eigen_c50_time] =
    benchmark(
        eigen_matrix1024x1024Col,
        eigen_matrix1024x1024Row,
        [](const auto& a, const auto& b) {
            return (a * b).eval();
        }
    );

// 512x1024 * 1024x512 -> 512x512
auto [eigen_c51, eigen_c51_time] =
    benchmark(
        eigen_matrix512x1024Col,
        eigen_matrix1024x512Row,
        [](const auto& a, const auto& b) {
            return (a * b).eval();
        }
    );

// 1024x512 * 512x1024 -> 1024x1024
auto [eigen_c52, eigen_c52_time] =
    benchmark(
        eigen_matrix1024x512Col,
        eigen_matrix512x1024Row,
        [](const auto& a, const auto& b) {
            return (a * b).eval();
        }
    );


// ============================================================
// C53 - C60
// ============================================================

auto [eigen_c53, eigen_c53_time] =
    benchmark(
        eigen_matrix512x512Row,
        eigen_matrix512x512Col,
        [](const auto& a, const auto& b) {
            return (a * b.transpose()).eval();
        }
    );

auto [eigen_c54, eigen_c54_time] =
    benchmark(
        eigen_matrix512x512Row,
        eigen_matrix512x512Col,
        [](const auto& a, const auto& b) {
            return (a.transpose() * b).eval();
        }
    );

auto [eigen_c55, eigen_c55_time] =
    benchmark(
        eigen_matrix512x512Col,
        eigen_matrix512x512Row,
        [](const auto& a, const auto& b) {
            return (a * b.transpose()).eval();
        }
    );

auto [eigen_c56, eigen_c56_time] =
    benchmark(
        eigen_matrix512x512Col,
        eigen_matrix512x512Row,
        [](const auto& a, const auto& b) {
            return (a.transpose() * b).eval();
        }
    );

auto [eigen_c57, eigen_c57_time] =
    benchmark(
        eigen_matrix1024x1024Row,
        eigen_matrix1024x1024Col,
        [](const auto& a, const auto& b) {
            return (a * b.transpose()).eval();
        }
    );

auto [eigen_c58, eigen_c58_time] =
    benchmark(
        eigen_matrix1024x1024Row,
        eigen_matrix1024x1024Col,
        [](const auto& a, const auto& b) {
            return (a.transpose() * b).eval();
        }
    );

auto [eigen_c59, eigen_c59_time] =
    benchmark(
        eigen_matrix1024x1024Col,
        eigen_matrix1024x1024Row,
        [](const auto& a, const auto& b) {
            return (a * b.transpose()).eval();
        }
    );

auto [eigen_c60, eigen_c60_time] =
    benchmark(
        eigen_matrix1024x1024Col,
        eigen_matrix1024x1024Row,
        [](const auto& a, const auto& b) {
            return (a.transpose() * b).eval();
        }
    );



compare_benchmark(numpp_c1,  numpp_c1_time,  eigen_c1,  eigen_c1_time);
compare_benchmark(numpp_c2,  numpp_c2_time,  eigen_c2,  eigen_c2_time);
compare_benchmark(numpp_c3,  numpp_c3_time,  eigen_c3,  eigen_c3_time);
compare_benchmark(numpp_c4,  numpp_c4_time,  eigen_c4,  eigen_c4_time);
compare_benchmark(numpp_c5,  numpp_c5_time,  eigen_c5,  eigen_c5_time);
compare_benchmark(numpp_c6,  numpp_c6_time,  eigen_c6,  eigen_c6_time);
compare_benchmark(numpp_c7,  numpp_c7_time,  eigen_c7,  eigen_c7_time);
compare_benchmark(numpp_c8,  numpp_c8_time,  eigen_c8,  eigen_c8_time);
compare_benchmark(numpp_c9,  numpp_c9_time,  eigen_c9,  eigen_c9_time);
compare_benchmark(numpp_c10, numpp_c10_time, eigen_c10, eigen_c10_time);

compare_benchmark(numpp_c11, numpp_c11_time, eigen_c11, eigen_c11_time);
compare_benchmark(numpp_c12, numpp_c12_time, eigen_c12, eigen_c12_time);
compare_benchmark(numpp_c13, numpp_c13_time, eigen_c13, eigen_c13_time);
compare_benchmark(numpp_c14, numpp_c14_time, eigen_c14, eigen_c14_time);
compare_benchmark(numpp_c15, numpp_c15_time, eigen_c15, eigen_c15_time);
compare_benchmark(numpp_c16, numpp_c16_time, eigen_c16, eigen_c16_time);

compare_benchmark(numpp_c17, numpp_c17_time, eigen_c17, eigen_c17_time);
compare_benchmark(numpp_c18, numpp_c18_time, eigen_c18, eigen_c18_time);
compare_benchmark(numpp_c19, numpp_c19_time, eigen_c19, eigen_c19_time);
compare_benchmark(numpp_c20, numpp_c20_time, eigen_c20, eigen_c20_time);
compare_benchmark(numpp_c21, numpp_c21_time, eigen_c21, eigen_c21_time);
compare_benchmark(numpp_c22, numpp_c22_time, eigen_c22, eigen_c22_time);

compare_benchmark(numpp_c23, numpp_c23_time, eigen_c23, eigen_c23_time);
compare_benchmark(numpp_c24, numpp_c24_time, eigen_c24, eigen_c24_time);
compare_benchmark(numpp_c25, numpp_c25_time, eigen_c25, eigen_c25_time);
compare_benchmark(numpp_c26, numpp_c26_time, eigen_c26, eigen_c26_time);
compare_benchmark(numpp_c27, numpp_c27_time, eigen_c27, eigen_c27_time);
compare_benchmark(numpp_c28, numpp_c28_time, eigen_c28, eigen_c28_time);
compare_benchmark(numpp_c29, numpp_c29_time, eigen_c29, eigen_c29_time);
compare_benchmark(numpp_c30, numpp_c30_time, eigen_c30, eigen_c30_time);
compare_benchmark(numpp_c31, numpp_c31_time, eigen_c31, eigen_c31_time);
compare_benchmark(numpp_c32, numpp_c32_time, eigen_c32, eigen_c32_time);

compare_benchmark(numpp_c33, numpp_c33_time, eigen_c33, eigen_c33_time);
compare_benchmark(numpp_c34, numpp_c34_time, eigen_c34, eigen_c34_time);
compare_benchmark(numpp_c35, numpp_c35_time, eigen_c35, eigen_c35_time);
compare_benchmark(numpp_c36, numpp_c36_time, eigen_c36, eigen_c36_time);
compare_benchmark(numpp_c37, numpp_c37_time, eigen_c37, eigen_c37_time);
compare_benchmark(numpp_c38, numpp_c38_time, eigen_c38, eigen_c38_time);

compare_benchmark(numpp_c39, numpp_c39_time, eigen_c39, eigen_c39_time);
compare_benchmark(numpp_c40, numpp_c40_time, eigen_c40, eigen_c40_time);
compare_benchmark(numpp_c41, numpp_c41_time, eigen_c41, eigen_c41_time);
compare_benchmark(numpp_c42, numpp_c42_time, eigen_c42, eigen_c42_time);
compare_benchmark(numpp_c43, numpp_c43_time, eigen_c43, eigen_c43_time);
compare_benchmark(numpp_c44, numpp_c44_time, eigen_c44, eigen_c44_time);

compare_benchmark(numpp_c45, numpp_c45_time, eigen_c45, eigen_c45_time);
compare_benchmark(numpp_c46, numpp_c46_time, eigen_c46, eigen_c46_time);
compare_benchmark(numpp_c47, numpp_c47_time, eigen_c47, eigen_c47_time);
compare_benchmark(numpp_c48, numpp_c48_time, eigen_c48, eigen_c48_time);
compare_benchmark(numpp_c49, numpp_c49_time, eigen_c49, eigen_c49_time);
compare_benchmark(numpp_c50, numpp_c50_time, eigen_c50, eigen_c50_time);
compare_benchmark(numpp_c51, numpp_c51_time, eigen_c51, eigen_c51_time);
compare_benchmark(numpp_c52, numpp_c52_time, eigen_c52, eigen_c52_time);

compare_benchmark(numpp_c53, numpp_c53_time, eigen_c53, eigen_c53_time);
compare_benchmark(numpp_c54, numpp_c54_time, eigen_c54, eigen_c54_time);
compare_benchmark(numpp_c55, numpp_c55_time, eigen_c55, eigen_c55_time);
compare_benchmark(numpp_c56, numpp_c56_time, eigen_c56, eigen_c56_time);
compare_benchmark(numpp_c57, numpp_c57_time, eigen_c57, eigen_c57_time);
compare_benchmark(numpp_c58, numpp_c58_time, eigen_c58, eigen_c58_time);
compare_benchmark(numpp_c59, numpp_c59_time, eigen_c59, eigen_c59_time);
compare_benchmark(numpp_c60, numpp_c60_time, eigen_c60, eigen_c60_time);
}