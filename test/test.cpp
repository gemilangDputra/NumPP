#include <numpp/matrix.hpp>
#include <numpp/matrix/linalg/matmul.hpp>
#include <iostream>

int main() {
    numpp::matrix<double> a({
        {11.1, 12.2, 13.3, 14.4, 15.5},
        {21.6, 22.7, 23.8, 24.9, 25.0},
        {31.1, 32.2, 33.3, 34.4, 35.5},
        {41.6, 42.7, 43.8, 44.9, 45.0},
        {51.1, 52.2, 53.3, 54.4, 55.5}
    }, numpp::layout::rowmajor);

    numpp::matrix<double> b({
        {11.1, 21.2, 31.3, 41.4, 51.5},
        {21.6, 22.7, 23.8, 24.9, 25.0},
        {31.1, 32.2, 33.3, 34.4, 35.5},
        {41.6, 42.7, 43.8, 44.9, 45.0},
        {51.1, 52.2, 53.3, 54.4, 55.5}
    }, numpp::layout::colmajor);


    // Matrix khusus untuk testing slice: 5 × 6
    numpp::matrix<double> c({
        {11, 12, 13, 14, 15, 16},
        {21, 22, 23, 24, 25, 26},
        {31, 32, 33, 34, 35, 36},
        {41, 42, 43, 44, 45, 46},
        {51, 52, 53, 54, 55, 56}
    }, numpp::layout::rowmajor);


    // Matrix khusus untuk testing reshape: 3 × 5
    // 15 elements → reshape menjadi 3 × 5
    numpp::matrix<double> d({
        {11, 12, 13, 14, 15},
        {21, 22, 23, 24, 25},
        {31, 32, 33, 34, 35}
    }, numpp::layout::rowmajor);


    // Basic matrix multiplication
    auto test_1 = numpp::linalg::matmul(a, a);

    // Same operation with column-major matrices
    auto test_2 = numpp::linalg::matmul(b, b);

    // Different layouts
    auto test_3 = numpp::linalg::matmul(a, b);

    // Transpose view
    auto test_4 = numpp::linalg::matmul(
        a,
        numpp::transpose(b)
    );

    // Transpose view
    auto test_5 = numpp::linalg::matmul(
        a,
        numpp::transpose(a)
    );

    // Actual slice:
    //
    // c = 5 × 6
    // columns [1, 4) → 5 × 3
    //
    // a = 5 × 5
    // slice(c) = 5 × 3
    //
    // 5 × 5 × 5 × 3 → 5 × 3
    auto test_6 = numpp::linalg::matmul(
        a,
        numpp::slice(
            c,
            numpp::all,
            numpp::slice_range(1, 4, 1)
        )
    );

    // Actual reshape + transpose:
    //
    // d = 3 × 5 = 15 elements
    //
    // reshape(d, 3, 5) → 3 × 5
    // transpose        → 5 × 3
    //
    // a = 5 × 5
    //
    // 5 × 5 × 5 × 3 → 5 × 3
    auto test_7 = numpp::linalg::matmul(
        a,
        numpp::transpose(
            numpp::reshape(d, 3, 5)
        )
    );


    std::cout << test_1 << '\n';
    std::cout << test_2 << '\n';
    std::cout << test_3 << '\n';
    std::cout << test_4 << '\n';
    std::cout << test_5 << '\n';
    std::cout << test_6 << '\n';
    std::cout << test_7 << '\n';
}