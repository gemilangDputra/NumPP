#include <numpp/matrix.hpp>
#include <eigen3/Eigen/Dense>

#include <cassert>
#include <cstddef>
#include <iostream>
#include <stdexcept>

template<numpp::numpp_matrix A, typename Derived>
bool compare(
    const A& a,
    const Eigen::DenseBase<Derived>& b
) {
    if (a.row() != b.rows() || a.col() != b.cols())
        return false;

    for (std::size_t i = 0; i < a.row(); ++i) {
        for (std::size_t j = 0; j < a.col(); ++j) {
            if (a(i, j) != b(i, j))
                return false;
        }
    }

    return true;
}

int main() {
    using Matrix = Eigen::Matrix<
        float,
        Eigen::Dynamic,
        Eigen::Dynamic,
        Eigen::RowMajor
    >;

    // ============================================================
    // NumPP
    // ============================================================

    numpp::matrixf np_a({
        { 1,  2,  3,  4,  5},
        { 6,  7,  8,  9, 10},
        {11, 12, 13, 14, 15},
        {16, 17, 18, 19, 20},
        {21, 22, 23, 24, 25}
    });

    numpp::matrixf np_b({
        {26, 27, 28, 29, 30},
        {31, 32, 33, 34, 35},
        {36, 37, 38, 39, 40},
        {41, 42, 43, 44, 45},
        {46, 47, 48, 49, 50}
    });

    numpp::matrixf np_broadcast({
        {1, 2, 3, 4, 5}
    });


    // ============================================================
    // Eigen
    // ============================================================

    Matrix e_a(5, 5);
    e_a <<  1,  2,  3,  4,  5,
             6,  7,  8,  9, 10,
            11, 12, 13, 14, 15,
            16, 17, 18, 19, 20,
            21, 22, 23, 24, 25;

    Matrix e_b(5, 5);
    e_b << 26, 27, 28, 29, 30,
           31, 32, 33, 34, 35,
           36, 37, 38, 39, 40,
           41, 42, 43, 44, 45,
           46, 47, 48, 49, 50;

    Eigen::RowVectorXf e_broadcast(5);
    e_broadcast << 1, 2, 3, 4, 5;


    // ============================================================
    // Matrix vs Matrix
    // ============================================================

    auto np_c0 = np_a <  np_b;
    auto np_c1 = np_a >  np_b;
    auto np_c2 = np_a <= np_b;
    auto np_c3 = np_a >= np_b;
    auto np_c4 = np_a == np_b;
    auto np_c5 = np_a != np_b;

    auto e_c0 = (e_a.array() <  e_b.array());
    auto e_c1 = (e_a.array() >  e_b.array());
    auto e_c2 = (e_a.array() <= e_b.array());
    auto e_c3 = (e_a.array() >= e_b.array());
    auto e_c4 = (e_a.array() == e_b.array());
    auto e_c5 = (e_a.array() != e_b.array());

    assert(compare(np_c0, e_c0));
    assert(compare(np_c1, e_c1));
    assert(compare(np_c2, e_c2));
    assert(compare(np_c3, e_c3));
    assert(compare(np_c4, e_c4));
    assert(compare(np_c5, e_c5));


    // ============================================================
    // Matrix vs Scalar
    // ============================================================

    auto np_c6  = np_a <  25;
    auto np_c7  = np_a >  25;
    auto np_c8  = np_a <= 25;
    auto np_c9  = np_a >= 25;
    auto np_c10 = np_a == 35;
    auto np_c11 = np_a != 15;

    auto e_c6  = (e_a.array() <  25);
    auto e_c7  = (e_a.array() >  25);
    auto e_c8  = (e_a.array() <= 25);
    auto e_c9  = (e_a.array() >= 25);
    auto e_c10 = (e_a.array() == 35);
    auto e_c11 = (e_a.array() != 15);

    assert(compare(np_c6,  e_c6));
    assert(compare(np_c7,  e_c7));
    assert(compare(np_c8,  e_c8));
    assert(compare(np_c9,  e_c9));
    assert(compare(np_c10, e_c10));
    assert(compare(np_c11, e_c11));


    // ============================================================
    // Scalar vs Matrix
    // ============================================================

    auto np_c12 = 25 <  np_a;
    auto np_c13 = 25 >  np_a;
    auto np_c14 = 25 <= np_a;
    auto np_c15 = 25 >= np_a;
    auto np_c16 = 25 == np_a;
    auto np_c17 = 25 != np_a;

    auto e_c12 = (25 <  e_a.array());
    auto e_c13 = (25 >  e_a.array());
    auto e_c14 = (25 <= e_a.array());
    auto e_c15 = (25 >= e_a.array());
    auto e_c16 = (25 == e_a.array());
    auto e_c17 = (25 != e_a.array());

    assert(compare(np_c12, e_c12));
    assert(compare(np_c13, e_c13));
    assert(compare(np_c14, e_c14));
    assert(compare(np_c15, e_c15));
    assert(compare(np_c16, e_c16));
    assert(compare(np_c17, e_c17));


    // ============================================================
    // Matrix vs Row Broadcast
    // ============================================================

    auto np_c18 = np_a <  np_broadcast;
    auto np_c19 = np_a >  np_broadcast;
    auto np_c20 = np_a <= np_broadcast;
    auto np_c21 = np_a >= np_broadcast;
    auto np_c22 = np_a == np_broadcast;
    auto np_c23 = np_a != np_broadcast;

    auto e_c18 = (e_a.array() < e_broadcast.replicate(e_a.rows(), 1).array());
    auto e_c19 = (e_a.array() > e_broadcast.replicate(e_a.rows(), 1).array());
    auto e_c20 = (e_a.array() <= e_broadcast.replicate(e_a.rows(), 1).array());
    auto e_c21 = (e_a.array() >= e_broadcast.replicate(e_a.rows(), 1).array());
    auto e_c22 = (e_a.array() == e_broadcast.replicate(e_a.rows(), 1).array());
    auto e_c23 = (e_a.array() != e_broadcast.replicate(e_a.rows(), 1).array());

    assert(compare(np_c18, e_c18));
    assert(compare(np_c19, e_c19));
    assert(compare(np_c20, e_c20));
    assert(compare(np_c21, e_c21));
    assert(compare(np_c22, e_c22));
    assert(compare(np_c23, e_c23));


    // ============================================================
    // Row Broadcast vs Matrix
    // ============================================================

    auto np_c24 = np_broadcast <  np_a;
    auto np_c25 = np_broadcast >  np_a;
    auto np_c26 = np_broadcast <= np_a;
    auto np_c27 = np_broadcast >= np_a;
    auto np_c28 = np_broadcast == np_a;
    auto np_c29 = np_broadcast != np_a;

    auto e_c24 =
        (e_broadcast.replicate(e_a.rows(), 1).array()
            < e_a.array());

    auto e_c25 =
        (e_broadcast.replicate(e_a.rows(), 1).array()
            > e_a.array());

    auto e_c26 =
        (e_broadcast.replicate(e_a.rows(), 1).array()
            <= e_a.array());

    auto e_c27 =
        (e_broadcast.replicate(e_a.rows(), 1).array()
            >= e_a.array());

    auto e_c28 =
        (e_broadcast.replicate(e_a.rows(), 1).array()
            == e_a.array());

    auto e_c29 =
        (e_broadcast.replicate(e_a.rows(), 1).array()
            != e_a.array());

    assert(compare(np_c24, e_c24));
    assert(compare(np_c25, e_c25));
    assert(compare(np_c26, e_c26));
    assert(compare(np_c27, e_c27));
    assert(compare(np_c28, e_c28));
    assert(compare(np_c29, e_c29));


    // ============================================================
    // Logical NOT
    // ============================================================

    auto np_c30 = !np_c0;
    auto e_c30 = !e_c0;

    assert(compare(np_c30, e_c30));


    std::cout << "All NumPP/Eigen comparison tests passed!\n";
}