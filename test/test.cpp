#include <numpp/matrix.hpp>
#include <numpp/matrix/linalg/matmul.hpp>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>

template<numpp::numpp_matrix A, numpp::numpp_matrix B>
void compare(const A& a, const B& b) {
    bool equal = true;
    if (a.row() != static_cast<std::size_t>(b.row()) ||
        a.col() != static_cast<std::size_t>(b.col())) {
        std::cout
            << "[MISMATCH] shape: "
            << a.row() << "x" << a.col()
            << " vs "
            << b.row() << "x" << b.col()
            << '\n';
        return;
    }
    for (std::size_t i = 0; i < a.row(); ++i) {
        for (std::size_t j = 0; j < a.col(); ++j) {
            if (a(i, j) != b(i, j)) {
                equal = false;
                std::cout
                    << "  ("
                    << i << ", " << j << "): "
                    << "a=" << a(i, j)
                    << ", b=" << b(i, j)
                    << '\n';
            }
        }
    }
    if (equal)
        std::cout << "[MATCH]\n";
    else
        std::cout << "[MISMATCH]\n";
}


template<numpp::numpp_matrix A, typename Derived>
void compare(const A& a, const Eigen::MatrixBase<Derived>& b) {
    bool equal = true;
    if (a.row() != static_cast<std::size_t>(b.rows()) ||
        a.col() != static_cast<std::size_t>(b.cols())) {
        std::cout
            << "[MISMATCH] shape: "
            << a.row() << "x" << a.col()
            << " vs "
            << b.rows() << "x" << b.cols()
            << '\n';
        return;
    }
    for (std::size_t i = 0; i < a.row(); ++i) {
        for (std::size_t j = 0; j < a.col(); ++j) {
            if (a(i, j) != b(i, j)) {
                equal = false;
                std::cout
                    << "  ("
                    << i << ", " << j << "): "
                    << "eigen=" << a(i, j)
                    << ", numpp=" << b(i, j)
                    << '\n';
            }
        }
    }
    if (equal)
        std::cout << "[MATCH]\n";
    else
        std::cout << "[MISMATCH]\n";
}

template<typename T>
class MyVector {
    private:
        std::vector<T> data_;
    public:
        using value_type = T;
        MyVector(size_t size) : data_(size) {}
        size_t size() const { return data_.size(); }
        const T* data() const { return data_.data(); }
        T* data() { return data_.data(); }
};

int main() {
    std::vector<float> vector10x10(10*10);
    std::vector<float> vector15x15(15*15);
    MyVector<float> myvector10x10(10*10);
    MyVector<float> myvector15x15(15*15);
    for (size_t i=1; i < 10*10; ++i) { vector10x10[i] = static_cast<float>(i); myvector10x10.data()[i] = static_cast<float>(i); }
    for (size_t i=1; i < 15*15; ++i) { vector15x15[i] = static_cast<float>(i); myvector15x15.data()[i] = static_cast<float>(i); }
    Eigen::MatrixXf eigen_matrix10x10 =
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(vector10x10.data(), 10, 10);
    Eigen::MatrixXf eigen_matrix15x15 =
    Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(vector15x15.data(), 15, 15);
    numpp::matrix<float> numpp_matrix10x10 = numpp::reshape(vector10x10, 10,10).to_matrix();
    numpp::matrix<float> numpp_matrix15x15 = numpp::reshape(vector15x15, 15,15).to_matrix();
    numpp::matrix<float> numpp_matrix10x10_an1 = numpp::reshape(vector10x10.data(), 10,10).to_matrix();
    numpp::matrix<float> numpp_matrix15x15_an2 = numpp::reshape(vector15x15.data(), 15,15).to_matrix();
    numpp::matrix<float> numpp_matrix10x10_an3 = numpp::reshape(myvector10x10, 10,10).to_matrix();
    numpp::matrix<float> numpp_matrix15x15_an4 = numpp::reshape(myvector15x15, 15,15).to_matrix();
    compare(numpp_matrix10x10, numpp_matrix10x10_an1);
    compare(numpp_matrix15x15, numpp_matrix15x15_an2);
    compare(numpp_matrix10x10, numpp_matrix10x10_an3);
    compare(numpp_matrix15x15, numpp_matrix15x15_an4);
    numpp::matrix_view numpp_slice10x10 = numpp_matrix15x15.slice(numpp::slice_range(0,10,1), numpp::slice_range(5,15));
    auto eigen_slice10x10 = eigen_matrix15x15( Eigen::seq(0, 9), Eigen::seq(5, 14));
    auto numpp_c_1 = numpp::linalg::matmul(numpp_matrix10x10, numpp_matrix10x10); //kalau work(M*N*K) diatas 5250000, biasanya aktif blasnya
    auto numpp_c_2 = numpp::linalg::matmul(numpp_matrix10x10, numpp_matrix10x10.transpose()); //aktif juga biasanya blas
    auto numpp_c_3 = numpp::linalg::matmul(numpp_matrix10x10, numpp_slice10x10);
    auto eigen_c_1 = eigen_matrix10x10 * eigen_matrix10x10;
    auto eigen_c_2 = eigen_matrix10x10 * eigen_matrix10x10.transpose();
    auto eigen_c_3 = eigen_matrix10x10 * eigen_slice10x10;
    std::cout << "test matrix multiplication\n";
    compare(numpp_c_1, eigen_c_1);
    compare(numpp_c_2, eigen_c_2);
    compare(numpp_c_3, eigen_c_3);
}