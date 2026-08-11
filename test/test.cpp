#include <numpp/matrix.hpp>
#include <numpp/matrix/linalg/matmul.hpp>
#include <iostream>

template<typename T>
class Mymatrix {
    private:
        numpp::matrix<T> data_;
    public:
        using value_type = T;

        Mymatrix(const numpp::matrix<T>& matrix)
            : data_(matrix) {}

        std::size_t row() const { return data_.row(); }
        std::size_t col() const { return data_.col(); }
        std::size_t size() const { return data_.size(); }
        std::size_t rowstride() const { return data_.rowstride(); }
        std::size_t colstride() const { return data_.colstride(); }
        numpp::layout order() const { return data_.order(); }
        std::size_t offset() const { return data_.offset(); }
        T* data() { return data_.data(); }
        const T* data() const { return data_.data(); }
        T& operator()(std::size_t i, std::size_t j) { return data_(i, j); }
        const T& operator()(std::size_t i, std::size_t j) const { return data_(i, j); }
};

int main() {
    numpp::matrix matrix({
        { 1, 2, 3, 4, 5},
        { 6, 7, 8, 9,10},
        {11,12,13,14,15},
        {16,17,18,19,20},
        {21,22,23,24,25}
    });
    Mymatrix custom1(matrix);

    numpp::matrix_view view = numpp::view(custom1);
    std::cout << view << '\n';

    auto c =  custom1 + matrix - 5 * matrix;
    std::cout << c << '\n';
}