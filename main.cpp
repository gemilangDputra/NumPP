#include <numpp/matrix.hpp>
#include <eigen3/Eigen/Dense>

template<typename Derived>
class EigenAdaptor {
    private:
        Derived& derived_;
    public:
        using value_type = typename Derived::Scalar;
        explicit EigenAdaptor(Derived& derived) : derived_(derived) {}

        const value_type* data() const { return derived_.data(); }
        value_type* data() { return derived_.data(); }

        size_t size() const { return static_cast<size_t>(derived_.size()); }

        size_t row() const { return static_cast<size_t>(derived_.rows()); }
        size_t col() const { return static_cast<size_t>(derived_.cols()); }

        size_t rowstride() const {
            if constexpr (Derived::IsRowMajor) {
                return static_cast<size_t>(derived_.outerStride());
            } else {
                return static_cast<size_t>(derived_.innerStride());
            }
        }

        size_t colstride() const {
            if constexpr (Derived::IsRowMajor) {
                return static_cast<size_t>(derived_.innerStride());
            } else {
                return static_cast<size_t>(derived_.outerStride());
            }
        }

        numpp::layout order() const {
            if constexpr (Derived::IsRowMajor) {
                return numpp::layout::rowmajor;
            } else {
                return numpp::layout::colmajor;
            }
        }

        size_t offset() const { return 0; }

        value_type& operator()(size_t row, size_t col) { return derived_(row, col); }
        const value_type& operator()(size_t row, size_t col) const { return derived_(row, col); }
};

int main() {
    numpp::matrixf matrix({
        { 1, 2, 3, 4, 5},
        { 6, 7, 8, 9,10},
        {11,12,13,14,15},
        {16,17,18,19,20},
        {21,22,23,24,25}
    }, numpp::layout::colmajor);

    Eigen::MatrixXf eigen_matrix(5,5);
    eigen_matrix << 
         1, 2, 3, 4, 5,
         6, 7, 8, 9,10,
        11,12,13,14,15,
        16,17,18,19,20,
        21,22,23,24,25;
    
    EigenAdaptor eigen_matrix_adpt(eigen_matrix);
    auto view = numpp::view(eigen_matrix_adpt); //EigenAdaptor enggak punya assignment jadi di transform dulu
    view *= matrix;

    std::cout << eigen_matrix << '\n';
}