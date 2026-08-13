#ifndef NUMPP_MATRIX_VIEW_RESHAPE_HPP
#define NUMPP_MATRIX_VIEW_RESHAPE_HPP

#include <numpp/matrix/forward.hpp>
#include <numpp/matrix/core.hpp>

#include <string>

namespace numpp {
    template<typename T>
    template<matrix_like EXPR>
    matrix_view<T>::matrix_view(EXPR& other, size_t newrow, size_t newcol) {
        if (newrow * newcol != other.size())
            throw std::invalid_argument(
                "numpp::operation<reshape> error: "
                "cannot reshape matrix with " +
                std::to_string(other.size()) +
                " elements into " +
                std::to_string(newrow) + "x" +
                std::to_string(newcol)
            );

        if (!is_contiguous(other))
            throw std::invalid_argument(
                "numpp::operation<reshape> error: "
                "cannot reshape non-contiguous matrix"
            );
        
        this->init_metadata(
            other.data(),
            other.size(),
            newrow,
            newcol,
            (other.order() == layout::rowmajor) ? newcol : 1,
            (other.order() == layout::rowmajor) ? 1 : newrow,
            other.order(),
            other.offset()
        );
    }

    template<class Derived, typename T>
    matrix_view<T> matrix_base<Derived, T>::reshape(size_t newrow, size_t newcol) {
        return matrix_view<T>(derived(), newrow, newcol);
    }
    
    template<class Derived, typename T>
    matrix_view<const T> matrix_base<Derived, T>::reshape(size_t newrow, size_t newcol) const {
        return matrix_view<const T>(derived(), newrow, newcol);
    }

    template<matrix_like EXPR>
    auto reshape(EXPR& mat, size_t newrow, size_t newcol) {
        return matrix_view<typename EXPR::value_type>(mat, newrow, newcol);
    }

    template<matrix_like EXPR>
    auto reshape(const EXPR& mat, size_t newrow, size_t newcol) {
        return matrix_view<const typename EXPR::value_type>(mat, newrow, newcol);
    }

    template<typename T>
    auto reshape(T* data, size_t row, size_t col, layout order=layout::rowmajor) {
        return matrix_view<T>(data, row, col, order);
    }
    
    template<typename T>
    auto reshape(const T* data, size_t row, size_t col, layout order=layout::rowmajor) {
        return matrix_view<const T>(data, row, col, order);
    }

    template<vector_1d_like VEC>
    auto reshape(VEC& vec, size_t row, size_t col, layout order=layout::rowmajor) {
        return matrix_view<typename VEC::value_type>(vec, row, col, order, "view");
    }
    
    template<vector_1d_like VEC>
    auto reshape(const VEC& vec, size_t row, size_t col, layout order=layout::rowmajor) {
        return matrix_view<const typename VEC::value_type>(vec, row, col, order, "view");
    }
}

#endif //NUMPP_MATRIX_VIEW_RESHAPE_HPP