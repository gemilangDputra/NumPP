#ifndef NUMPP_MATRIX_VIEW_RESHAPE_HPP
#define NUMPP_MATRIX_VIEW_RESHAPE_HPP

#include <numpp/matrix/forward.hpp>
#include <numpp/matrix/core.hpp>

namespace numpp {
    template<typename T>
    template<strided_matrix EXPR>
    matrix_view<T>::matrix_view(EXPR& other, size_t newrow, size_t newcol) {
        if (newrow * newcol != other.size())
            throw std::invalid_argument("cannot reshape matrix: newrow * newcol must equal matrix.size()");
        if (!is_contiguous(other))
            throw std::invalid_argument("cannot reshape non-contiguous matrix");
        
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

    template<strided_matrix EXPR>
    auto reshape(EXPR& mat, size_t newrow, size_t newcol) {
        return matrix_view<typename EXPR::value_type>(mat, newrow, newcol);
    }

    template<strided_matrix EXPR>
    auto reshape(const EXPR& mat, size_t newrow, size_t newcol) {
        return matrix_view<const typename EXPR::value_type>(mat, newrow, newcol);
    }
}

#endif //NUMPP_MATRIX_VIEW_RESHAPE_HPP