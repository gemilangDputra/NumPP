#ifndef NUMPP_MATRIX_VIEW_TRANSPOSE_HPP
#define NUMPP_MATRIX_VIEW_TRANSPOSE_HPP

#include <numpp/matrix/forward.hpp>
#include <numpp/matrix/core.hpp>

namespace numpp {
    template<typename T>
    template<strided_matrix EXPR>
    matrix_view<T>::matrix_view(EXPR& other, detail::transflag_t) {
        this->init_metadata(
            other.data(),
            other.size(),
            other.col(),
            other.row(),
            other.colstride(),
            other.rowstride(),
            other.order(),
            other.offset()
        );
    }

    template<class Derived, typename T>
    matrix_view<T> matrix_base<Derived, T>::transpose() {
        return matrix_view<T>(derived(), detail::transflag);
    }
    
    template<class Derived, typename T>
    matrix_view<const T> matrix_base<Derived, T>::transpose() const {
        return matrix_view<const T>(derived(), detail::transflag);
    }

    template<strided_matrix EXPR>
    auto transpose(EXPR& mat)  {
        return matrix_view<typename EXPR::value_type>(mat, detail::transflag);
    }

    template<strided_matrix EXPR>
    auto transpose(const EXPR& mat)  {
        return matrix_view<const typename EXPR::value_type>(mat, detail::transflag);
    }
}

#endif //NUMPP_MATRIX_VIEW_TRANSPOSE_HPP