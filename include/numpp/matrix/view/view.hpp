#ifndef NUMPP_MATRIX_VIEW_HPP
#define NUMPP_MATRIX_VIEW_HPP

#include <numpp/matrix/forward.hpp>
#include <numpp/matrix/core.hpp>

namespace numpp {
    template<typename T>
    class matrix_view : public matrix_base<matrix_view<T>, T> {
        public:
            template<strided_matrix EXPR>
            matrix_view(EXPR& other) {
                this->init_metadata(
                    other.data(),
                    other.size(),
                    other.row(),
                    other.col(),
                    other.rowstride(),
                    other.colstride(),
                    other.order(),
                    other.offset()
                );
            }

            template<strided_matrix EXPR>
            matrix_view(EXPR& other, detail::transflag_t) ;

            template<strided_matrix EXPR>
            matrix_view(EXPR& other, size_t newrow, size_t newcol);

            template<strided_matrix EXPR>
            matrix_view(EXPR& other, slice_range row, slice_range col);
            template<strided_matrix EXPR>
            matrix_view(EXPR& other, detail::all_t, slice_range col);
            template<strided_matrix EXPR>
            matrix_view(EXPR& mat, slice_range row, detail::all_t);
            template<strided_matrix EXPR>
            matrix_view(EXPR& mat, detail::all_t);

            matrix<T> to_matrix() const {
                matrix<T> out = matrix<T>::empty(this->row_, this->col_, this->order_);
                for (size_t i = 0; i < this->row_; ++i) {
                    for (size_t j = 0; j < this->col_; ++j) {
                        out(i, j) = (*this)(i, j);
                    }
                }
                return out;
            }
    };

    template<class Derived, typename T>
    matrix_view<T> matrix_base<Derived, T>::view() {
        return matrix_view<T>(derived());
    }

    template<class Derived, typename T>
    matrix_view<const T> matrix_base<Derived, T>::view() const {
        return matrix_view<const T>(derived());
    }

    template<strided_matrix EXPR>
    auto view(EXPR& mat) {
        return matrix_view<typename EXPR::value_type>(mat);
    }

    template<strided_matrix EXPR>
    auto view(const EXPR& mat) {
        return matrix_view<const typename EXPR::value_type>(mat);
    }
}

#endif //NUMPP_MATRIX_VIEW_HPP