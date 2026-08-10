#ifndef NUMPP_MATRIX_VIEW_HPP
#define NUMPP_MATRIX_VIEW_HPP

#include <numpp/matrix/forward.hpp>
#include <numpp/matrix/core.hpp>

#include <string>

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

            matrix_view(const matrix_view& other) {
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
            matrix_view& operator=(EXPR& other) {
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

                return *this;
            }

            matrix_view& operator=(const matrix_view& other) {
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

                return *this;
            }

            matrix_view(T* data,
                size_t size,
                size_t row,
                size_t col,
                size_t rowstride,
                size_t colstride,
                layout order,
                size_t offset) {
                this->init_metadata(
                    data,
                    size,
                    row,
                    col,
                    rowstride,
                    colstride,
                    order,
                    offset
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
            matrix_view(EXPR& other, slice_range row, detail::all_t);
            template<strided_matrix EXPR>
            matrix_view(EXPR& other, detail::all_t);
            template<strided_matrix EXPR>
            matrix_view(EXPR& other, detail::all_t, detail::all_t);

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
    auto view(EXPR& other) {
        return matrix_view<typename EXPR::value_type>(other);
    }

    template<strided_matrix EXPR>
    auto view(const EXPR& other) {
        return matrix_view<const typename EXPR::value_type>(other);
    }

    template<typename T>
    auto view(T* data, size_t row, size_t col, layout order) {
        const size_t rowstride = order == layout::rowmajor ? col : 1;
        const size_t colstride = order == layout::rowmajor ? 1 : row;
        return matrix_view<T>(data, row*col, row, col, rowstride, colstride, order, 0);
    }
    
    template<typename T>
    auto view(const T* data, size_t row, size_t col, layout order) {
        const size_t rowstride = order == layout::rowmajor ? col : 1;
        const size_t colstride = order == layout::rowmajor ? 1 : row;
        return matrix_view<const T>(data, row*col, row, col, rowstride, colstride, order, 0);
    }

    template<vector_1d EXPR>
    auto view(EXPR& vector, size_t row, size_t col, layout order) {
        if (vector.size() != row*col)
            throw std::invalid_argument(
                "numpp::operation<view(vector, ...)> error: "
                "cannot reshape vector: requested shape is " +
                std::to_string(row) + "x" + std::to_string(col) +
                ", but vector contains " +
                std::to_string(vector.size()) + " elements"
            );
        
        const size_t rowstride = order == layout::rowmajor ? col : 1;
        const size_t colstride = order == layout::rowmajor ? 1 : row;
        return matrix_view<typename EXPR::value_type>(vector.data(),vector.size(), row, col, rowstride, colstride, order, 0);
    }
}

#endif //NUMPP_MATRIX_VIEW_HPP