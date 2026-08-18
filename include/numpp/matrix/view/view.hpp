#ifndef NUMPP_MATRIX_VIEW_HPP
#define NUMPP_MATRIX_VIEW_HPP

#include <numpp/matrix/forward.hpp>
#include <numpp/matrix/core.hpp>

#include <string>
#include <string_view>

namespace numpp {
    template<typename T>
    class matrix_view : public matrix_base<matrix_view<T>, T> {
        public:
            template<matrix_like EXPR>
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

            template<matrix_like EXPR>
            matrix_view& operator=(const EXPR& other) {
                if (this->row_ != other.row() || this->col_ != other.col()) {
                    throw std::invalid_argument(
                        "numpp::matrix_view::operator=: "
                        "shape mismatch: destination is " +
                        std::to_string(this->row_) + "x" +
                        std::to_string(this->col_) +
                        ", source is " +
                        std::to_string(other.row()) + "x" +
                        std::to_string(other.col())
                    );
                }

                for (size_t i = 0; i < this->row_; ++i)
                    for (size_t j = 0; j < this->col_; ++j)
                        (*this)(i, j) = other(i, j);

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

            template<matrix_like EXPR>
            matrix_view(EXPR& other, detail::transflag_t) ;
            template<matrix_like EXPR>
            matrix_view(EXPR& other, size_t newrow, size_t newcol);
            template<matrix_like EXPR>
            matrix_view(EXPR& other, slice_range row, slice_range col);
            template<matrix_like EXPR>
            matrix_view(EXPR& other, detail::all_t, slice_range col);
            template<matrix_like EXPR>
            matrix_view(EXPR& other, slice_range row, detail::all_t);
            template<matrix_like EXPR>
            matrix_view(EXPR& other, detail::all_t);
            template<matrix_like EXPR>
            matrix_view(EXPR& other, detail::all_t, detail::all_t);

            matrix_view(T* data, size_t row, size_t col, layout order) {
                this->init_metadata(
                    data,
                    row*col,
                    row,
                    col,
                    order == layout::rowmajor ? col : 1,
                    order == layout::rowmajor ? 1 : row,
                    order,
                    0
                );
            }

            template<vector_1d_like VEC>
            matrix_view(VEC& vec, size_t row, size_t col, layout order, std::string_view op) {
                if (vec.size() != row*col)
                    throw std::invalid_argument(
                        std::string("numpp::operation<") +
                        std::string(op) +
                        "(, ...)> error: "
                        "cannot reshape vec: requested shape is " +
                        std::to_string(row) + "x" + std::to_string(col) +
                        ", but vec contains " +
                        std::to_string(vec.size()) + " elements"
                    );
                this->init_metadata(
                    vec.data(),
                    vec.size(),
                    row,
                    col,
                    order == layout::rowmajor ? col : 1,
                    order == layout::rowmajor ? 1 : row,
                    order,
                    0
                );
            }

            matrix<T> to_matrix() const {
                matrix<T> out = matrix<T>::empty(this->row_, this->col_, this->order_);
                for (size_t i = 0; i < this->row_; ++i) {
                    for (size_t j = 0; j < this->col_; ++j) {
                        out(i, j) = (*this)(i, j);
                    }
                }
                return out;
            }
            
            template<matrix_like EXPR>
            void rebind(EXPR& other) {
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
            
    };

    template<class Derived, typename T>
    matrix_view<T> matrix_base<Derived, T>::view() {
        return matrix_view<T>(derived());
    }

    template<class Derived, typename T>
    matrix_view<const T> matrix_base<Derived, T>::view() const {
        return matrix_view<const T>(derived());
    }

    template<matrix_like EXPR>
    auto view(EXPR& other) {
        return matrix_view<typename EXPR::value_type>(other);
    }

    template<matrix_like EXPR>
    auto view(const EXPR& other) {
        return matrix_view<const typename EXPR::value_type>(other);
    }

    template<typename T>
    auto view(T* data, size_t row, size_t col, layout order=layout::rowmajor) {
        return matrix_view<T>(data, row, col, order);
    }
    
    template<typename T>
    auto view(const T* data, size_t row, size_t col, layout order=layout::rowmajor) {
        return matrix_view<const T>(data, row, col, order);
    }

    template<vector_1d_like VEC>
    auto view(VEC& vec, size_t row, size_t col, layout order=layout::rowmajor) {
        return matrix_view<typename VEC::value_type>(vec, row, col, order, "view");
    }
    
    template<vector_1d_like VEC>
    auto view(const VEC& vec, size_t row, size_t col, layout order=layout::rowmajor) {
        return matrix_view<const typename VEC::value_type>(vec, row, col, order, "view");
    }
}

#endif //NUMPP_MATRIX_VIEW_HPP