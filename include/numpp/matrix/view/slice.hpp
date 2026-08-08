#ifndef NUMPP_MATRIX_VIEW_SLICE_HPP
#define NUMPP_MATRIX_VIEW_SLICE_HPP

#include <numpp/matrix/forward.hpp>
#include <numpp/matrix/core.hpp>

namespace numpp {
    struct slice_range {
        size_t start;
        size_t stop;
        size_t step=1;

        slice_range(size_t start_, size_t stop_, size_t step_=1) : start(start_), stop(stop_), step(step_) {}
    };

    template<typename T>
    template<strided_matrix EXPR>
    matrix_view<T>::matrix_view(EXPR& other, slice_range row, slice_range col) {
        if (row.step == 0)
            throw std::invalid_argument("row slice step must be greater than zero");

        if (col.step == 0)
            throw std::invalid_argument("column slice step must be greater than zero");

        if (row.start > row.stop)
            throw std::out_of_range("row slice start cannot be greater than stop");

        if (col.start > col.stop)
            throw std::out_of_range("column slice start cannot be greater than stop");

        if (row.stop > other.row())
            throw std::out_of_range("row slice stop exceeds matrix row count");

        if (col.stop > other.col())
            throw std::out_of_range("column slice stop exceeds matrix column count");

        size_t rows = (row.stop - row.start + row.step - 1) / row.step;
        size_t cols = (col.stop - col.start + col.step - 1) / col.step;

        this->init_metadata(
            other.data(),
            rows * cols,
            rows,
            cols,
            other.rowstride() * row.step,
            other.colstride() * col.step,
            other.order(),
            other.offset()
                + row.start * other.rowstride()
                + col.start * other.colstride()
        );
    }

    template<typename T>
    template<strided_matrix EXPR>
    matrix_view<T>::matrix_view(EXPR& other, detail::all_t, slice_range col) {
        if (col.step == 0)
            throw std::invalid_argument("column slice step must be greater than zero");

        if (col.start > col.stop)
            throw std::out_of_range("column slice start cannot be greater than stop");

        if (col.stop > other.col())
            throw std::out_of_range("column slice stop exceeds matrix column count");

        size_t cols = ((col.stop - col.start) + col.step - 1) / col.step;

        this->copy_metadata(
            other.data(),
            other.row() * cols,
            other.row(),
            cols,
            other.rowstride(),
            other.colstride() * col.step,
            other.order(),
            other.offset()
                + col.start * other.colstride()
        );
    }

    template<typename T>
    template<strided_matrix EXPR>
    matrix_view<T>::matrix_view(EXPR& mat, slice_range row, detail::all_t) {
        if (row.step == 0)
            throw std::invalid_argument("row slice step must be greater than zero");

        if (row.start > row.stop)
            throw std::out_of_range("row slice start cannot be greater than stop");

        if (row.stop > mat.row())
            throw std::out_of_range("row slice stop exceeds matrix row count");

        size_t rows = ((row.stop - row.start) + row.step - 1) / row.step;

        this->init_metadata(
            mat.data(),
            rows * mat.col(),
            rows,
            mat.col(),
            mat.rowstride() * row.step,
            mat.colstride(),
            mat.order(),
            mat.offset()
            + row.start * mat.rowstride()
        );
    }

    template<typename T>
    template<strided_matrix EXPR>
    matrix_view<T>::matrix_view(EXPR& mat, detail::all_t) {
        this->copy_metadata(
            mat.data(),
            mat.size(),
            mat.row(),
            mat.col(),
            mat.rowstride(),
            mat.colstride(),
            mat.order(),
            mat.offset()
        );
    }
    
    template<class Derived, typename T>
    matrix_view<T> matrix_base<Derived, T>::slice(slice_range row, slice_range col) {
        return matrix_view<T>(derived(), row, col);
    }

    template<class Derived, typename T>
    matrix_view<T> matrix_base<Derived, T>::slice(detail::all_t, slice_range col) {
        return matrix_view<T>(derived(), all, col);
    }

    template<class Derived, typename T>
    matrix_view<T> matrix_base<Derived, T>::slice(slice_range row, detail::all_t) {
        return matrix_view<T>(derived(), row, all);
    }

    template<class Derived, typename T>
    matrix_view<T> matrix_base<Derived, T>::slice(detail::all_t, detail::all_t) {
        return matrix_view<T>(derived(), all, all);
    }

    template<class Derived, typename T>
    matrix_view<const T> matrix_base<Derived, T>::slice(slice_range row, slice_range col) const {
        return matrix_view<const T>(derived(), row, col);
    }

    template<class Derived, typename T>
    matrix_view<const T> matrix_base<Derived, T>::slice(detail::all_t, slice_range col) const {
        return matrix_view<const T>(derived(), all, col);
    }

    template<class Derived, typename T>
    matrix_view<const T> matrix_base<Derived, T>::slice(slice_range row, detail::all_t) const {
        return matrix_view<const T>(derived(), row, all);
    }

    template<class Derived, typename T>
    matrix_view<const T> matrix_base<Derived, T>::slice(detail::all_t, detail::all_t) const {
        return matrix_view<const T>(derived(), all, all);
    }
    
    template<strided_matrix EXPR>
    auto slice(EXPR& mat, slice_range row, slice_range col) {
        return matrix_view<typename EXPR::value_type>(mat, row, col);
    }

    template<strided_matrix EXPR>
    auto slice(EXPR& mat, detail::all_t, slice_range col) {
        return matrix_view<typename EXPR::value_type>(mat, all, col);
    }

    template<strided_matrix EXPR>
    auto slice(EXPR& mat, slice_range row, detail::all_t) {
        return matrix_view<typename EXPR::value_type>(mat, row, all);
    }

    template<strided_matrix EXPR>
    auto slice(EXPR& mat, detail::all_t, detail::all_t) {
        return matrix_view<typename EXPR::value_type>(mat, all, all);
    }
    
    template<strided_matrix EXPR>
    auto slice(const EXPR& mat, slice_range row, slice_range col) {
        return matrix_view<const typename EXPR::value_type>(mat, row, col);
    }

    template<strided_matrix EXPR>
    auto slice(const EXPR& mat, detail::all_t, slice_range col) {
        return matrix_view<const typename EXPR::value_type>(mat, all, col);
    }
    
    template<strided_matrix EXPR>
    auto slice(const EXPR& mat, slice_range row, detail::all_t) {
        return matrix_view<const typename EXPR::value_type>(mat, row, all);
    }
    
    template<strided_matrix EXPR>
    auto slice(const EXPR& mat, detail::all_t, detail::all_t) {
        return matrix_view<const typename EXPR::value_type>(mat, all, all);
    }
}

#endif //NUMPP_MATRIX_VIEW_SLICE_HPP