#ifndef NUMPP_MATRIX_CONSTRUCTOR_HPP
#define NUMPP_MATRIX_CONSTRUCTOR_HPP

#include <numpp/matrix/tool.hpp>
#include <numpp/matrix/matrix.hpp>
#include <numpp/matrix/core.hpp>

#include <algorithm>
#include <stdexcept>

namespace numpp {
    template<typename T>
    matrix<T> matrix<T>::empty(size_t row, size_t col, layout order) {
        matrix<T> out;
        out.row_ = row;
        out.col_ = col;

        out.compute_stride(order);
        out.size_ = row*col;
        out.alloc();
        return out;
    }

    template<typename T>
    matrix<T> matrix<T>::zeros(size_t row, size_t col, layout order) {
        matrix<T> out;
        out.row_ = row;
        out.col_ = col;

        out.compute_stride(order);
        out.size_ = row*col;
        out.alloc(true);
        return out;
    }
    
    template<typename T>
    matrix<T> matrix<T>::ones(size_t row, size_t col, layout order) {
        matrix<T> out;
        out.row_ = row;
        out.col_ = col;

        out.compute_stride(order);
        out.size_ = row*col;
        out.alloc();
        for (size_t i=0; i < out.size(); ++i) out.data()[i] = T{1};
        return out;
    }
    
    template<typename T>
    matrix<T> matrix<T>::full(size_t row, size_t col, const T& value, layout order) {
        matrix<T> out;
        out.row_ = row;
        out.col_ = col;

        out.compute_stride(order);
        out.size_ = row*col;
        out.alloc();
        for (size_t i=0; i < out.size(); ++i) out.data()[i] = value;
        return out;
    }

    template<typename T>
    template<strided_matrix EXPR>
    matrix<T> matrix<T>::empty_like(const EXPR& other) {
        matrix<T> out;
        out.row_ = other.row();
        out.col_ = other.col();

        out.rowstride_ = other.rowstride();
        out.colstride_ = other.colstride();

        out.size_ = other.size();

        out.order_ = other.order();
        out.offset_ = other.offset();

        out.alloc();

        return out;
    }

    template<typename T>
    template<strided_matrix EXPR>
    matrix<T> matrix<T>::zeros_like(const EXPR& other) {
        matrix<T> out;
        out.row_ = other.row();
        out.col_ = other.col();

        out.rowstride_ = other.rowstride();
        out.colstride_ = other.colstride();

        out.size_ = other.size();

        out.order_ = other.order();
        out.offset_ = other.offset();

        out.alloc(true);
    }

    template<typename T>
    template<strided_matrix EXPR>
    matrix<T> matrix<T>::ones_like(const EXPR& other) {
        matrix<T> out;
        out.row_ = other.row();
        out.col_ = other.col();

        out.rowstride_ = other.rowstride();
        out.colstride_ = other.colstride();

        out.size_ = other.size();

        out.order_ = other.order();
        out.offset_ = other.offset();

        out.alloc();

        for (size_t i=0; i < other.size(); ++i) out->data_[i] = T{1};
        return out;
    }

    template<typename T>
    template<strided_matrix EXPR>
    matrix<T> matrix<T>::full_like(const EXPR& other, const T& value) {
        matrix<T> out;
        out.row_ = other.row();
        out.col_ = other.col();

        out.rowstride_ = other.rowstride();
        out.colstride_ = other.colstride();

        out.size_ = other.size();

        out.order_ = other.order();
        out.offset_ = other.offset();

        out.alloc();

        for (size_t i=0; i < other.size(); ++i) out->data_[i] = value;
        return out;
    }

    template<typename T>
    matrix<T> matrix<T>::eye(size_t row, size_t col, int k, layout order) {
        matrix<T> out;
        out.row_ = row;
        out.col_ = col;

        out.compute_stride(order);
        out.size_ = row*col;
        out.alloc(true);

        const int begin = std::max(0, -k);
        const int end  = std::min<int>(row, col - k);

        for (int i = begin; i < end; ++i) out(i, i + k) = T{1};
        return out;
    }

    template<typename T>
    matrix<T> matrix<T>::diag(std::initializer_list<T> list, int k, layout order) {
        size_t n = list.size();
        size_t row = n + std::max(0, -k);
        size_t col = n + std::max(0,  k);

        matrix<T> out;
        out.row_ = row;
        out.col_ = col;

        out.compute_stride(order);
        out.size_ = row * col;
        out.alloc(true);

        size_t i = 0;
        const int row_offset = (k < 0) ? -k : 0;
        const int col_offset = (k > 0) ?  k : 0;

        for (const auto& value : list) {
            out(i + row_offset, i + col_offset) = value;
            ++i;
        }

        return out;
    }

    template<typename T>
    matrix<T>::matrix(std::initializer_list<std::initializer_list<T>> list, layout order) {
        this->row_ = list.size();
        this->col_ = list.begin()->size();

        for (const auto& row : list) {
            if (row.size() != this->col_)
                throw std::invalid_argument("inconsistent row sizes");
        }

        this->compute_stride(order);
        this->size_ = this->row_ * this->col_;
        alloc();

        size_t idx = 0;
        for (const auto& row : list) {
            for (const auto& value : row) {
                this->data_[idx++] = value;
            }
        }
    }

    template<typename T>
    template<strided_matrix EXPR>
    matrix<T>::matrix(const EXPR& other) {
        this->row_ = other.row();
        this->col_ = other.col();
        this->compute_stride(other.order());

        this->size_ = this->row_ * this->col_;
        this->offset_ = 0;

        alloc();
        for (size_t i = 0; i < this->row_; ++i)
            for (size_t j = 0; j < this->col_; ++j)
                (*this)(i, j) = other(i, j);
    }

    template<typename T>
    template<strided_matrix EXPR>
    matrix<T>& matrix<T>::operator=(const EXPR& other) {
        if (*this != &other) {
            del();
            this->row_ = other.row();
            this->col_ = other.col();
            this->compute_stride(other.order());

            this->size_ = this->row_ * this->col_;
            this->offset_ = 0;

            alloc();
            for (size_t i = 0; i < this->row_; ++i)
                for (size_t j = 0; j < this->col_; ++j)
                    (*this)(i, j) = other(i, j);
        }
        return *this;
    }

    template<typename T>
    matrix<T>::matrix(const matrix<T>& other) {
        this->row_ = other.row();
        this->col_ = other.col();
        this->compute_stride(other.order());

        this->size_ = this->row_ * this->col_;
        this->offset_ = 0;

        alloc();
        for (size_t i = 0; i < this->row_; ++i)
            for (size_t j = 0; j < this->col_; ++j)
                (*this)(i, j) = other(i, j);
    }

    
    template<typename T>
    matrix<T>& matrix<T>::operator=(const matrix<T>& other) {
        if (*this != &other) {
            del();
            this->row_ = other.row();
            this->col_ = other.col();
            this->compute_stride(other.order());

            this->size_ = this->row_ * this->col_;
            this->offset_ = 0;

            alloc();
            for (size_t i = 0; i < this->row_; ++i)
                for (size_t j = 0; j < this->col_; ++j)
                    (*this)(i, j) = other(i, j);
        }
        return *this;
    }

    template<typename T>
    matrix<T>::matrix(matrix<T>&& other) noexcept {
        this->row_ = other.row_;
        this->col_ = other.col_;
        this->rowstride_ = other.rowstride_;
        this->colstride_ = other.colstride_;
        this->size_ = other.size_;
        this->order_ = other.order_;
        this->offset_ = other.offset_;
        this->data_ = other.data_;
        other.data_ = nullptr;
    }

    template<typename T>
    matrix<T>& matrix<T>::operator=(matrix<T>&& other) noexcept {
        if (this != &other) {
            this->row_ = other.row_;
            this->col_ = other.col_;
            this->rowstride_ = other.rowstride_;
            this->colstride_ = other.colstride_;
            this->size_ = other.size_;
            this->order_ = other.order_;
            this->offset_ = other.offset_;
            this->data_ = other.data_;
            other.data_ = nullptr;
        }
        return *this;
    }
}

#endif //NUMPP_MATRIX_CONSTRUCTOR_HPP