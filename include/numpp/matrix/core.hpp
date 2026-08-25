#ifndef NUMPP_MATRIX_CORE_HPP
#define NUMPP_MATRIX_CORE_HPP

#include <numpp/matrix/forward.hpp>
#include <numpp/matrix/tool.hpp>
#include <cstdint>

namespace numpp {
    template<class Derived, typename T>
    class matrix_base {
        protected:
            T* data_;
            size_t size_;

            size_t col_;
            size_t row_;

            size_t rowstride_;
            size_t colstride_;

            layout order_;
            size_t offset_ = 0;
            
            void compute_stride(layout order) {
                order_ = order;

                if (order == layout::rowmajor) {
                    rowstride_ = col_;
                    colstride_ = 1;
                } else {
                    rowstride_ = 1;
                    colstride_ = row_;
                }
            }

            void init_metadata(
                T* data,
                size_t size,
                size_t row,
                size_t col,
                size_t rowstride,
                size_t colstride,
                layout order,
                size_t offset
            ) {
                data_ = data;
                size_ = size;

                row_ = row;
                col_ = col;

                rowstride_ = rowstride;
                colstride_ = colstride;

                order_ = order;
                offset_ = offset;
            }
        public:
            using value_type = std::remove_const_t<T>;
            
            size_t index_translate(size_t row, size_t col) const {
                return  offset_ + row * rowstride_ + col * colstride_;
            }
            
            const T* data() const { return data_; }
            T* data() { return data_; }

            size_t size() const { return size_; }
            size_t row() const { return row_; }
            size_t col() const { return col_; }
            size_t rowstride() const { return rowstride_; }
            size_t colstride() const { return colstride_; }

            layout order() const { return order_; }
            size_t offset() const { return offset_; }

            const T& operator()(size_t row, size_t col) const { return data_[index_translate(row, col)]; }
            T& operator()(size_t row, size_t col) { return data_[index_translate(row, col)]; }

            matrix_view<T> transpose();
            matrix_view<T> view();

            matrix_view<const T> transpose() const;
            matrix_view<const T> view() const;

            matrix_view<T> slice(slice_range row, slice_range col);
            matrix_view<T> slice(detail::all_t, slice_range col);
            matrix_view<T> slice(slice_range row, detail::all_t);
            matrix_view<T> slice(detail::all_t, detail::all_t);

            matrix_view<const T> slice(slice_range row, slice_range col) const;
            matrix_view<const T> slice(detail::all_t, slice_range col) const;
            matrix_view<const T> slice(slice_range row, detail::all_t) const;
            matrix_view<const T> slice(detail::all_t, detail::all_t) const;

            matrix_view<T> reshape(size_t newrow, size_t newcol);
            matrix_view<const T> reshape(size_t newrow, size_t newcol) const;

            template<matrix_like B>
            requires (std::same_as<value_type, typename B::value_type> && can_add_assign<T>)
            matrix_base<Derived, T>& operator+=(const B& b);
            
            template<matrix_like B>
            requires (std::same_as<value_type, typename B::value_type> && can_sub_assign<T>)
            matrix_base<Derived, T>& operator-=(const B& b);
            
            template<matrix_like B>
            requires (std::same_as<value_type, typename B::value_type> && can_mul_assign<T>)
            matrix_base<Derived, T>& operator*=(const B& b);
            
            template<matrix_like B>
            requires (std::same_as<value_type, typename B::value_type> && can_div_assign<T>)
            matrix_base<Derived, T>& operator/=(const B& b);
            
            matrix_base<Derived, T>& operator+=(const T& scalar) requires (can_add_assign<T>);
            matrix_base<Derived, T>& operator-=(const T& scalar) requires (can_sub_assign<T>);
            matrix_base<Derived, T>& operator*=(const T& scalar) requires (can_mul_assign<T>);
            matrix_base<Derived, T>& operator/=(const T& scalar) requires (can_div_assign<T>);

            matrix<T> operator-() const requires requires(T a) { -a; };
            matrix<T> operator+() const requires requires(T a) { +a; };

            bool all() requires(std::same_as<T, bool>) {
                if (is_contiguous(derived())) {
                    for (size_t i = 0; i < size_; ++i) {
                        if (!data_[i]) return false;
                    }
                    return true;
                } else {
                    for (size_t i = 0; i < row_; ++i) {
                        for (size_t j = 0; j < col_; ++j) {
                            if (!(*this)(i,j)) return false;
                        }
                    }
                    return true;
                }
            }

            bool any() requires(std::same_as<T, bool>) {
                if (is_contiguous(derived())) {
                    for (size_t i = 0; i < size_; ++i) {
                        if (data_[i]) return true;
                    }
                    return false;
                } else {
                    for (size_t i = 0; i < row_; ++i) {
                        for (size_t j = 0; j < col_; ++j) {
                            if ((*this)(i,j)) return true;
                        }
                    }
                    return false;
                }
            }

            matrix<T> sqrt() const;
            matrix<T> cbrt() const;

            matrix<T> cos() const;
            matrix<T> sin() const;
            matrix<T> tan() const;
            
            matrix<T> acos() const;
            matrix<T> asin() const;
            matrix<T> atan() const;

            matrix<T> cosh() const;
            matrix<T> sinh() const;
            matrix<T> tanh() const;

            matrix<T> log() const;
            matrix<T> log2() const;
            matrix<T> log10() const;

            matrix<T> exp() const;

            matrix<T> ceil() const;
            matrix<T> floor() const;
            matrix<T> round() const;
            matrix<T> trunc() const;
            matrix<T> abs() const;
            matrix<T> neg() const;

            T max() const;
            T min() const;
            size_t argmax() const;
            size_t argmin() const;

            T sum() const requires(can_add_assign<T>);
            T prod() const requires(can_mul_assign<T>);
            T mean() const requires(can_add_assign<T> && can_div<T>);
        private: 
            Derived& derived() {
                return static_cast<Derived&>(*this);
            }

            const Derived& derived() const {
                return static_cast<const Derived&>(*this);
            }
    };
}

#endif //NUMPP_MATRIX_CORE_HPP