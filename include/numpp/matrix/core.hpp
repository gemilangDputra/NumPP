#ifndef NUMPP_MATRIX_CORE_HPP
#define NUMPP_MATRIX_CORE_HPP

#include <initializer_list>
#include <stdexcept>

#include <numpp/matrix/forward.hpp>
#include <numpp/matrix/tool.hpp>

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

            size_t index_translate(size_t row, size_t col) const {
                return  offset_ + row * rowstride_ + col * colstride_;
            }
        public:
            using value_type = std::remove_const_t<T>;
            using pointer_type = T*;
            
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

            template<strided_matrix B>
            requires (std::same_as<T, typename B::value_type> && can_add_assign<T>)
            matrix_base<Derived, T>& operator+=(const B& b);
            
            template<strided_matrix B>
            requires (std::same_as<T, typename B::value_type> && can_sub_assign<T>)
            matrix_base<Derived, T>& operator-=(const B& b);
            
            template<strided_matrix B>
            requires (std::same_as<T, typename B::value_type> && can_mul_assign<T>)
            matrix_base<Derived, T>& operator*=(const B& b);
            
            template<strided_matrix B>
            requires (std::same_as<T, typename B::value_type> && can_div_assign<T>)
            matrix_base<Derived, T>& operator/=(const B& b);
            
            matrix_base<Derived, T>& operator+=(const T& scalar) requires (can_add_assign<T>);
            matrix_base<Derived, T>& operator-=(const T& scalar) requires (can_sub_assign<T>);
            matrix_base<Derived, T>& operator*=(const T& scalar) requires (can_mul_assign<T>);
            matrix_base<Derived, T>& operator/=(const T& scalar) requires (can_div_assign<T>);
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