#ifndef NUMPP_MATRIX_SCALAR_ARITHMETIC_HPP
#define NUMPP_MATRIX_SCALAR_ARITHMETIC_HPP

#include <numpp/matrix/core.hpp>
#include <numpp/matrix/tool.hpp>
#include <numpp/matrix/matrix.hpp>
#include <numpp/matrix/view/broadcast.hpp>

namespace numpp {
    namespace detail {
        template<numpp_matrix EXPR, typename Op>
        matrix<typename EXPR::value_type> matrix_op_scalar_expr(
            const EXPR& a, const typename EXPR::value_type& scalar, Op op) {
            using T = typename EXPR::value_type;
            matrix<T> out = matrix<T>::empty_like(a);
            if (is_contiguous(a)) {
                const auto* adata = a.data() + a.offset();
                auto* cdata = out.data();
                for (size_t i = 0; i < out.size(); ++i)
                    cdata[i] = op(adata[i], scalar);
            }
            else {
                for (size_t i = 0; i < a.row(); ++i) {
                    for (size_t j = 0; j < a.col(); ++j)
                        out(i, j) = op(a(i, j), scalar);
                }
            }
            return out;
        }

        template<numpp_matrix EXPR, typename Op>
        EXPR& matrix_op_assign_scalar_expr( EXPR& mat, const typename EXPR::value_type& scalar, Op op) {
            if (is_contiguous(mat)) {
                auto* adata = mat.data() + mat.offset();
                for (size_t i = 0; i < mat.size(); ++i) {
                    op(adata[i], scalar);
                }
            }
            else {
                for (size_t i = 0; i < mat.row(); ++i) {
                    for (size_t j = 0; j < mat.col(); ++j) {
                        op(mat(i, j), scalar);
                    }
                }
            }
            return mat;
        }
    }
    template<numpp_matrix EXPR> requires (can_add<typename EXPR::value_type>)
    matrix<typename EXPR::value_type> operator+(const EXPR& mat, const typename EXPR::value_type& scalar) {
        return detail::matrix_op_scalar_expr(mat, scalar,[](auto x, auto y) { return x + y; });
    }
    
    template<numpp_matrix EXPR> requires (can_sub<typename EXPR::value_type>)
    matrix<typename EXPR::value_type> operator-(const EXPR& mat, const typename EXPR::value_type& scalar) {
        return detail::matrix_op_scalar_expr(mat, scalar,[](auto x, auto y) { return x - y; });
    }
    
    template<numpp_matrix EXPR> requires (can_mul<typename EXPR::value_type>)
    matrix<typename EXPR::value_type> operator*(const EXPR& mat, const typename EXPR::value_type& scalar) {
        return detail::matrix_op_scalar_expr(mat, scalar,[](auto x, auto y) { return x * y; });
    }

    
    template<numpp_matrix EXPR> requires (can_div<typename EXPR::value_type>)
    matrix<typename EXPR::value_type> operator/(const EXPR& mat, const typename EXPR::value_type& scalar) {
        return detail::matrix_op_scalar_expr(mat, scalar,[](auto x, auto y) { return x / y; });
    }

    template<numpp_matrix EXPR> requires (can_add<typename EXPR::value_type>)
    matrix<typename EXPR::value_type> operator+(const typename EXPR::value_type& scalar, const EXPR& mat) {
        return detail::matrix_op_scalar_expr(mat, scalar,[](auto x, auto y) { return y + x; });
    }

    template<numpp_matrix EXPR> requires (can_sub<typename EXPR::value_type>)
    matrix<typename EXPR::value_type> operator-(const typename EXPR::value_type& scalar, const EXPR& mat) {
        return detail::matrix_op_scalar_expr(mat, scalar,[](auto x, auto y) { return y - x; });
    }
    
    template<numpp_matrix EXPR> requires (can_mul<typename EXPR::value_type>)
    matrix<typename EXPR::value_type> operator*(const typename EXPR::value_type& scalar, const EXPR& mat) {
        return detail::matrix_op_scalar_expr(mat, scalar,[](auto x, auto y) { return y * x; });
    }
    
    template<numpp_matrix EXPR> requires (can_div<typename EXPR::value_type>)
    matrix<typename EXPR::value_type> operator/(const typename EXPR::value_type& scalar, const EXPR& mat) {
        return detail::matrix_op_scalar_expr(mat, scalar,[](auto x, auto y) { return y / x; });
    }

    template<class Derived, typename T>
    matrix_base<Derived, T>& matrix_base<Derived, T>::operator+=(const T& scalar) requires (can_add_assign<T>) {
        return detail::matrix_op_assign_scalar_expr(derived(), scalar, [](auto& x, const auto& y) { x += y; });
    }

    template<class Derived, typename T>
    matrix_base<Derived, T>& matrix_base<Derived, T>::operator-=(const T& scalar) requires (can_sub_assign<T>) {
        return detail::matrix_op_assign_scalar_expr(derived(), scalar, [](auto& x, const auto& y) { x -= y; });
    }
    
    template<class Derived, typename T>
    matrix_base<Derived, T>& matrix_base<Derived, T>::operator*=(const T& scalar) requires (can_mul_assign<T>) {
        return detail::matrix_op_assign_scalar_expr(derived(), scalar, [](auto& x, const auto& y) { x *= y; });
    }

    template<class Derived, typename T>
    matrix_base<Derived, T>& matrix_base<Derived, T>::operator/=(const T& scalar) requires (can_div_assign<T>) {
        return detail::matrix_op_assign_scalar_expr(derived(), scalar, [](auto& x, const auto& y) { x /= y; });
    }
}

#endif //NUMPP_MATRIX_SCALAR_ARITHMETIC_HPP