#ifndef NUMPP_MATRIX_SCALAR_ARITHMETIC_HPP
#define NUMPP_MATRIX_SCALAR_ARITHMETIC_HPP

#include <numpp/matrix/core.hpp>
#include <numpp/matrix/tool.hpp>
#include <numpp/matrix/matrix.hpp>
#include <numpp/matrix/op_expr.hpp>

namespace numpp {
    template<numpp_matrix EXPR> requires (can_add<typename EXPR::value_type>)
    matrix<typename EXPR::value_type> operator+(const EXPR& mat, const typename EXPR::value_type& scalar) {
        return detail::matrix_scalar_expr(mat, scalar,[](auto x, auto y) { return x + y; });
    }
    
    template<numpp_matrix EXPR> requires (can_sub<typename EXPR::value_type>)
    matrix<typename EXPR::value_type> operator-(const EXPR& mat, const typename EXPR::value_type& scalar) {
        return detail::matrix_scalar_expr(mat, scalar,[](auto x, auto y) { return x - y; });
    }
    
    template<numpp_matrix EXPR> requires (can_mul<typename EXPR::value_type>)
    matrix<typename EXPR::value_type> operator*(const EXPR& mat, const typename EXPR::value_type& scalar) {
        return detail::matrix_scalar_expr(mat, scalar,[](auto x, auto y) { return x * y; });
    }

    
    template<numpp_matrix EXPR> requires (can_div<typename EXPR::value_type>)
    matrix<typename EXPR::value_type> operator/(const EXPR& mat, const typename EXPR::value_type& scalar) {
        return detail::matrix_scalar_expr(mat, scalar,[](auto x, auto y) { return x / y; });
    }

    template<numpp_matrix EXPR> requires (can_add<typename EXPR::value_type>)
    matrix<typename EXPR::value_type> operator+(const typename EXPR::value_type& scalar, const EXPR& mat) {
        return detail::matrix_scalar_expr(mat, scalar,[](auto x, auto y) { return y + x; });
    }

    template<numpp_matrix EXPR> requires (can_sub<typename EXPR::value_type>)
    matrix<typename EXPR::value_type> operator-(const typename EXPR::value_type& scalar, const EXPR& mat) {
        return detail::matrix_scalar_expr(mat, scalar,[](auto x, auto y) { return y - x; });
    }
    
    template<numpp_matrix EXPR> requires (can_mul<typename EXPR::value_type>)
    matrix<typename EXPR::value_type> operator*(const typename EXPR::value_type& scalar, const EXPR& mat) {
        return detail::matrix_scalar_expr(mat, scalar,[](auto x, auto y) { return y * x; });
    }
    
    template<numpp_matrix EXPR> requires (can_div<typename EXPR::value_type>)
    matrix<typename EXPR::value_type> operator/(const typename EXPR::value_type& scalar, const EXPR& mat) {
        return detail::matrix_scalar_expr(mat, scalar,[](auto x, auto y) { return y / x; });
    }

    template<class Derived, typename T>
    Derived& matrix_base<Derived, T>::operator+=(const T& scalar) requires (can_add_assign<T>) {
        return detail::matrix_scalar_assign_expr(derived(), scalar, [](auto& x, const auto& y) { x += y; });
    }

    template<class Derived, typename T>
    Derived& matrix_base<Derived, T>::operator-=(const T& scalar) requires (can_sub_assign<T>) {
        return detail::matrix_scalar_assign_expr(derived(), scalar, [](auto& x, const auto& y) { x -= y; });
    }
    
    template<class Derived, typename T>
    Derived& matrix_base<Derived, T>::operator*=(const T& scalar) requires (can_mul_assign<T>) {
        return detail::matrix_scalar_assign_expr(derived(), scalar, [](auto& x, const auto& y) { x *= y; });
    }

    template<class Derived, typename T>
    Derived& matrix_base<Derived, T>::operator/=(const T& scalar) requires (can_div_assign<T>) {
        return detail::matrix_scalar_assign_expr(derived(), scalar, [](auto& x, const auto& y) { x /= y; });
    }
}

#endif //NUMPP_MATRIX_SCALAR_ARITHMETIC_HPP