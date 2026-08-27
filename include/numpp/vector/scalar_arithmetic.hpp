#ifndef NUMPP_VECTOR_SCALAR_ARITHMETIC_HPP
#define NUMPP_VECTOR_SCALAR_ARITHMETIC_HPP

#include <numpp/vector/core.hpp>
#include <numpp/vector/tool.hpp>
#include <numpp/vector/vector.hpp>
#include <numpp/vector/op_expr.hpp>

namespace numpp {
    template<numpp_vector EXPR> requires (can_add<typename EXPR::value_type>)
    vector<typename EXPR::value_type> operator+(const EXPR& mat, const typename EXPR::value_type& scalar) {
        return detail::vector_scalar_expr(mat, scalar,[](auto x, auto y) { return x + y; });
    }
    
    template<numpp_vector EXPR> requires (can_sub<typename EXPR::value_type>)
    vector<typename EXPR::value_type> operator-(const EXPR& mat, const typename EXPR::value_type& scalar) {
        return detail::vector_scalar_expr(mat, scalar,[](auto x, auto y) { return x - y; });
    }
    
    template<numpp_vector EXPR> requires (can_mul<typename EXPR::value_type>)
    vector<typename EXPR::value_type> operator*(const EXPR& mat, const typename EXPR::value_type& scalar) {
        return detail::vector_scalar_expr(mat, scalar,[](auto x, auto y) { return x * y; });
    }

    
    template<numpp_vector EXPR> requires (can_div<typename EXPR::value_type>)
    vector<typename EXPR::value_type> operator/(const EXPR& mat, const typename EXPR::value_type& scalar) {
        return detail::vector_scalar_expr(mat, scalar,[](auto x, auto y) { return x / y; });
    }

    template<numpp_vector EXPR> requires (can_add<typename EXPR::value_type>)
    vector<typename EXPR::value_type> operator+(const typename EXPR::value_type& scalar, const EXPR& mat) {
        return detail::vector_scalar_expr(mat, scalar,[](auto x, auto y) { return y + x; });
    }

    template<numpp_vector EXPR> requires (can_sub<typename EXPR::value_type>)
    vector<typename EXPR::value_type> operator-(const typename EXPR::value_type& scalar, const EXPR& mat) {
        return detail::vector_scalar_expr(mat, scalar,[](auto x, auto y) { return y - x; });
    }
    
    template<numpp_vector EXPR> requires (can_mul<typename EXPR::value_type>)
    vector<typename EXPR::value_type> operator*(const typename EXPR::value_type& scalar, const EXPR& mat) {
        return detail::vector_scalar_expr(mat, scalar,[](auto x, auto y) { return y * x; });
    }
    
    template<numpp_vector EXPR> requires (can_div<typename EXPR::value_type>)
    vector<typename EXPR::value_type> operator/(const typename EXPR::value_type& scalar, const EXPR& mat) {
        return detail::vector_scalar_expr(mat, scalar,[](auto x, auto y) { return y / x; });
    }

    template<class Derived, typename T>
    Derived& vector_base<Derived, T>::operator+=(const T& scalar) requires (can_add_assign<T>) {
        return detail::vector_scalar_assign_expr(derived(), scalar, [](auto& x, const auto& y) { x += y; });
    }

    template<class Derived, typename T>
    Derived& vector_base<Derived, T>::operator-=(const T& scalar) requires (can_sub_assign<T>) {
        return detail::vector_scalar_assign_expr(derived(), scalar, [](auto& x, const auto& y) { x -= y; });
    }
    
    template<class Derived, typename T>
    Derived& vector_base<Derived, T>::operator*=(const T& scalar) requires (can_mul_assign<T>) {
        return detail::vector_scalar_assign_expr(derived(), scalar, [](auto& x, const auto& y) { x *= y; });
    }

    template<class Derived, typename T>
    Derived& vector_base<Derived, T>::operator/=(const T& scalar) requires (can_div_assign<T>) {
        return detail::vector_scalar_assign_expr(derived(), scalar, [](auto& x, const auto& y) { x /= y; });
    }
}

#endif //NUMPP_VECTOR_SCALAR_ARITHMETIC_HPP