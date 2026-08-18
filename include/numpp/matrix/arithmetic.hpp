#ifndef NUMPP_MATRIX_ARITHMETIC_HPP
#define NUMPP_MATRIX_ARITHMETIC_HPP

#include <numpp/matrix/core.hpp>
#include <numpp/matrix/tool.hpp>
#include <numpp/matrix/matrix.hpp>
#include <numpp/matrix/linalg/matmul.hpp>
#include <numpp/matrix/op_expr.hpp>

namespace numpp {
    template<numpp_matrix A, matrix_like B>
    requires (
        std::same_as<
            typename A::value_type,
            typename B::value_type
        > &&
        can_add<typename A::value_type>
    )
    matrix<typename A::value_type> operator+(const A& a, const B& b) {
        return detail::matrix_op_expr(a,b,[](auto x, auto y) { return x + y; }, "add");
    }

    template<numpp_matrix A, matrix_like B>
    requires (
        std::same_as<
            typename A::value_type,
            typename B::value_type
        > &&
        can_sub<typename A::value_type>
    )
    matrix<typename A::value_type> operator-(const A& a, const B& b) {
        return detail::matrix_op_expr(a,b,[](auto x, auto y) { return x - y; }, "sub");
    }
    
    template<numpp_matrix A, matrix_like B>
    requires (
        std::same_as<
            typename A::value_type,
            typename B::value_type
        > &&
        can_mul<typename A::value_type>
    )
    matrix<typename A::value_type> elementwise_mul(const A& a, const B& b) {
        return detail::matrix_op_expr(a,b,[](auto x, auto y) { return x * y; }, "mul");
    }
    
    template<numpp_matrix A, matrix_like B>
    requires (
        std::same_as<
            typename A::value_type,
            typename B::value_type
        > &&
        is_numeric<typename A::value_type>
    )
    matrix<typename A::value_type> operator*(const A& a, const B& b) {
        return linalg::matmul(a,b);
    }
    
    template<numpp_matrix A, matrix_like B>
    requires (
        std::same_as<
            typename A::value_type,
            typename B::value_type
        > &&
        can_div<typename A::value_type>
    )
    matrix<typename A::value_type> operator/(const A& a, const B& b) {
        return detail::matrix_op_expr(a,b,[](auto x, auto y) { return x / y; }, "div");
    }
    
    template<matrix_like A, numpp_matrix B>
    requires (
        !numpp_matrix<A> &&
        std::same_as<
            typename A::value_type,
            typename B::value_type
        > &&
        can_add<typename A::value_type>
    )
    matrix<typename A::value_type> operator+(const A& a, const B& b) {
        return detail::matrix_op_expr(a,b,[](auto x, auto y) { return x + y; }, "add");
    }

    template<matrix_like A, numpp_matrix B>
    requires (
        !numpp_matrix<A> &&
        std::same_as<
            typename A::value_type,
            typename B::value_type
        > &&
        can_sub<typename A::value_type>
    )
    matrix<typename A::value_type> operator-(const A& a, const B& b) {
        return detail::matrix_op_expr(a,b,[](auto x, auto y) { return x - y; }, "sub");
    }
    
    template<matrix_like A, numpp_matrix B>
    requires (
        !numpp_matrix<A> &&
        std::same_as<
            typename A::value_type,
            typename B::value_type
        > &&
        is_numeric<typename A::value_type>
    )
    matrix<typename A::value_type> operator*(const A& a, const B& b) {
        return linalg::matmul(a,b);
    }
    
    template<matrix_like A, numpp_matrix B>
    requires (
        !numpp_matrix<A> &&
        std::same_as<
            typename A::value_type,
            typename B::value_type
        > &&
        can_div<typename A::value_type>
    )
    matrix<typename A::value_type> operator/(const A& a, const B& b) {
        return detail::matrix_op_expr(a,b,[](auto x, auto y) { return x / y; }, "div");
    }

    template<class Derived, typename T>
    template<matrix_like B>
    requires (std::same_as<typename matrix_base<Derived, T>::value_type, typename B::value_type> && can_add_assign<T>)
    matrix_base<Derived, T>& matrix_base<Derived, T>::operator+=(const B& b) {
        return detail::matrix_op_assign_expr(derived(), b, [](auto& x, const auto& y) { x += y; }, "add-assign");
    }

    template<class Derived, typename T>
    template<matrix_like B>
    requires (std::same_as<typename matrix_base<Derived, T>::value_type, typename B::value_type> && can_sub_assign<T>)
    matrix_base<Derived, T>& matrix_base<Derived, T>::operator-=(const B& b) {
        return detail::matrix_op_assign_expr(derived(), b, [](auto& x, const auto& y) { x -= y; }, "sub-assign");
    }

    template<class Derived, typename T>
    template<matrix_like B>
    requires (std::same_as<typename matrix_base<Derived, T>::value_type, typename B::value_type> && can_mul_assign<T>)
    matrix_base<Derived, T>& matrix_base<Derived, T>::operator*=(const B& b) {
        Derived& self = derived();
        self = linalg::matmul(self, b);
        return *this;
    }

    template<class Derived, typename T>
    template<matrix_like B>
    requires (std::same_as<typename matrix_base<Derived, T>::value_type, typename B::value_type> && can_div_assign<T>)
    matrix_base<Derived, T>& matrix_base<Derived, T>::operator/=(const B& b) {
        return detail::matrix_op_assign_expr(derived(), b, [](auto& x, const auto& y) { x /= y; }, "div-assign");
    }
}

#endif //NUMPP_MATRIX_ARITHMETIC_HPP