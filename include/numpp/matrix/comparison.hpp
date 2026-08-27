#ifndef NUMPP_MATRIX_COMPARISON_HPP
#define NUMPP_MATRIX_COMPARISON_HPP

#include <numpp/matrix/tool.hpp>
#include <numpp/matrix/matrix.hpp>
#include <numpp/matrix/core.hpp>
#include <numpp/matrix/op_expr.hpp>
#include <string>

namespace numpp {
    template<matrix_like A, matrix_like B>
    requires(std::same_as<typename A::value_type, typename B::value_type> && (numpp_matrix<A> || numpp_matrix<B>))
    matrix<bool> operator<(const A& a, const B& b) {
        return detail::matrix_binary_expr(a,b,[](auto x, auto y) { return x < y; }, "less");
    }

    template<matrix_like A, matrix_like B>
    requires(std::same_as<typename A::value_type, typename B::value_type> && (numpp_matrix<A> || numpp_matrix<B>))
    matrix<bool> operator>(const A& a, const B& b) {
        return detail::matrix_binary_expr(a,b,[](auto x, auto y) { return x > y; }, "greater");
    }

    template<matrix_like A, matrix_like B>
    requires(std::same_as<typename A::value_type, typename B::value_type> && (numpp_matrix<A> || numpp_matrix<B>))
    matrix<bool> operator<=(const A& a, const B& b) {
        return detail::matrix_binary_expr(a,b,[](auto x, auto y) { return x <= y; }, "less-equal");
    }
    
    template<matrix_like A, matrix_like B>
    requires(std::same_as<typename A::value_type, typename B::value_type> && (numpp_matrix<A> || numpp_matrix<B>))
    matrix<bool> operator>=(const A& a, const B& b) {
        return detail::matrix_binary_expr(a,b,[](auto x, auto y) { return x >= y; }, "greater-equal");
    }
    
    template<matrix_like A, matrix_like B>
    requires(std::same_as<typename A::value_type, typename B::value_type> && (numpp_matrix<A> || numpp_matrix<B>))
    matrix<bool> operator==(const A& a, const B& b) {
        return detail::matrix_binary_expr(a,b,[](auto x, auto y) { return x == y; }, "equal");
    }
    
    template<matrix_like A, matrix_like B>
    requires(std::same_as<typename A::value_type, typename B::value_type> && (numpp_matrix<A> || numpp_matrix<B>))
    matrix<bool> operator!=(const A& a, const B& b) {
        return detail::matrix_binary_expr(a,b,[](auto x, auto y) { return x != y; }, "not-equal");
    }
    
    template<matrix_like A, matrix_like B>
    requires(std::same_as<typename A::value_type, typename B::value_type> && (numpp_matrix<A> || numpp_matrix<B>))
    matrix<bool> operator&&(const A& a, const B& b) {
        return detail::matrix_binary_expr(a,b,[](auto x, auto y) { return x && y; }, "and");
    }
    
    template<matrix_like A, matrix_like B>
    requires(std::same_as<typename A::value_type, typename B::value_type> && (numpp_matrix<A> || numpp_matrix<B>))
    matrix<bool> operator||(const A& a, const B& b) {
        return detail::matrix_binary_expr(a,b,[](auto x, auto y) { return x || y; }, "or");
    }
}

#endif //NUMPP_MATRIX_COMPARISON_HPP