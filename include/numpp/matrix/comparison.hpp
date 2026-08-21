#ifndef NUMPP_MATRIX_COMPARISON_HPP
#define NUMPP_MATRIX_COMPARISON_HPP

#include <numpp/matrix/tool.hpp>
#include <numpp/matrix/matrix.hpp>
#include <numpp/matrix/core.hpp>
#include <numpp/matrix/op_expr.hpp>
#include <string>

namespace numpp {
    template<numpp_matrix A, matrix_like B>
    requires(std::same_as<typename A::value_type, typename B::value_type>)
    matrix<bool> operator<(const A& a, const B& b) {
        return detail::matrix_op_expr(a,b,[](auto x, auto y) { return x < y; }, "less");
    }

    template<numpp_matrix A, matrix_like B>
    requires(std::same_as<typename A::value_type, typename B::value_type>)
    matrix<bool> operator>(const A& a, const B& b) {
        return detail::matrix_op_expr(a,b,[](auto x, auto y) { return x > y; }, "greater");
    }

    template<numpp_matrix A, matrix_like B>
    requires(std::same_as<typename A::value_type, typename B::value_type>)
    matrix<bool> operator<=(const A& a, const B& b) {
        return detail::matrix_op_expr(a,b,[](auto x, auto y) { return x <= y; }, "less-equal");
    }
    
    template<numpp_matrix A, matrix_like B>
    requires(std::same_as<typename A::value_type, typename B::value_type>)matrix<bool> operator>=(const A& a, const B& b) {
        return detail::matrix_op_expr(a,b,[](auto x, auto y) { return x >= y; }, "greater-equal");
    }
    
    template<numpp_matrix A, matrix_like B>
    requires(std::same_as<typename A::value_type, typename B::value_type>)
    matrix<bool> operator==(const A& a, const B& b) {
        return detail::matrix_op_expr(a,b,[](auto x, auto y) { return x == y; }, "equal");
    }
    
    template<numpp_matrix A, matrix_like B>
    requires(std::same_as<typename A::value_type, typename B::value_type>)
    matrix<bool> operator!=(const A& a, const B& b) {
        return detail::matrix_op_expr(a,b,[](auto x, auto y) { return x != y; }, "not-equal");
    }
    
    template<numpp_matrix A, matrix_like B>
    requires(std::same_as<typename A::value_type, typename B::value_type>)
    matrix<bool> operator&&(const A& a, const B& b) {
        return detail::matrix_op_expr(a,b,[](auto x, auto y) { return x && y; }, "and");
    }
    
    template<numpp_matrix A, matrix_like B>
    requires(std::same_as<typename A::value_type, typename B::value_type>)
    matrix<bool> operator||(const A& a, const B& b) {
        return detail::matrix_op_expr(a,b,[](auto x, auto y) { return x || y; }, "or");
    }

    template<matrix_like A, numpp_matrix B>
    requires(!numpp_matrix<A> && std::same_as<typename A::value_type, typename B::value_type>)
    matrix<bool> operator<(const A& a, const B& b) {
        return detail::matrix_op_expr(a,b,[](auto x, auto y) { return x < y; }, "less");
    }

    template<matrix_like A, numpp_matrix B>
    requires(!numpp_matrix<A> && std::same_as<typename A::value_type, typename B::value_type>)
    matrix<bool> operator>(const A& a, const B& b) {
        return detail::matrix_op_expr(a,b,[](auto x, auto y) { return x > y; }, "greater");
    }

    template<matrix_like A, numpp_matrix B>
    requires(!numpp_matrix<A> && std::same_as<typename A::value_type, typename B::value_type>)
    matrix<bool> operator<=(const A& a, const B& b) {
        return detail::matrix_op_expr(a,b,[](auto x, auto y) { return x <= y; }, "less-equal");
    }
    
    template<matrix_like A, numpp_matrix B>
    requires(!numpp_matrix<A> && std::same_as<typename A::value_type, typename B::value_type>)
    matrix<bool> operator>=(const A& a, const B& b) {
        return detail::matrix_op_expr(a,b,[](auto x, auto y) { return x >= y; }, "greater-equal");
    }
    
    template<matrix_like A, numpp_matrix B>
    requires(!numpp_matrix<A> && std::same_as<typename A::value_type, typename B::value_type>)
    matrix<bool> operator==(const A& a, const B& b) {
        return detail::matrix_op_expr(a,b,[](auto x, auto y) { return x == y; }, "equal");
    }
    
    template<matrix_like A, numpp_matrix B>
    requires(!numpp_matrix<A> && std::same_as<typename A::value_type, typename B::value_type>)
    matrix<bool> operator!=(const A& a, const B& b) {
        return detail::matrix_op_expr(a,b,[](auto x, auto y) { return x != y; }, "not-equal");
    }
    
    template<matrix_like A, numpp_matrix B>
    requires(!numpp_matrix<A> && std::same_as<typename A::value_type, typename B::value_type>)
    matrix<bool> operator&&(const A& a, const B& b) {
        return detail::matrix_op_expr(a,b,[](auto x, auto y) { return x && y; }, "and");
    }
    
    template<matrix_like A, numpp_matrix B>
    requires(!numpp_matrix<A> && std::same_as<typename A::value_type, typename B::value_type>)
    matrix<bool> operator||(const A& a, const B& b) {
        return detail::matrix_op_expr(a,b,[](auto x, auto y) { return x || y; }, "or");
    }

    template<numpp_matrix EXPR>
    requires(std::same_as<typename EXPR::value_type, bool>)
    matrix<bool> operator!(const EXPR& mat) {
        return detail::matrix_one_op_expr(mat,[](auto& x) { return !x; });
    }
}

#endif //NUMPP_MATRIX_COMPARISON_HPP