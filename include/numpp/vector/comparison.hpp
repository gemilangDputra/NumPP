#ifndef NUMPP_VECTOR_COMPARISON_HPP
#define NUMPP_VECTOR_COMPARISON_HPP

#include <numpp/vector/tool.hpp>
#include <numpp/vector/vector.hpp>
#include <numpp/vector/core.hpp>
#include <numpp/vector/op_expr.hpp>
#include <string>

namespace numpp {
    template<general_vector_like A, general_vector_like B>
    requires(std::same_as<typename A::value_type, typename B::value_type> && (numpp_vector<A> || numpp_vector<B>))
    vector<bool> operator<(const A& a, const B& b) {
        return detail::vector_binary_expr(a,b,[](auto x, auto y) { return x < y; }, "less");
    }

    template<general_vector_like A, general_vector_like B>
    requires(std::same_as<typename A::value_type, typename B::value_type> && (numpp_vector<A> || numpp_vector<B>))
    vector<bool> operator>(const A& a, const B& b) {
        return detail::vector_binary_expr(a,b,[](auto x, auto y) { return x > y; }, "greater");
    }

    template<general_vector_like A, general_vector_like B>
    requires(std::same_as<typename A::value_type, typename B::value_type> && (numpp_vector<A> || numpp_vector<B>))
    vector<bool> operator<=(const A& a, const B& b) {
        return detail::vector_binary_expr(a,b,[](auto x, auto y) { return x <= y; }, "less-equal");
    }
    
    template<general_vector_like A, general_vector_like B>
    requires(std::same_as<typename A::value_type, typename B::value_type> && (numpp_vector<A> || numpp_vector<B>))
    vector<bool> operator>=(const A& a, const B& b) {
        return detail::vector_binary_expr(a,b,[](auto x, auto y) { return x >= y; }, "greater-equal");
    }
    
    template<general_vector_like A, general_vector_like B>
    requires(std::same_as<typename A::value_type, typename B::value_type> && (numpp_vector<A> || numpp_vector<B>))
    vector<bool> operator==(const A& a, const B& b) {
        return detail::vector_binary_expr(a,b,[](auto x, auto y) { return x == y; }, "equal");
    }
    
    template<general_vector_like A, general_vector_like B>
    requires(std::same_as<typename A::value_type, typename B::value_type> && (numpp_vector<A> || numpp_vector<B>))
    vector<bool> operator!=(const A& a, const B& b) {
        return detail::vector_binary_expr(a,b,[](auto x, auto y) { return x != y; }, "not-equal");
    }
    
    template<general_vector_like A, general_vector_like B>
    requires(std::same_as<typename A::value_type, typename B::value_type> && (numpp_vector<A> || numpp_vector<B>))
    vector<bool> operator&&(const A& a, const B& b) {
        return detail::vector_binary_expr(a,b,[](auto x, auto y) { return x && y; }, "and");
    }
    
    template<general_vector_like A, general_vector_like B>
    requires(std::same_as<typename A::value_type, typename B::value_type> && (numpp_vector<A> || numpp_vector<B>))
    vector<bool> operator||(const A& a, const B& b) {
        return detail::vector_binary_expr(a,b,[](auto x, auto y) { return x || y; }, "or");
    }
}

#endif //NUMPP_VECTOR_COMPARISON_HPP