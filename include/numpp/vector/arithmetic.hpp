#ifndef NUMPP_VECTOR_ARITHMETIC_HPP
#define NUMPP_VECTOR_ARITHMETIC_HPP

#include <numpp/vector/core.hpp>
#include <numpp/vector/tool.hpp>
#include <numpp/vector/vector.hpp>
#include <numpp/vector/op_expr.hpp>

namespace numpp {
    template<general_vector_like A, general_vector_like B>
    requires (
        std::same_as<
            typename A::value_type,
            typename B::value_type
        > &&
        can_add<typename A::value_type> &&
        (numpp_vector<A> || numpp_vector<B>)
    )
    vector<typename A::value_type> operator+(const A& a, const B& b) {
        return detail::vector_binary_expr(a,b,[](auto x, auto y){ return x + y; }, "add");
    }
    
    template<general_vector_like A, general_vector_like B>
    requires (
        std::same_as<
            typename A::value_type,
            typename B::value_type
        > &&
        can_sub<typename A::value_type> &&
        (numpp_vector<A> || numpp_vector<B>)
    )
    vector<typename A::value_type> operator-(const A& a, const B& b) {
        return detail::vector_binary_expr(a,b,[](auto x, auto y){ return x - y; }, "sub");
    }
    
    template<general_vector_like A, general_vector_like B>
    requires (
        std::same_as<
            typename A::value_type,
            typename B::value_type
        > &&
        can_mul<typename A::value_type> &&
        (numpp_vector<A> || numpp_vector<B>)
    )
    vector<typename A::value_type> operator*(const A& a, const B& b) {
        return detail::vector_binary_expr(a,b,[](auto x, auto y){ return x * y; }, "mul");
    }
    
    template<general_vector_like A, general_vector_like B>
    requires (
        std::same_as<
            typename A::value_type,
            typename B::value_type
        > &&
        can_div<typename A::value_type> &&
        (numpp_vector<A> || numpp_vector<B>)
    )
    vector<typename A::value_type> operator/(const A& a, const B& b) {
        return detail::vector_binary_expr(a,b,[](auto x, auto y){ return x / y; }, "div");
    }
    
    template<class Derived, typename T>
    template<general_vector_like B>
    requires (std::same_as<typename vector_base<Derived, T>::value_type, typename B::value_type> && can_add_assign<T>)
    Derived& vector_base<Derived, T>::operator+=(const B& b) {
        return detail::vector_binary_assign_expr(derived(),b,[](auto& x, auto y){ x += y; }, "add-assign");
    }
    
    template<class Derived, typename T>
    template<general_vector_like B>
    requires (std::same_as<typename vector_base<Derived, T>::value_type, typename B::value_type> && can_sub_assign<T>)
    Derived& vector_base<Derived, T>::operator-=(const B& b) {
        return detail::vector_binary_assign_expr(derived(),b,[](auto& x, auto y){ x -= y; }, "sub-assign");
    }
    
    template<class Derived, typename T>
    template<general_vector_like B>
    requires (std::same_as<typename vector_base<Derived, T>::value_type, typename B::value_type> && can_mul_assign<T>)
    Derived& vector_base<Derived, T>::operator*=(const B& b) {
        return detail::vector_binary_assign_expr(derived(),b,[](auto& x, auto y){ x *= y; }, "mul-assign");
    }
    
    template<class Derived, typename T>
    template<general_vector_like B>
    requires (std::same_as<typename vector_base<Derived, T>::value_type, typename B::value_type> && can_div_assign<T>)
    Derived& vector_base<Derived, T>::operator/=(const B& b) {
        return detail::vector_binary_assign_expr(derived(),b,[](auto& x, auto y){ x /= y; }, "div-assign");
    }
}

#endif //NUMPP_VECTOR_ARITHMETIC_HPP