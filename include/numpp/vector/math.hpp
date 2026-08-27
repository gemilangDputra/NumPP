#ifndef NUMPP_VECTOR_MATH_HPP
#define NUMPP_VECTOR_MATH_HPP

#include <numpp/vector/core.hpp>
#include <numpp/vector/vector.hpp>
#include <numpp/vector/tool.hpp>
#include <numpp/vector/op_expr.hpp>
#include <numpp/math.hpp>

namespace numpp {
    template<typename Derived, typename T>
    vector<T> vector_base<Derived,T>::sqrt() const {
        return detail::vector_unary_expr(derived(),[](auto x){ return ::numpp::sqrt(x); });
    }
    
    template<typename Derived, typename T>
    vector<T> vector_base<Derived,T>::cbrt() const {
        return detail::vector_unary_expr(derived(),[](auto x){ return ::numpp::cbrt(x); });
    }
    
    template<typename Derived, typename T>
    vector<T> vector_base<Derived,T>::cos() const {
        return detail::vector_unary_expr(derived(),[](auto x){ return ::numpp::cos(x); });
    }
    
    template<typename Derived, typename T>
    vector<T> vector_base<Derived,T>::sin() const {
        return detail::vector_unary_expr(derived(),[](auto x){ return ::numpp::sin(x); });
    }

    template<typename Derived, typename T>
    vector<T> vector_base<Derived,T>::tan() const {
        return detail::vector_unary_expr(derived(),[](auto x){ return ::numpp::tan(x); });
    }
    
    template<typename Derived, typename T>
    vector<T> vector_base<Derived,T>::acos() const {
        return detail::vector_unary_expr(derived(),[](auto x){ return ::numpp::acos(x); });
    }
    
    template<typename Derived, typename T>
    vector<T> vector_base<Derived,T>::asin() const {
        return detail::vector_unary_expr(derived(),[](auto x){ return ::numpp::asin(x); });
    }

    template<typename Derived, typename T>
    vector<T> vector_base<Derived,T>::atan() const {
        return detail::vector_unary_expr(derived(),[](auto x){ return ::numpp::atan(x); });
    }
    
    template<typename Derived, typename T>
    vector<T> vector_base<Derived,T>::cosh() const {
        return detail::vector_unary_expr(derived(),[](auto x){ return ::numpp::cosh(x); });
    }
    
    template<typename Derived, typename T>
    vector<T> vector_base<Derived,T>::sinh() const {
        return detail::vector_unary_expr(derived(),[](auto x){ return ::numpp::sinh(x); });
    }

    template<typename Derived, typename T>
    vector<T> vector_base<Derived,T>::tanh() const {
        return detail::vector_unary_expr(derived(),[](auto x){ return ::numpp::tanh(x); });
    }

    template<typename Derived, typename T>
    vector<T> vector_base<Derived,T>::log() const {
        return detail::vector_unary_expr(derived(),[](auto x){ return ::numpp::log(x); });
    }
    
    template<typename Derived, typename T>
    vector<T> vector_base<Derived,T>::log2() const {
        return detail::vector_unary_expr(derived(),[](auto x){ return ::numpp::log2(x); });
    }
    
    template<typename Derived, typename T>
    vector<T> vector_base<Derived,T>::log10() const {
        return detail::vector_unary_expr(derived(),[](auto x){ return ::numpp::log10(x); });
    }
    
    template<typename Derived, typename T>
    vector<T> vector_base<Derived,T>::exp() const {
        return detail::vector_unary_expr(derived(),[](auto x){ return ::numpp::exp(x); });
    }
    
    template<typename Derived, typename T>
    vector<T> vector_base<Derived,T>::ceil() const {
        return detail::vector_unary_expr(derived(),[](auto x){ return ::numpp::ceil(x); });
    }
    
    template<typename Derived, typename T>
    vector<T> vector_base<Derived,T>::floor() const {
        return detail::vector_unary_expr(derived(),[](auto x){ return ::numpp::floor(x); });
    }
    
    template<typename Derived, typename T>
    vector<T> vector_base<Derived,T>::round() const {
        return detail::vector_unary_expr(derived(),[](auto x){ return ::numpp::round(x); });
    }
    
    template<typename Derived, typename T>
    vector<T> vector_base<Derived,T>::trunc() const {
        return detail::vector_unary_expr(derived(),[](auto x){ return ::numpp::trunc(x); });
    }
    
    template<typename Derived, typename T>
    vector<T> vector_base<Derived,T>::abs() const {
        return detail::vector_unary_expr(derived(),[](auto x){ return ::numpp::abs(x); });
    }

    template<typename Derived, typename T>
    vector<T> vector_base<Derived,T>::neg() const {
        return detail::vector_unary_expr(derived(),[](auto x){ return ::numpp::neg(x); });
    }

    template<typename Derived, typename T>
    T vector_base<Derived,T>::max() const {
        return detail::vector_reduction_expr(derived(),[](auto& x, auto y){ x = ::numpp::max(x,y); }, "max");
    }
    
    template<typename Derived, typename T>
    T vector_base<Derived,T>::min() const {
        return detail::vector_reduction_expr(derived(),[](auto& x, auto y){ x = ::numpp::min(x,y); }, "min");
    }
    
    template<typename Derived, typename T>
    size_t vector_base<Derived, T>::argmax() const {
        if (size_ == 0)
            throw std::invalid_argument("numpp::operation<argmax> error: cannot find maximum of an empty vector");
        size_t out = index_translate(0);
        for (size_t i = 1; i < size_; ++i)
            if ((*this)[i] > (*this)[out]) out = i;
        return out;
    }

    template<typename Derived, typename T>
    size_t vector_base<Derived, T>::argmin() const {
        if (size_ == 0)
            throw std::invalid_argument("numpp::operation<argmin> error: cannot find minimum of an empty vector");
        size_t out = index_translate(0);
        for (size_t i = 1; i < size_; ++i)
            if ((*this)[i] < (*this)[out]) out = i;
        return out;
    }

    template<typename Derived, typename T>
    T vector_base<Derived, T>::sum() const requires(can_add_assign<T>) {
        return detail::vector_reduction_expr(derived(),[](auto& x, auto y){ x += y; }, "sum");
    }
    
    template<typename Derived, typename T>
    T vector_base<Derived, T>::prod() const requires(can_mul_assign<T>) {
        return detail::vector_reduction_expr(derived(),[](auto& x, auto y){ x *= y; }, "prod");
    }
    
    template<typename Derived, typename T>
    T vector_base<Derived, T>::mean() const requires(can_add_assign<T> && can_div<T>) {
        return (*this).sum() / size_;
    }

    template<vector_like EXPR>
    auto sqrt(const EXPR& vec) {
        return detail::vector_unary_expr(vec,[](auto x){ return ::numpp::sqrt(x); });
    }
    
    template<vector_like EXPR>
    auto cbrt(const EXPR& vec) {
        return detail::vector_unary_expr(vec,[](auto x){ return ::numpp::cbrt(x); });
    }

    template<vector_like A, vector_like B>
    auto hypot(const A& a, const B& b) {
        return detail::vector_binary_expr(a,b,[](auto x, auto y){ return ::numpp::hypot(x,y); }, "hypot");
    }
    
    template<vector_like A, vector_like B>
    auto pow(const A& a, const B& b) {
        return detail::vector_binary_expr(a,b,[](auto x, auto y){ return ::numpp::pow(x,y); }, "pow");
    }

    template<vector_like EXPR>
    auto pow(const EXPR& a, const typename EXPR::value_type& scalar) {
        return detail::vector_scalar_expr(a,scalar,[](auto x, auto y){ return ::numpp::pow(x,y); });
    }
    
    template<vector_like EXPR>
    auto cos(const EXPR& vec) {
        return detail::vector_unary_expr(vec,[](auto x){ return ::numpp::cos(x); });
    }
    
    template<vector_like EXPR>
    auto sin(const EXPR& vec) {
        return detail::vector_unary_expr(vec,[](auto x){ return ::numpp::sin(x); });
    }

    template<vector_like EXPR>
    auto tan(const EXPR& vec) {
        return detail::vector_unary_expr(vec,[](auto x){ return ::numpp::tan(x); });
    }
    
    template<vector_like EXPR>
    auto acos(const EXPR& vec) {
        return detail::vector_unary_expr(vec,[](auto x){ return ::numpp::acos(x); });
    }
    
    template<vector_like EXPR>
    auto asin(const EXPR& vec) {
        return detail::vector_unary_expr(vec,[](auto x){ return ::numpp::asin(x); });
    }

    template<vector_like EXPR>
    auto atan(const EXPR& vec) {
        return detail::vector_unary_expr(vec,[](auto x){ return ::numpp::atan(x); });
    }
    
    template<vector_like EXPR>
    auto cosh(const EXPR& vec) {
        return detail::vector_unary_expr(vec,[](auto x){ return ::numpp::cosh(x); });
    }
    
    template<vector_like EXPR>
    auto sinh(const EXPR& vec) {
        return detail::vector_unary_expr(vec,[](auto x){ return ::numpp::sinh(x); });
    }

    template<vector_like EXPR>
    auto tanh(const EXPR& vec) {
        return detail::vector_unary_expr(vec,[](auto x){ return ::numpp::tanh(x); });
    }

    template<vector_like EXPR>
    auto log(const EXPR& vec) {
        return detail::vector_unary_expr(vec,[](auto x){ return ::numpp::log(x); });
    }
    
    template<vector_like EXPR>
    auto log2(const EXPR& vec) {
        return detail::vector_unary_expr(vec,[](auto x){ return ::numpp::log2(x); });
    }
    
    template<vector_like EXPR>
    auto log10(const EXPR& vec) {
        return detail::vector_unary_expr(vec,[](auto x){ return ::numpp::log10(x); });
    }
    
    template<vector_like EXPR>
    auto exp(const EXPR& vec) {
        return detail::vector_unary_expr(vec,[](auto x){ return ::numpp::exp(x); });
    }
    
    template<vector_like EXPR>
    auto ceil(const EXPR& vec) {
        return detail::vector_unary_expr(vec,[](auto x){ return ::numpp::ceil(x); });
    }
    
    template<vector_like EXPR>
    auto floor(const EXPR& vec) {
        return detail::vector_unary_expr(vec,[](auto x){ return ::numpp::floor(x); });
    }
    
    template<vector_like EXPR>
    auto round(const EXPR& vec) {
        return detail::vector_unary_expr(vec,[](auto x){ return ::numpp::round(x); });
    }
    
    template<vector_like EXPR>
    auto trunc(const EXPR& vec) {
        return detail::vector_unary_expr(vec,[](auto x){ return ::numpp::trunc(x); });
    }
    
    template<vector_like EXPR>
    auto abs(const EXPR& vec) {
        return detail::vector_unary_expr(vec,[](auto x){ return ::numpp::abs(x); });
    }
    
    template<vector_like EXPR>
    auto neg(const EXPR& vec) {
        return detail::vector_unary_expr(vec,[](auto x){ return ::numpp::neg(x); });
    }
    
    template<vector_like EXPR>
    auto max(const EXPR& vec) {
        return detail::vector_reduction_expr(vec,[](auto& x, auto y){ x = ::numpp::max(x,y); }, "max");
    }
    
    template<vector_like EXPR>
    auto min(const EXPR& vec) {
        return detail::vector_reduction_expr(vec,[](auto& x, auto y){ x = ::numpp::min(x,y); }, "min");
    }
    
    template<vector_like EXPR>
    size_t argmax(const EXPR& vec) {
        if (vec.size() == 0)
            throw std::invalid_argument("numpp::operation<argmax> error: cannot find maximum of an empty vector");
        size_t out = vec.index_translate(0);
        for (size_t i = 1; i < vec.size(); ++i)
            if (vec[i] > vec[out]) out = i;
        return out;
    }

    template<vector_like EXPR>
    size_t argmin(const EXPR& vec) {
        if (vec.size() == 0)
            throw std::invalid_argument("numpp::operation<argmin> error: cannot find minimum of an empty vector");
        size_t out = vec.index_translate(0);
        for (size_t i = 1; i < vec.size(); ++i)
            if (vec[i] < vec[out]) out = i;
        return out;
    }

    template<vector_like A, vector_like B>
    auto maximum(const A& a, const B& b) {
        return detail::vector_binary_expr(a,b,[](auto x, auto y){ return ::numpp::max(x,y); }, "maximum");
    }
    
    template<vector_like A, vector_like B>
    auto minimum(const A& a, const B& b) {
        return detail::vector_binary_expr(a,b,[](auto x, auto y){ return ::numpp::min(x,y); }, "minimum");
    }
    
    template<vector_like EXPR>
    auto maximum(const EXPR& vec, const typename EXPR::value_type& scalar) {
        return detail::vector_scalar_expr(vec,scalar,[](auto x, auto y){ return ::numpp::max(x,y); });
    }
    
    template<vector_like EXPR>
    auto minimum(const EXPR& vec, const typename EXPR::value_type& scalar) {
        return detail::vector_scalar_expr(vec,scalar,[](auto x, auto y){ return ::numpp::min(x,y); });
    }
    
    template<vector_like EXPR>
    auto maximum(const typename EXPR::value_type& scalar, const EXPR& vec) {
        return detail::vector_scalar_expr(vec,scalar,[](auto x, auto y){ return ::numpp::max(y,x); });
    }
    
    template<vector_like EXPR>
    auto minimum(const typename EXPR::value_type& scalar, const EXPR& vec) {
        return detail::vector_scalar_expr(vec,scalar,[](auto x, auto y){ return ::numpp::min(y,x); });
    }
    
    template<vector_like EXPR>
    requires(can_add_assign<typename EXPR::value_type>)
    auto sum(const EXPR& vec) {
        return detail::vector_reduction_expr(vec,[](auto& x, auto y){ x += y; }, "sum");
    }
    
    template<vector_like EXPR>
    requires(can_mul_assign<typename EXPR::value_type>)
    auto prod(const EXPR& vec) {
        return detail::vector_reduction_expr(vec,[](auto& x, auto y){ x *= y; }, "prod");
    }
    
    template<vector_like EXPR>
    requires(can_add_assign<typename EXPR::value_type> && can_div<typename EXPR::value_type>)
    auto mean(const EXPR& vec) {
        return vec.sum() / vec.size();
    }
}

#endif //NUMPP_VECTOR_MATH_HPP