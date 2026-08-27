#ifndef NUMPP_MATRIX_MATH_HPP
#define NUMPP_MATRIX_MATH_HPP

#include <numpp/matrix/core.hpp>
#include <numpp/matrix/matrix.hpp>
#include <numpp/matrix/tool.hpp>
#include <numpp/matrix/op_expr.hpp>
#include <numpp/math.hpp>

namespace numpp {
    template<typename Derived, typename T>
    matrix<T> matrix_base<Derived,T>::sqrt() const {
        return detail::matrix_unary_expr(derived(),[](auto x){ return ::numpp::sqrt(x); });
    }
    
    template<typename Derived, typename T>
    matrix<T> matrix_base<Derived,T>::cbrt() const {
        return detail::matrix_unary_expr(derived(),[](auto x){ return ::numpp::cbrt(x); });
    }
    
    template<typename Derived, typename T>
    matrix<T> matrix_base<Derived,T>::cos() const {
        return detail::matrix_unary_expr(derived(),[](auto x){ return ::numpp::cos(x); });
    }
    
    template<typename Derived, typename T>
    matrix<T> matrix_base<Derived,T>::sin() const {
        return detail::matrix_unary_expr(derived(),[](auto x){ return ::numpp::sin(x); });
    }

    template<typename Derived, typename T>
    matrix<T> matrix_base<Derived,T>::tan() const {
        return detail::matrix_unary_expr(derived(),[](auto x){ return ::numpp::tan(x); });
    }
    
    template<typename Derived, typename T>
    matrix<T> matrix_base<Derived,T>::acos() const {
        return detail::matrix_unary_expr(derived(),[](auto x){ return ::numpp::acos(x); });
    }
    
    template<typename Derived, typename T>
    matrix<T> matrix_base<Derived,T>::asin() const {
        return detail::matrix_unary_expr(derived(),[](auto x){ return ::numpp::asin(x); });
    }

    template<typename Derived, typename T>
    matrix<T> matrix_base<Derived,T>::atan() const {
        return detail::matrix_unary_expr(derived(),[](auto x){ return ::numpp::atan(x); });
    }
    
    template<typename Derived, typename T>
    matrix<T> matrix_base<Derived,T>::cosh() const {
        return detail::matrix_unary_expr(derived(),[](auto x){ return ::numpp::cosh(x); });
    }
    
    template<typename Derived, typename T>
    matrix<T> matrix_base<Derived,T>::sinh() const {
        return detail::matrix_unary_expr(derived(),[](auto x){ return ::numpp::sinh(x); });
    }

    template<typename Derived, typename T>
    matrix<T> matrix_base<Derived,T>::tanh() const {
        return detail::matrix_unary_expr(derived(),[](auto x){ return ::numpp::tanh(x); });
    }

    template<typename Derived, typename T>
    matrix<T> matrix_base<Derived,T>::log() const {
        return detail::matrix_unary_expr(derived(),[](auto x){ return ::numpp::log(x); });
    }
    
    template<typename Derived, typename T>
    matrix<T> matrix_base<Derived,T>::log2() const {
        return detail::matrix_unary_expr(derived(),[](auto x){ return ::numpp::log2(x); });
    }
    
    template<typename Derived, typename T>
    matrix<T> matrix_base<Derived,T>::log10() const {
        return detail::matrix_unary_expr(derived(),[](auto x){ return ::numpp::log10(x); });
    }
    
    template<typename Derived, typename T>
    matrix<T> matrix_base<Derived,T>::exp() const {
        return detail::matrix_unary_expr(derived(),[](auto x){ return ::numpp::exp(x); });
    }
    
    template<typename Derived, typename T>
    matrix<T> matrix_base<Derived,T>::ceil() const {
        return detail::matrix_unary_expr(derived(),[](auto x){ return ::numpp::ceil(x); });
    }
    
    template<typename Derived, typename T>
    matrix<T> matrix_base<Derived,T>::floor() const {
        return detail::matrix_unary_expr(derived(),[](auto x){ return ::numpp::floor(x); });
    }
    
    template<typename Derived, typename T>
    matrix<T> matrix_base<Derived,T>::round() const {
        return detail::matrix_unary_expr(derived(),[](auto x){ return ::numpp::round(x); });
    }
    
    template<typename Derived, typename T>
    matrix<T> matrix_base<Derived,T>::trunc() const {
        return detail::matrix_unary_expr(derived(),[](auto x){ return ::numpp::trunc(x); });
    }
    
    template<typename Derived, typename T>
    matrix<T> matrix_base<Derived,T>::abs() const {
        return detail::matrix_unary_expr(derived(),[](auto x){ return ::numpp::abs(x); });
    }

    template<typename Derived, typename T>
    matrix<T> matrix_base<Derived,T>::neg() const {
        return detail::matrix_unary_expr(derived(),[](auto x){ return ::numpp::neg(x); });
    }

    template<typename Derived, typename T>
    T matrix_base<Derived,T>::max() const {
        return detail::matrix_reduction_expr(derived(),[](auto& x, auto y){ x = ::numpp::max(x,y); }, "max");
    }
    
    template<typename Derived, typename T>
    T matrix_base<Derived,T>::min() const {
        return detail::matrix_reduction_expr(derived(),[](auto& x, auto y){ x = ::numpp::min(x,y); }, "min");
    }
    
    template<typename Derived, typename T>
    size_t matrix_base<Derived, T>::argmax() const {
        if (size_ == 0)
            throw std::invalid_argument("numpp::operation<argmax> error: cannot find maximum of an empty matrix");
        if (is_contiguous(derived())) {
            size_t out = 0;
            for (size_t i = 1; i < size_; ++i) {
                if (data_[i+offset_] > data_[out+offset_])
                    out = i;
            }
            return out;
        }

        size_t out = index_translate(0, 0);
        T max_value = (*this)(0, 0);
        for (size_t i = 0; i < row_; ++i) {
            for (size_t j = 0; j < col_; ++j) {
                const T& value = (*this)(i, j);
                if (value > max_value) {
                    max_value = value;
                    out = index_translate(i, j);
                }
            }
        }
        return out;
    }

    template<typename Derived, typename T>
    size_t matrix_base<Derived, T>::argmin() const {
        if (size_ == 0)
            throw std::invalid_argument("numpp::operation<argmin> error: cannot find minimum of an empty matrix");
        if (is_contiguous(derived())) {
            size_t out = 0;
            for (size_t i = 1; i < size_; ++i) {
                if (data_[i+offset_] < data_[out+offset_])
                    out = i;
            }
            return out;
        }

        size_t out = index_translate(0, 0);
        T min_value = (*this)(0, 0);
        for (size_t i = 0; i < row_; ++i) {
            for (size_t j = 0; j < col_; ++j) {
                const T& value = (*this)(i, j);
                if (value < min_value) {
                    min_value = value;
                    out = index_translate(i, j);
                }
            }
        }
        return out;
    }
    
    template<typename Derived, typename T>
    T matrix_base<Derived, T>::sum() const requires(can_add_assign<T>) {
        return detail::matrix_reduction_expr(derived(),[](auto& x, auto y){ x += y; }, "sum");
    }
    
    template<typename Derived, typename T>
    T matrix_base<Derived, T>::prod() const requires(can_mul_assign<T>) {
        return detail::matrix_reduction_expr(derived(),[](auto& x, auto y){ x *= y; }, "prod");
    }
    
    template<typename Derived, typename T>
    T matrix_base<Derived, T>::mean() const requires(can_add_assign<T> && can_div<T>) {
        return (*this).sum() / size_;
    }

    template<matrix_like EXPR>
    auto sqrt(const EXPR& mat) {
        return detail::matrix_unary_expr(mat,[](auto x){ return ::numpp::sqrt(x); });
    }
    
    template<matrix_like EXPR>
    auto cbrt(const EXPR& mat) {
        return detail::matrix_unary_expr(mat,[](auto x){ return ::numpp::cbrt(x); });
    }

    template<matrix_like A, matrix_like B>
    auto hypot(const A& a, const B& b) {
        return detail::matrix_binary_expr(a,b,[](auto x, auto y){ return ::numpp::hypot(x,y); }, "hypot");
    }
    
    template<matrix_like A, matrix_like B>
    auto pow(const A& a, const B& b) {
        return detail::matrix_binary_expr(a,b,[](auto x, auto y){ return ::numpp::pow(x,y); }, "pow");
    }

    template<matrix_like EXPR>
    auto pow(const EXPR& a, const typename EXPR::value_type& scalar) {
        return detail::matrix_scalar_expr(a,scalar,[](auto x, auto y){ return ::numpp::pow(x,y); });
    }
    
    template<matrix_like EXPR>
    auto cos(const EXPR& mat) {
        return detail::matrix_unary_expr(mat,[](auto x){ return ::numpp::cos(x); });
    }
    
    template<matrix_like EXPR>
    auto sin(const EXPR& mat) {
        return detail::matrix_unary_expr(mat,[](auto x){ return ::numpp::sin(x); });
    }

    template<matrix_like EXPR>
    auto tan(const EXPR& mat) {
        return detail::matrix_unary_expr(mat,[](auto x){ return ::numpp::tan(x); });
    }
    
    template<matrix_like EXPR>
    auto acos(const EXPR& mat) {
        return detail::matrix_unary_expr(mat,[](auto x){ return ::numpp::acos(x); });
    }
    
    template<matrix_like EXPR>
    auto asin(const EXPR& mat) {
        return detail::matrix_unary_expr(mat,[](auto x){ return ::numpp::asin(x); });
    }

    template<matrix_like EXPR>
    auto atan(const EXPR& mat) {
        return detail::matrix_unary_expr(mat,[](auto x){ return ::numpp::atan(x); });
    }
    
    template<matrix_like EXPR>
    auto cosh(const EXPR& mat) {
        return detail::matrix_unary_expr(mat,[](auto x){ return ::numpp::cosh(x); });
    }
    
    template<matrix_like EXPR>
    auto sinh(const EXPR& mat) {
        return detail::matrix_unary_expr(mat,[](auto x){ return ::numpp::sinh(x); });
    }

    template<matrix_like EXPR>
    auto tanh(const EXPR& mat) {
        return detail::matrix_unary_expr(mat,[](auto x){ return ::numpp::tanh(x); });
    }

    template<matrix_like EXPR>
    auto log(const EXPR& mat) {
        return detail::matrix_unary_expr(mat,[](auto x){ return ::numpp::log(x); });
    }
    
    template<matrix_like EXPR>
    auto log2(const EXPR& mat) {
        return detail::matrix_unary_expr(mat,[](auto x){ return ::numpp::log2(x); });
    }
    
    template<matrix_like EXPR>
    auto log10(const EXPR& mat) {
        return detail::matrix_unary_expr(mat,[](auto x){ return ::numpp::log10(x); });
    }
    
    template<matrix_like EXPR>
    auto exp(const EXPR& mat) {
        return detail::matrix_unary_expr(mat,[](auto x){ return ::numpp::exp(x); });
    }
    
    template<matrix_like EXPR>
    auto ceil(const EXPR& mat) {
        return detail::matrix_unary_expr(mat,[](auto x){ return ::numpp::ceil(x); });
    }
    
    template<matrix_like EXPR>
    auto floor(const EXPR& mat) {
        return detail::matrix_unary_expr(mat,[](auto x){ return ::numpp::floor(x); });
    }
    
    template<matrix_like EXPR>
    auto round(const EXPR& mat) {
        return detail::matrix_unary_expr(mat,[](auto x){ return ::numpp::round(x); });
    }
    
    template<matrix_like EXPR>
    auto trunc(const EXPR& mat) {
        return detail::matrix_unary_expr(mat,[](auto x){ return ::numpp::trunc(x); });
    }
    
    template<matrix_like EXPR>
    auto abs(const EXPR& mat) {
        return detail::matrix_unary_expr(mat,[](auto x){ return ::numpp::abs(x); });
    }
    
    template<matrix_like EXPR>
    auto neg(const EXPR& mat) {
        return detail::matrix_unary_expr(mat,[](auto x){ return ::numpp::neg(x); });
    }
    
    template<matrix_like EXPR>
    auto max(const EXPR& mat) {
        return detail::matrix_reduction_expr(mat,[](auto& x, auto y){ x = ::numpp::max(x,y); }, "max");
    }
    
    template<matrix_like EXPR>
    auto min(const EXPR& mat) {
        return detail::matrix_reduction_expr(mat,[](auto& x, auto y){ x = ::numpp::min(x,y); }, "min");
    }
    
    template<matrix_like EXPR>
    size_t argmax(const EXPR& mat) {
        if (mat.size() == 0)
            throw std::invalid_argument("numpp::operation<argmax> error: cannot find maximum of an empty matrix");

        if (is_contiguous(mat)) {
            size_t out = 0;
            const auto* data = mat.data() + mat.offset();
            for (size_t i = 1; i < mat.size(); ++i) {
                if (data[i] > data[out]) out = i;
            }
            return out;
        }

        size_t out = mat.index_translate(0, 0);
        auto max_value = mat(0, 0);
        for (size_t i = 0; i < mat.row(); ++i) {
            for (size_t j = 0; j < mat.col(); ++j) {
                const auto& value = mat(i, j);
                if (value > max_value) {
                    max_value = value;
                    out = mat.index_translate(i, j);
                }
            }
        }
        return out;
    }

    template<matrix_like EXPR>
    size_t argmin(const EXPR& mat) {
        if (mat.size() == 0)
            throw std::invalid_argument("numpp::operation<argmin> error: cannot find minimum of an empty matrix");

        if (is_contiguous(mat)) {
            size_t out = 0;
            const auto* data = mat.data() + mat.offset();
            for (size_t i = 1; i < mat.size(); ++i) {
                if (data[i] < data[out]) out = i;
            }
            return out;
        }

        size_t out = mat.index_translate(0, 0);
        auto min_value = mat(0, 0);
        for (size_t i = 0; i < mat.row(); ++i) {
            for (size_t j = 0; j < mat.col(); ++j) {
                const auto& value = mat(i, j);
                if (value < min_value) {
                    min_value = value;
                    out = mat.index_translate(i, j);
                }
            }
        }
        return out;
    }

    template<matrix_like A, matrix_like B>
    auto maximum(const A& a, const B& b) {
        return detail::matrix_binary_expr(a,b,[](auto x, auto y){ return ::numpp::max(x,y); }, "maximum");
    }
    
    template<matrix_like A, matrix_like B>
    auto minimum(const A& a, const B& b) {
        return detail::matrix_binary_expr(a,b,[](auto x, auto y){ return ::numpp::min(x,y); }, "minimum");
    }
    
    template<matrix_like EXPR>
    auto maximum(const EXPR& mat, const typename EXPR::value_type& scalar) {
        return detail::matrix_scalar_expr(mat,scalar,[](auto x, auto y){ return ::numpp::max(x,y); });
    }
    
    template<matrix_like EXPR>
    auto minimum(const EXPR& mat, const typename EXPR::value_type& scalar) {
        return detail::matrix_scalar_expr(mat,scalar,[](auto x, auto y){ return ::numpp::min(x,y); });
    }
    
    template<matrix_like EXPR>
    auto maximum(const typename EXPR::value_type& scalar, const EXPR& mat) {
        return detail::matrix_scalar_expr(mat,scalar,[](auto x, auto y){ return ::numpp::max(y,x); });
    }
    
    template<matrix_like EXPR>
    auto minimum(const typename EXPR::value_type& scalar, const EXPR& mat) {
        return detail::matrix_scalar_expr(mat,scalar,[](auto x, auto y){ return ::numpp::min(y,x); });
    }
    
    template<matrix_like EXPR>
    requires(can_add_assign<typename EXPR::value_type>)
    auto sum(const EXPR& mat) {
        return detail::matrix_reduction_expr(mat,[](auto& x, auto y){ x += y; }, "sum");
    }
    
    template<matrix_like EXPR>
    requires(can_mul_assign<typename EXPR::value_type>)
    auto prod(const EXPR& mat) {
        return detail::matrix_reduction_expr(mat,[](auto& x, auto y){ x *= y; }, "prod");
    }
    
    template<matrix_like EXPR>
    requires(can_add_assign<typename EXPR::value_type> && can_div<typename EXPR::value_type>)
    auto mean(const EXPR& mat) {
        return mat.sum() / mat.size();
    }
}

#endif //NUMPP_MATRIX_MATH_HPP