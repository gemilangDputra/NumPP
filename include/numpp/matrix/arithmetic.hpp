#ifndef NUMPP_MATRIX_ARITHMETIC_HPP
#define NUMPP_MATRIX_ARITHMETIC_HPP

#include <numpp/matrix/core.hpp>
#include <numpp/matrix/tool.hpp>
#include <numpp/matrix/matrix.hpp>
#include <numpp/matrix/view/broadcast.hpp>
#include <utility>

namespace numpp {
    template<matrix_derived A, strided_matrix B>
    requires (
        std::same_as<
            typename A::value_type,
            typename B::value_type
        > &&
        can_add<typename A::value_type>
    )
    matrix<typename A::value_type> operator+(const A& a, const B& b) {
        using T = typename A::value_type;
        if (a.row() == b.row() && a.col() == b.col()) {
            matrix<T> out = matrix<T>::empty_like(a);
            if (is_contiguous(a) && is_contiguous(b)) {
                const auto* adata = a.data();
                const auto* bdata = b.data();
                auto* cdata = out.data();
                for (size_t i = 0; i < out.size(); ++i) {
                    cdata[i] = adata[i] + bdata[i];
                }
            }
            else {
                for (size_t i = 0; i < out.row(); ++i) {
                    for (size_t j = 0; j < out.col(); ++j) {
                        out(i, j) = a(i, j) + b(i, j);
                    }
                }
            }
            return out;
        }

        if (detail::is_broadcastable(a, b)) {
            auto [ab, bb] = detail::broadcast(a, b);
            matrix<T> out = matrix<T>::empty_like(ab);
            for (size_t i = 0; i < out.row(); ++i) {
                for (size_t j = 0; j < out.col(); ++j) {
                    out(i, j) = ab(i, j) + bb(i, j);
                }
            }

            return out;
        }
        throw std::invalid_argument("matrix dimensions are incompatible for addition");
    }

    template<matrix_derived A, strided_matrix B>
    requires (
        std::same_as<
            typename A::value_type,
            typename B::value_type
        > &&
        can_sub<typename A::value_type>
    )
    matrix<typename A::value_type> operator-(const A& a, const B& b) {
        using T = typename A::value_type;
        if (a.row() == b.row() && a.col() == b.col()) {
            matrix<T> out = matrix<T>::empty_like(a);
            if (is_contiguous(a) && is_contiguous(b)) {
                const auto* adata = a.data();
                const auto* bdata = b.data();
                auto* cdata = out.data();
                for (size_t i = 0; i < out.size(); ++i) {
                    cdata[i] = adata[i] - bdata[i];
                }
            }
            else {
                for (size_t i = 0; i < out.row(); ++i) {
                    for (size_t j = 0; j < out.col(); ++j) {
                        out(i, j) = a(i, j) - b(i, j);
                    }
                }
            }
            return out;
        }

        if (detail::is_broadcastable(a, b)) {
            auto [ab, bb] = detail::broadcast(a, b);
            matrix<T> out = matrix<T>::empty_like(ab);
            for (size_t i = 0; i < out.row(); ++i) {
                for (size_t j = 0; j < out.col(); ++j) {
                    out(i, j) = ab(i, j) - bb(i, j);
                }
            }

            return out;
        }
        throw std::invalid_argument("matrix dimensions are incompatible for subtraction");
    }

    template<matrix_derived A, strided_matrix B>
    requires (
        std::same_as<
            typename A::value_type,
            typename B::value_type
        > &&
        can_mul<typename A::value_type>
    )
    matrix<typename A::value_type> operator*(const A& a, const B& b) {
        using T = typename A::value_type;
        if (a.row() == b.row() && a.col() == b.col()) {
            matrix<T> out = matrix<T>::empty_like(a);
            if (is_contiguous(a) && is_contiguous(b)) {
                const auto* adata = a.data();
                const auto* bdata = b.data();
                auto* cdata = out.data();
                for (size_t i = 0; i < out.size(); ++i) {
                    cdata[i] = adata[i] * bdata[i];
                }
            }
            else {
                for (size_t i = 0; i < out.row(); ++i) {
                    for (size_t j = 0; j < out.col(); ++j) {
                        out(i, j) = a(i, j) * b(i, j);
                    }
                }
            }
            return out;
        }

        if (detail::is_broadcastable(a, b)) {
            auto [ab, bb] = detail::broadcast(a, b);
            matrix<T> out = matrix<T>::empty_like(ab);
            for (size_t i = 0; i < out.row(); ++i) {
                for (size_t j = 0; j < out.col(); ++j) {
                    out(i, j) = ab(i, j) * bb(i, j);
                }
            }

            return out;
        }
        throw std::invalid_argument("matrix dimensions are incompatible for element-wise multiplication");
    }
    
    template<matrix_derived A, strided_matrix B>
    requires (
        std::same_as<
            typename A::value_type,
            typename B::value_type
        > &&
        can_div<typename A::value_type>
    )
    matrix<typename A::value_type> operator/(const A& a, const B& b) {
        using T = typename A::value_type;
        if (a.row() == b.row() && a.col() == b.col()) {
            matrix<T> out = matrix<T>::empty_like(a);
            if (is_contiguous(a) && is_contiguous(b)) {
                const auto* adata = a.data();
                const auto* bdata = b.data();
                auto* cdata = out.data();
                for (size_t i = 0; i < out.size(); ++i) {
                    cdata[i] = adata[i] / bdata[i];
                }
            }
            else {
                for (size_t i = 0; i < out.row(); ++i) {
                    for (size_t j = 0; j < out.col(); ++j) {
                        out(i, j) = a(i, j) / b(i, j);
                    }
                }
            }
            return out;
        }

        if (detail::is_broadcastable(a, b)) {
            auto [ab, bb] = detail::broadcast(a, b);
            matrix<T> out = matrix<T>::empty_like(ab);
            for (size_t i = 0; i < out.row(); ++i) {
                for (size_t j = 0; j < out.col(); ++j) {
                    out(i, j) = ab(i, j) / bb(i, j);
                }
            }

            return out;
        }
        throw std::invalid_argument("matrix dimensions are incompatible for division");
    }

    template<class Derived, typename T>
    template<strided_matrix B>
    requires (std::same_as<T, typename B::value_type> && can_add_assign<T>)
    matrix_base<Derived, T>& matrix_base<Derived, T>::operator+=(const B& b) {
        if (row_ == b.row() && col_ == b.col()) {
            if (is_contiguous(derived()) && is_contiguous(b)) {
                const auto* bdata = b.data();
                for (size_t i = 0; i < size_; ++i) {
                    data_[i] += bdata[i];
                }
            }
            else {
                for (size_t i = 0; i < row_; ++i) {
                    for (size_t j = 0; j < col_; ++j) {
                        (*this)(i, j) += b(i, j);
                    }
                }
            }
            return *this;
        }

        if (detail::is_broadcastable(derived(), b)) {
            if (row_ != b.row() && b.row() != 1)
                throw std::invalid_argument("right-hand side cannot be broadcast to destination shape");
            if (col_ != b.col() && b.col() != 1)
                throw std::invalid_argument("right-hand side cannot be broadcast to destination shape");
            auto [_, b_view] = detail::broadcast(derived(), b);
            for (size_t i = 0; i < row_; ++i) {
                for (size_t j = 0; j < col_; ++j) {
                    (*this)(i, j) += b_view(i, j);
                }
            }
            return *this;
        }

        throw std::invalid_argument("matrix dimensions are incompatible for addition");
    }
}

#endif //NUMPP_MATRIX_ARITHMETIC_HPP