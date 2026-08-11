#ifndef NUMPP_MATRIX_ARITHMETIC_HPP
#define NUMPP_MATRIX_ARITHMETIC_HPP

#include <numpp/matrix/core.hpp>
#include <numpp/matrix/tool.hpp>
#include <numpp/matrix/matrix.hpp>
#include <numpp/matrix/view/broadcast.hpp>
#include <utility>
#include <string>

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
        throw std::invalid_argument(
            "numpp::operation<add> error: "
            "cannot add matrices with incompatible shapes: " +
            std::to_string(a.row()) + "x" +
            std::to_string(a.col()) + " and " +
            std::to_string(b.row()) + "x" +
            std::to_string(b.col())
        );
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
        throw std::invalid_argument(
            "numpp::operation<sub> error: "
            "cannot sub matrices with incompatible shapes: " +
            std::to_string(a.row()) + "x" +
            std::to_string(a.col()) + " and " +
            std::to_string(b.row()) + "x" +
            std::to_string(b.col())
        );
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
        throw std::invalid_argument(
            "numpp::operation<mul> error: "
            "cannot mul matrices with incompatible shapes: " +
            std::to_string(a.row()) + "x" +
            std::to_string(a.col()) + " and " +
            std::to_string(b.row()) + "x" +
            std::to_string(b.col())
        );
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
        throw std::invalid_argument(
            "numpp::operation<div> error: "
            "cannot div matrices with incompatible shapes: " +
            std::to_string(a.row()) + "x" +
            std::to_string(a.col()) + " and " +
            std::to_string(b.row()) + "x" +
            std::to_string(b.col())
        );
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
                throw std::invalid_argument(
                    "numpp::operation<add_assign> error: "
                    "cannot broadcast " +
                    std::to_string(b.row()) + "x" +
                    std::to_string(b.col()) +
                    " to destination shape " +
                    std::to_string(row_) + "x" +
                    std::to_string(col_) +
                    ": source rows (" +
                    std::to_string(b.row()) +
                    ") must match destination rows (" +
                    std::to_string(row_) +
                    ") or be 1"
                );

            if (col_ != b.col() && b.col() != 1)
                throw std::invalid_argument(
                    "numpp::operation<add_assign> error: "
                    "cannot broadcast " +
                    std::to_string(b.row()) + "x" +
                    std::to_string(b.col()) +
                    " to destination shape " +
                    std::to_string(row_) + "x" +
                    std::to_string(col_) +
                    ": source columns (" +
                    std::to_string(b.col()) +
                    ") must match destination columns (" +
                    std::to_string(col_) +
                    ") or be 1"
                );
            auto [_, b_view] = detail::broadcast(derived(), b);
            for (size_t i = 0; i < row_; ++i) {
                for (size_t j = 0; j < col_; ++j) {
                    (*this)(i, j) += b_view(i, j);
                }
            }
            return *this;
        }
        throw std::invalid_argument(
            "numpp::operation<add_assign> error: "
            "cannot add-assign matrices with incompatible shapes: " +
            std::to_string(row_) + "x" +
            std::to_string(col_) + " and " +
            std::to_string(b.row()) + "x" +
            std::to_string(b.col())
        );
    }

    template<class Derived, typename T>
    template<strided_matrix B>
    requires (std::same_as<T, typename B::value_type> && can_sub_assign<T>)
    matrix_base<Derived, T>& matrix_base<Derived, T>::operator-=(const B& b) {
        if (row_ == b.row() && col_ == b.col()) {
            if (is_contiguous(derived()) && is_contiguous(b)) {
                const auto* bdata = b.data();
                for (size_t i = 0; i < size_; ++i) {
                    data_[i] -= bdata[i];
                }
            }
            else {
                for (size_t i = 0; i < row_; ++i) {
                    for (size_t j = 0; j < col_; ++j) {
                        (*this)(i, j) -= b(i, j);
                    }
                }
            }
            return *this;
        }

        if (detail::is_broadcastable(derived(), b)) {
            if (row_ != b.row() && b.row() != 1)
                throw std::invalid_argument(
                    "numpp::operation<sub_assign> error: "
                    "cannot broadcast " +
                    std::to_string(b.row()) + "x" +
                    std::to_string(b.col()) +
                    " to destination shape " +
                    std::to_string(row_) + "x" +
                    std::to_string(col_) +
                    ": source rows (" +
                    std::to_string(b.row()) +
                    ") must match destination rows (" +
                    std::to_string(row_) +
                    ") or be 1"
                );

            if (col_ != b.col() && b.col() != 1)
                throw std::invalid_argument(
                    "numpp::operation<sub_assign> error: "
                    "cannot broadcast " +
                    std::to_string(b.row()) + "x" +
                    std::to_string(b.col()) +
                    " to destination shape " +
                    std::to_string(row_) + "x" +
                    std::to_string(col_) +
                    ": source columns (" +
                    std::to_string(b.col()) +
                    ") must match destination columns (" +
                    std::to_string(col_) +
                    ") or be 1"
                );
            auto [_, b_view] = detail::broadcast(derived(), b);
            for (size_t i = 0; i < row_; ++i) {
                for (size_t j = 0; j < col_; ++j) {
                    (*this)(i, j) -= b_view(i, j);
                }
            }
            return *this;
        }
        throw std::invalid_argument(
            "numpp::operation<sub_assign> error: "
            "cannot sub-assign matrices with incompatible shapes: " +
            std::to_string(row_) + "x" +
            std::to_string(col_) + " and " +
            std::to_string(b.row()) + "x" +
            std::to_string(b.col())
        );
    }

    template<class Derived, typename T>
    template<strided_matrix B>
    requires (std::same_as<T, typename B::value_type> && can_mul_assign<T>)
    matrix_base<Derived, T>& matrix_base<Derived, T>::operator*=(const B& b) {
        if (row_ == b.row() && col_ == b.col()) {
            if (is_contiguous(derived()) && is_contiguous(b)) {
                const auto* bdata = b.data();
                for (size_t i = 0; i < size_; ++i) {
                    data_[i] *= bdata[i];
                }
            }
            else {
                for (size_t i = 0; i < row_; ++i) {
                    for (size_t j = 0; j < col_; ++j) {
                        (*this)(i, j) *= b(i, j);
                    }
                }
            }
            return *this;
        }

        if (detail::is_broadcastable(derived(), b)) {
            if (row_ != b.row() && b.row() != 1)
                throw std::invalid_argument(
                    "numpp::operation<mul_assign> error: "
                    "cannot broadcast " +
                    std::to_string(b.row()) + "x" +
                    std::to_string(b.col()) +
                    " to destination shape " +
                    std::to_string(row_) + "x" +
                    std::to_string(col_) +
                    ": source rows (" +
                    std::to_string(b.row()) +
                    ") must match destination rows (" +
                    std::to_string(row_) +
                    ") or be 1"
                );

            if (col_ != b.col() && b.col() != 1)
                throw std::invalid_argument(
                    "numpp::operation<mul_assign> error: "
                    "cannot broadcast " +
                    std::to_string(b.row()) + "x" +
                    std::to_string(b.col()) +
                    " to destination shape " +
                    std::to_string(row_) + "x" +
                    std::to_string(col_) +
                    ": source columns (" +
                    std::to_string(b.col()) +
                    ") must match destination columns (" +
                    std::to_string(col_) +
                    ") or be 1"
                );
            auto [_, b_view] = detail::broadcast(derived(), b);
            for (size_t i = 0; i < row_; ++i) {
                for (size_t j = 0; j < col_; ++j) {
                    (*this)(i, j) *= b_view(i, j);
                }
            }
            return *this;
        }
        throw std::invalid_argument(
            "numpp::operation<mul_assign> error: "
            "cannot mul-assign matrices with incompatible shapes: " +
            std::to_string(row_) + "x" +
            std::to_string(col_) + " and " +
            std::to_string(b.row()) + "x" +
            std::to_string(b.col())
        );
    }

    template<class Derived, typename T>
    template<strided_matrix B>
    requires (std::same_as<T, typename B::value_type> && can_div_assign<T>)
    matrix_base<Derived, T>& matrix_base<Derived, T>::operator/=(const B& b) {
        if (row_ == b.row() && col_ == b.col()) {
            if (is_contiguous(derived()) && is_contiguous(b)) {
                const auto* bdata = b.data();
                for (size_t i = 0; i < size_; ++i) {
                    data_[i] /= bdata[i];
                }
            }
            else {
                for (size_t i = 0; i < row_; ++i) {
                    for (size_t j = 0; j < col_; ++j) {
                        (*this)(i, j) /= b(i, j);
                    }
                }
            }
            return *this;
        }

        if (detail::is_broadcastable(derived(), b)) {
            if (row_ != b.row() && b.row() != 1)
                throw std::invalid_argument(
                    "numpp::operation<div_assign> error: "
                    "cannot broadcast " +
                    std::to_string(b.row()) + "x" +
                    std::to_string(b.col()) +
                    " to destination shape " +
                    std::to_string(row_) + "x" +
                    std::to_string(col_) +
                    ": source rows (" +
                    std::to_string(b.row()) +
                    ") must match destination rows (" +
                    std::to_string(row_) +
                    ") or be 1"
                );

            if (col_ != b.col() && b.col() != 1)
                throw std::invalid_argument(
                    "numpp::operation<div_assign> error: "
                    "cannot broadcast " +
                    std::to_string(b.row()) + "x" +
                    std::to_string(b.col()) +
                    " to destination shape " +
                    std::to_string(row_) + "x" +
                    std::to_string(col_) +
                    ": source columns (" +
                    std::to_string(b.col()) +
                    ") must match destination columns (" +
                    std::to_string(col_) +
                    ") or be 1"
                );
            auto [_, b_view] = detail::broadcast(derived(), b);
            for (size_t i = 0; i < row_; ++i) {
                for (size_t j = 0; j < col_; ++j) {
                    (*this)(i, j) /= b_view(i, j);
                }
            }
            return *this;
        }
        throw std::invalid_argument(
            "numpp::operation<div_assign> error: "
            "cannot div-assign matrices with incompatible shapes: " +
            std::to_string(row_) + "x" +
            std::to_string(col_) + " and " +
            std::to_string(b.row()) + "x" +
            std::to_string(b.col())
        );
    }
}

#endif //NUMPP_MATRIX_ARITHMETIC_HPP