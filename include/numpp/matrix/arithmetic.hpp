#ifndef NUMPP_MATRIX_ARITHMETIC_HPP
#define NUMPP_MATRIX_ARITHMETIC_HPP

#include <numpp/matrix/core.hpp>
#include <numpp/matrix/tool.hpp>
#include <numpp/matrix/matrix.hpp>
#include <numpp/matrix/view/broadcast.hpp>
#include <numpp/matrix/linalg/matmul.hpp>
#include <utility>
#include <string>
#include <string_view>
#include <iostream>

namespace numpp {
    namespace detail {
        template<matrix_like A, matrix_like B, typename Op>
        requires std::same_as<
            typename A::value_type,
            typename B::value_type
        >
        matrix<typename A::value_type>
        matrix_op_expr(const A& a, const B& b, Op op, std::string_view operation_name) {
            using T = typename A::value_type;
            if (a.row() == b.row() && a.col() == b.col()) {
                matrix<T> out = matrix<T>::empty_like(a);
                if (is_contiguous(a) && is_contiguous(b)) {
                    if (a.rowstride() == b.rowstride() && a.colstride() == b.colstride()) {
                        const auto* adata = a.data() + a.offset();
                        const auto* bdata = b.data() + b.offset();
                        auto* cdata = out.data();
                        for (size_t i = 0; i < out.size(); ++i)
                            cdata[i] = op(adata[i], bdata[i]);
                    }
                    else {
                        const auto* adata = a.data() + a.offset();
                        const auto* bdata = b.data() + b.offset();
                        auto* cdata = out.data();
                        const size_t rows = out.row();
                        const size_t cols = out.col();
                        if (a.colstride() == 1) {
                            for (size_t i = 0; i < rows; ++i) {
                                const auto* ap = adata + i * a.rowstride();
                                const auto* bp = bdata + i * b.rowstride();
                                auto* cp = cdata + i * out.rowstride();
                                for (size_t j = 0; j < cols; ++j) {
                                    *cp++ = op(*ap++, *bp);
                                    bp += b.colstride();
                                }
                            }
                        }
                        else {
                            for (size_t j = 0; j < cols; ++j) {
                                const auto* ap = adata + j * a.colstride();
                                const auto* bp = bdata + j * b.colstride();
                                auto* cp = cdata + j * out.colstride();
                                for (size_t i = 0; i < rows; ++i) {
                                    *cp++ = op(*ap++, *bp);
                                    bp += b.rowstride();
                                }
                            }
                        }
                    }
                }
                else {
                    for (size_t i = 0; i < out.row(); ++i) {
                        for (size_t j = 0; j < out.col(); ++j)
                            out(i, j) = op(a(i, j), b(i, j));
                    }
                }
                return out;
            }

            if (is_broadcastable(a, b)) {
                auto [ab, bb] = broadcast(a, b);
                matrix<T> out = matrix<T>::empty_like(ab);
                for (size_t i = 0; i < out.row(); ++i) {
                    for (size_t j = 0; j < out.col(); ++j) 
                        out(i, j) = op(ab(i, j), bb(i, j));
                }
                return out;
            }

            throw std::invalid_argument(
                "numpp::operation<" +
                std::string(operation_name) +
                "> error: cannot " +
                std::string(operation_name) +
                " matrices with incompatible shapes: " +
                std::to_string(a.row()) + "x" +
                std::to_string(a.col()) + " and " +
                std::to_string(b.row()) + "x" +
                std::to_string(b.col())
            );
        }

        
        template<matrix_like A, matrix_like B, typename Op>
        requires std::same_as<
            typename A::value_type,
            typename B::value_type
        >
        A& matrix_op_assign_expr(A& a, const B& b, Op op, std::string_view operation_name) {
            if (a.row() == b.row() && a.col() == b.col()) {
                if (is_contiguous(a) && is_contiguous(b)) {
                    if (a.rowstride() == b.rowstride() && a.colstride() == b.colstride()) {
                        auto* adata = a.data() + a.offset();
                        const auto* bdata = b.data() + b.offset();
                        for (size_t i = 0; i < a.size(); ++i)
                            op(adata[i], bdata[i]);
                    } else {
                        auto* adata = a.data() + a.offset();
                        const auto* bdata = b.data() + b.offset();
                        const size_t rows = a.row();
                        const size_t cols = a.col();
                        if (a.colstride() == 1) {
                            for (size_t i = 0; i < rows; ++i) {
                                auto* ap = adata + i * a.rowstride();
                                const auto* bp = bdata + i * b.rowstride();

                                for (size_t j = 0; j < cols; ++j) {
                                    op(*ap++, *bp);
                                    bp += b.colstride();
                                }
                            }
                        }
                        else {
                            for (size_t j = 0; j < cols; ++j) {
                                auto* ap = adata + j * a.colstride();
                                const auto* bp = bdata + j * b.colstride();
                                for (size_t i = 0; i < rows; ++i) {
                                    op(*ap++, *bp);
                                    bp += b.rowstride();
                                }
                            }
                        }
                    }
                }
                else {
                    for (size_t i = 0; i < a.row(); ++i) {
                        for (size_t j = 0; j < a.col(); ++j)
                            op(a(i, j), b(i, j));
                    }
                }

                return a;
            }
            if (is_broadcastable(a, b)) {
                if (a.row() != b.row() && b.row() != 1) {
                    throw std::invalid_argument(
                        "numpp::operation<" +
                        std::string(operation_name) +
                        "> error: cannot broadcast " +
                        std::to_string(b.row()) + "x" +
                        std::to_string(b.col()) +
                        " to destination shape " +
                        std::to_string(a.row()) + "x" +
                        std::to_string(a.col()) +
                        ": source rows (" +
                        std::to_string(b.row()) +
                        ") must match destination rows (" +
                        std::to_string(a.row()) +
                        ") or be 1"
                    );
                }

                if (a.col() != b.col() && b.col() != 1) {
                    throw std::invalid_argument(
                        "numpp::operation<" +
                        std::string(operation_name) +
                        "> error: cannot broadcast " +
                        std::to_string(b.row()) + "x" +
                        std::to_string(b.col()) +
                        " to destination shape " +
                        std::to_string(a.row()) + "x" +
                        std::to_string(a.col()) +
                        ": source columns (" +
                        std::to_string(b.col()) +
                        ") must match destination columns (" +
                        std::to_string(a.col()) +
                        ") or be 1"
                    );
                }

                auto [_, b_view] = broadcast(a, b);
                for (size_t i = 0; i < a.row(); ++i) {
                    for (size_t j = 0; j < a.col(); ++j)
                        op(a(i, j), b_view(i, j));
                }
                return a;
            }
            throw std::invalid_argument(
                "numpp::operation<" +
                std::string(operation_name) +
                "> error: cannot " +
                std::string(operation_name) +
                " matrices with incompatible shapes: " +
                std::to_string(a.row()) + "x" +
                std::to_string(a.col()) +
                " and " +
                std::to_string(b.row()) + "x" +
                std::to_string(b.col())
            );
        }
    }

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
    requires (
        std::same_as<
            typename matrix_base<Derived, T>::value_type,
            typename B::value_type
        >
        && can_mul_assign<T>
    )
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