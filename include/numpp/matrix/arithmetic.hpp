#ifndef NUMPP_MATRIX_ARITHMETIC_HPP
#define NUMPP_MATRIX_ARITHMETIC_HPP

#include <numpp/matrix/core.hpp>
#include <numpp/matrix/tool.hpp>
#include <numpp/matrix/matrix.hpp>
#include <numpp/matrix/view/broadcast.hpp>
#include <utility>
#include <string>
#include <string_view>

namespace numpp {
    namespace detail {
        template<strided_matrix A, strided_matrix B, typename Op>
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
                    const auto* adata = a.data();
                    const auto* bdata = b.data();
                    auto* cdata = out.data();
                    for (size_t i = 0; i < out.size(); ++i)
                        cdata[i] = op(adata[i], bdata[i]);
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

        
        template<strided_matrix A, strided_matrix B, typename Op>
        requires std::same_as<
            typename A::value_type,
            typename B::value_type
        >
        A& matrix_op_assign_expr(
            A& a,
            const B& b,
            Op op,
            std::string_view operation_name
        ) {
            if (a.row() == b.row() && a.col() == b.col()) {
                if (is_contiguous(a) && is_contiguous(b)) {
                    auto* adata = a.data();
                    const auto* bdata = b.data();

                    for (size_t i = 0; i < a.size(); ++i)
                        op(adata[i], bdata[i]);
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

    template<matrix_derived A, strided_matrix B>
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

    template<matrix_derived A, strided_matrix B>
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
    
    template<matrix_derived A, strided_matrix B>
    requires (
        std::same_as<
            typename A::value_type,
            typename B::value_type
        > &&
        can_mul<typename A::value_type>
    )
    matrix<typename A::value_type> operator*(const A& a, const B& b) {
        return detail::matrix_op_expr(a,b,[](auto x, auto y) { return x * y; }, "mul");
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
        return detail::matrix_op_expr(a,b,[](auto x, auto y) { return x / y; }, "div");
    }
    
    template<strided_matrix A, matrix_derived B>
    requires (
        !matrix_derived<A> &&
        std::same_as<
            typename A::value_type,
            typename B::value_type
        > &&
        can_add<typename A::value_type>
    )
    matrix<typename A::value_type> operator+(const A& a, const B& b) {
        return detail::matrix_op_expr(a,b,[](auto x, auto y) { return x + y; }, "add");
    }

    template<strided_matrix A, matrix_derived B>
    requires (
        !matrix_derived<A> &&
        std::same_as<
            typename A::value_type,
            typename B::value_type
        > &&
        can_sub<typename A::value_type>
    )
    matrix<typename A::value_type> operator-(const A& a, const B& b) {
        return detail::matrix_op_expr(a,b,[](auto x, auto y) { return x - y; }, "sub");
    }
    
    template<strided_matrix A, matrix_derived B>
    requires (
        !matrix_derived<A> &&
        std::same_as<
            typename A::value_type,
            typename B::value_type
        > &&
        can_mul<typename A::value_type>
    )
    matrix<typename A::value_type> operator*(const A& a, const B& b) {
        return detail::matrix_op_expr(a,b,[](auto x, auto y) { return x * y; }, "mul");
    }
    
    template<strided_matrix A, matrix_derived B>
    requires (
        !matrix_derived<A> &&
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
    template<strided_matrix B>
    requires (std::same_as<T, typename B::value_type> && can_add_assign<T>)
    matrix_base<Derived, T>& matrix_base<Derived, T>::operator+=(const B& b) {
        detail::matrix_op_assign_expr(derived(), b, [](auto& x, const auto& y) { x += y; }, "add-assign");
        return *this;
    }

    template<class Derived, typename T>
    template<strided_matrix B>
    requires (std::same_as<T, typename B::value_type> && can_sub_assign<T>)
    matrix_base<Derived, T>& matrix_base<Derived, T>::operator-=(const B& b) {
        detail::matrix_op_assign_expr(derived(), b, [](auto& x, const auto& y) { x -= y; }, "sub-assign");
        return *this;
    }

    template<class Derived, typename T>
    template<strided_matrix B>
    requires (std::same_as<T, typename B::value_type> && can_mul_assign<T>)
    matrix_base<Derived, T>& matrix_base<Derived, T>::operator*=(const B& b) {
        detail::matrix_op_assign_expr(derived(), b, [](auto& x, const auto& y) { x *= y; }, "mul-assign");
        return *this;
    }

    template<class Derived, typename T>
    template<strided_matrix B>
    requires (std::same_as<T, typename B::value_type> && can_div_assign<T>)
    matrix_base<Derived, T>& matrix_base<Derived, T>::operator/=(const B& b) {
        detail::matrix_op_assign_expr(derived(), b, [](auto& x, const auto& y) { x /= y; }, "div-assign");
        return *this;
    }
}

#endif //NUMPP_MATRIX_ARITHMETIC_HPP