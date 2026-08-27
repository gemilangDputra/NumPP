#ifndef NUMPP_MATRIX_OP_EXPR_HPP
#define NUMPP_MATRIX_OP_EXPR_HPP

#include <numpp/matrix/core.hpp>
#include <numpp/matrix/tool.hpp>
#include <numpp/matrix/matrix.hpp>
#include <numpp/matrix/view/broadcast.hpp>
#include <utility>
#include <string>
#include <string_view>

namespace numpp {
    namespace detail {
        template<matrix_like A, matrix_like B, typename Op>
        requires std::same_as<
            typename A::value_type,
            typename B::value_type
        >
        matrix<typename A::value_type>
        matrix_binary_expr(const A& a, const B& b, Op op, std::string_view operation_name) {
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
        A& matrix_binary_assign_expr(A& a, const B& b, Op op, std::string_view operation_name) {
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
        
        template<numpp_matrix EXPR, typename Op>
        matrix<typename EXPR::value_type>
        matrix_scalar_expr(
            const EXPR& a, const typename EXPR::value_type& scalar, Op op) {
            using T = typename EXPR::value_type;
            matrix<T> out = matrix<T>::empty_like(a);
            if (is_contiguous(a)) {
                const auto* adata = a.data() + a.offset();
                auto* cdata = out.data();
                for (size_t i = 0; i < out.size(); ++i)
                    cdata[i] = op(adata[i], scalar);
            }
            else {
                for (size_t i = 0; i < a.row(); ++i) {
                    for (size_t j = 0; j < a.col(); ++j)
                        out(i, j) = op(a(i, j), scalar);
                }
            }
            return out;
        }

        template<numpp_matrix EXPR, typename Op>
        EXPR& matrix_scalar_assign_expr( EXPR& mat, const typename EXPR::value_type& scalar, Op op) {
            if (is_contiguous(mat)) {
                auto* adata = mat.data() + mat.offset();
                for (size_t i = 0; i < mat.size(); ++i) {
                    op(adata[i], scalar);
                }
            }
            else {
                for (size_t i = 0; i < mat.row(); ++i) {
                    for (size_t j = 0; j < mat.col(); ++j) {
                        op(mat(i, j), scalar);
                    }
                }
            }
            return mat;
        }

        template<matrix_like EXPR, typename Op>
        matrix<typename EXPR::value_type> matrix_unary_expr(const EXPR& mat, Op op) {
            using T = typename EXPR::value_type;
            matrix<T> out = matrix<T>::empty_like(mat);
            if (is_contiguous(mat)) {
                const auto adata = mat.data() + mat.offset();
                auto cdata = out.data();
                for (size_t i = 0; i < out.size(); ++i) {
                    cdata[i] = op(adata[i]);
                }
            } else {
                for (size_t i = 0; i < out.row(); ++i) {
                    for (size_t j = 0; j < out.col(); ++j)
                        out(i, j) = op(mat(i,j));
                }
            }
            return out;
        }
        
        template<matrix_like EXPR, typename Op>
        auto matrix_reduction_expr(const EXPR& mat, Op op, std::string_view operation_name) {
            using T = typename EXPR::value_type;
            if (mat.size() == 0) {
                throw std::invalid_argument(
                    "numpp::operation<" +
                    std::string(operation_name) +
                    "> error: cannot reduce an empty matrix"
                );
            }
            T out = mat(0, 0);
            if (is_contiguous(mat)) {
                const auto data = mat.data() + mat.offset();
                for (size_t i = 1; i < mat.size(); ++i) {
                    op(out, data[i]);
                }
            } else {
                for (size_t i = 0; i < mat.row(); ++i) {
                    for (size_t j = 0; j < mat.col(); ++j) {
                        if (i == 0 && j == 0) continue;
                        op(out, mat(i, j));
                    }
                }
            }
            return out;
        }
    }
}

#endif //NUMPP_MATRIX_OP_EXPR_HPP