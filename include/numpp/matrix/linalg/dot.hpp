#ifndef NUMPP_MATRIX_DOT_HPP
#define NUMPP_MATRIX_DOT_HPP

#include <numpp/matrix/tool.hpp>
#include <numpp/matrix/matrix.hpp>
#include <numpp/matrix/core.hpp>
#include <numpp/matrix/linalg/matmul.hpp>
#include <numpp/vector/tool.hpp>
#include <numpp/vector/vector.hpp>
#include <numpp/vector/core.hpp>
#include <numpp/backend/blas.hpp>
#include <string>

namespace numpp {
    namespace linalg {
        namespace detail {
            #if NUMPP_USE_BLAS
            template<class A, class B>
            auto dot_blas_mv(const A& a, const B& b) {
                using T = typename A::value_type;
                if constexpr (matrix_like<A>) {
                    const bool col_major = a.rowstride() == 1;
                    const CBLAS_LAYOUT layout = col_major ? CblasColMajor : CblasRowMajor;
                    const int lda = col_major ? a.colstride() : a.rowstride();
                    vector<T> out = vector<T>::empty(a.row());
                    const T* adata = a.data() + a.offset();
                    const T* bdata = b.data();
                    int incB = 1;
                    if constexpr (vector_like<B>) {incB = static_cast<int>(b.stride()); bdata = bdata + b.offset(); }
                    ::numpp::detail::blas_gemv<T>(
                        layout,
                        CblasNoTrans,
                        a.row(),
                        a.col(),
                        T{1},
                        adata,
                        lda,
                        bdata,
                        incB,
                        T{0},
                        out.data(),
                        1
                    );
                    return out;
                } else {
                    const bool col_major = b.rowstride() == 1;
                    const CBLAS_LAYOUT layout = col_major ? CblasColMajor : CblasRowMajor;
                    const int lda = col_major ? b.colstride() : b.rowstride();
                    vector<T> out = vector<T>::empty(b.col());
                    const T* adata = a.data();
                    const T* bdata = b.data() + b.offset();
                    int incA = 1;
                    if constexpr (vector_like<A>) { incA = static_cast<int>(a.stride()); adata = adata + a.offset(); }
                    ::numpp::detail::blas_gemv<T>(
                        layout,
                        CblasTrans,
                        b.row(),
                        b.col(),
                        T{1},
                        bdata,
                        lda,
                        adata,
                        incA,
                        T{0},
                        out.data(),
                        1
                    );
                    return out;
                }
            }
            #endif
            
            template<class A, class B>
            auto dot_native_mv(const A& a, const B& b) {
                using T = typename A::value_type;
                if constexpr (matrix_like<A>) {
                    vector<T> out = vector<T>::empty(a.row());
                    for (size_t i = 0; i < a.row(); ++i) {
                        T sum = T{0};
                        for (size_t j = 0; j < a.col(); ++j) {
                            sum += a(i,j) * b[j];
                        }
                        out[i] = sum;
                    }
                    return out;
                } else {
                    vector<T> out = vector<T>::empty(b.col());
                    auto bt = b.transpose();
                    for (size_t i = 0; i < bt.row(); ++i) {
                        T sum = T{0};
                        for (size_t j = 0; j < bt.col(); ++j) {
                            sum += a[j] * bt(i,j);
                        }
                        out[i] = sum;
                    }
                    return out;
                }
            }
        }

        template<class A, class B>
        requires (
            (matrix_like<A> || matrix_like<B>) &&
            (general_vector_like<A> || general_vector_like<B>) &&
            std::same_as<
                typename A::value_type,
                typename B::value_type
            > &&
            is_numeric<typename A::value_type>)
        auto dot(const A& a, const B& b) {
            if constexpr (matrix_like<A>) {
                if (a.col() != b.size())
                    throw std::invalid_argument(
                        "numpp::operation<dot> error: "
                        "dimension mismatch: "
                        "lhs has shape (" + std::to_string(a.row()) + "x" +
                        std::to_string(a.col()) + "), "
                        "rhs has size " + std::to_string(b.size()) +
                        "; expected matrix columns to match vector size"
                    );
            } else {
                if (a.size() != b.row())
                    throw std::invalid_argument(
                        "numpp::operation<dot> error: "
                        "dimension mismatch: "
                        "lhs has size " + std::to_string(a.size()) +
                        ", rhs has shape (" + std::to_string(b.row()) + "x" +
                        std::to_string(b.col()) +
                        "); expected vector size to match matrix rows"
                    );
            }

            #if NUMPP_USE_BLAS
            if constexpr (matrix_like<A>) {
                if (a.size() > ::numpp::threshold::dot_blas) return detail::dot_blas_mv(a,b);
            } else {
                if (b.size() > ::numpp::threshold::dot_blas) return detail::dot_blas_mv(a,b);
            }
            #endif
            return detail::dot_native_mv(a,b);
        }
        
        template<matrix_like A, matrix_like B>
        requires (std::same_as<
            typename A::value_type,
            typename B::value_type> &&
            is_numeric<typename A::value_type>
        )
        auto dot(const A& a, const B& b) {
            return ::numpp::linalg::matmul(a, b);
        }
    }

    template<class A, class B>
    requires (
        (matrix_like<A> || matrix_like<B>) &&
        (general_vector_like<A> || general_vector_like<B>) &&
        std::same_as<
            typename A::value_type,
            typename B::value_type
        > &&
        is_numeric<typename A::value_type>)
    auto dot(const A& a, const B& b) {
        return linalg::dot(a,b);
    }
    
    template<matrix_like A, matrix_like B>
    requires (std::same_as<
        typename A::value_type,
        typename B::value_type> &&
        is_numeric<typename A::value_type>
    )
    auto dot(const A& a, const B& b) {
        return linalg::matmul(a,b);
    }
}

#endif //NUMPP_MATRIX_DOT_HPP