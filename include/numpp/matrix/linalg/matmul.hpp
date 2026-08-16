#ifndef NUMPP_MATRIX_LINALG_MATMUL_HPP
#define NUMPP_MATRIX_LINALG_MATMUL_HPP

#include <numpp/matrix/tool.hpp>
#include <numpp/matrix/matrix.hpp>
#include <numpp/matrix/core.hpp>

#include <numpp/backend/blas.hpp>
#include <type_traits>
#include <string>

namespace numpp {
    namespace linalg {
        namespace detail {
            #if NUMPP_USE_BLAS
            template<matrix_like A, matrix_like B>
            auto matmul_blas_same_strides(const A& a, const B& b) {
                using T = typename A::value_type;
                matrix<T> out = matrix<T>::empty(a.row(), b.col());
                
                const int M = static_cast<int>(a.row());
                const int N = static_cast<int>(b.col());
                const int K = static_cast<int>(a.col());

                if (a.rowstride() == a.col() &&
                    a.colstride() == 1) {
                    ::numpp::detail::blas_gemm<T>(
                        CblasRowMajor,
                        CblasNoTrans,
                        CblasNoTrans,
                        M, N, K,
                        T{1},
                        a.data() + a.offset(),
                        static_cast<int>(a.rowstride()),
                        b.data() + b.offset(),
                        static_cast<int>(b.rowstride()),
                        T{0},
                        out.data(),
                        static_cast<int>(out.rowstride())
                    );

                } else {
                    ::numpp::detail::blas_gemm<T>(
                        CblasColMajor,
                        CblasTrans,
                        CblasTrans,
                        N, M, K,
                        T{1},
                        b.data() + b.offset(),
                        static_cast<int>(b.colstride()),
                        a.data() + a.offset(),
                        static_cast<int>(a.colstride()),
                        T{0},
                        out.data(),
                        static_cast<int>(out.rowstride())
                    );
                }
                return out;
            }

template<matrix_like A, matrix_like B>
auto matmul_blas_different_strides(const A& a, const B& b) {
    using T = typename A::value_type;
    matrix<T> out = matrix<T>::empty(a.row(), b.col());

    const int M = static_cast<int>(a.row());
    const int N = static_cast<int>(b.col());
    const int K = static_cast<int>(a.col());

    if (a.rowstride() == a.col() && a.colstride() == 1) {
        ::numpp::detail::blas_gemm<T>(
            CblasRowMajor,
            CblasNoTrans,
            CblasTrans,
            M, N, K,
            T{1},
            a.data() + a.offset(),
            static_cast<int>(a.rowstride()),
            b.data() + b.offset(),
            static_cast<int>(b.colstride()),
            T{0},
            out.data(),
            static_cast<int>(out.rowstride())
        );
    }
    else {
        ::numpp::detail::blas_gemm<T>(
        CblasColMajor,
        CblasNoTrans,
        CblasTrans,
        N, M, K,
        T{1},
        b.data() + b.offset(),
        static_cast<int>(b.rowstride()),
        a.data() + a.offset(),
        static_cast<int>(a.colstride()),
        T{0},
        out.data(),
        static_cast<int>(out.rowstride())
        );
    }
    return out;
}

            #endif

            template<matrix_like A, matrix_like B>
            auto matmul_native_contiguous_same_strides(const A& a, const B& b) {
                using T = typename A::value_type;
                matrix<T> out = matrix<T>::zeros(a.row(), b.col());

                const T* adata = a.data()+a.offset();
                const T* bdata = b.data()+b.offset();
                T* cdata = out.data();

                const size_t M = a.row();
                const size_t N = b.col();
                const size_t K = a.col();

                if (a.rowstride() == a.col() && a.colstride() == 1) {
                    for (size_t i = 0; i < M; ++i) {
                        for (size_t k = 0; k < K; ++k) {
                            const T aik = adata[i * K + k];
                            for (size_t j = 0; j < N; ++j)
                                cdata[i * N + j] += aik * bdata[k * N + j];
                        }
                    }
                }
                else {
                    for (size_t j = 0; j < N; ++j) {
                        for (size_t k = 0; k < K; ++k) {
                            const T bkj = bdata[k + j * K];
                            for (size_t i = 0; i < M; ++i)
                                cdata[i * N + j] += adata[i + k * M] * bkj;
                        }
                    }
                }
                return out;
            }

            template<matrix_like A, matrix_like B>
            auto matmul_native_general(const A& a, const B& b) {
                using T = typename A::value_type;
                matrix<T> out = matrix<T>::empty(a.row(), b.col());
                for (size_t i = 0; i < a.row(); ++i) {
                    for (size_t j = 0; j < b.col(); ++j) {
                        T sum{};
                        for (size_t k = 0; k < a.col(); ++k)
                            sum += a(i, k) * b(k, j);
                        out(i, j) = sum;
                    }
                }
                return out;
            }
        }
        
        template<matrix_like A, matrix_like B>
        requires (std::same_as<
            typename A::value_type,
            typename B::value_type> &&
            is_numeric<typename A::value_type>
        )
        auto matmul(const A& a, const B& b) {
            using T = typename A::value_type;
            if (a.col() != b.row())
                throw std::invalid_argument(
                    "numpp::operation<matmul> error: "
                    "cannot multiply " +
                    std::to_string(a.row()) + "x" +
                    std::to_string(a.col()) +
                    " by " +
                    std::to_string(b.row()) + "x" +
                    std::to_string(b.col()) +
                    ": left columns (" +
                    std::to_string(a.col()) +
                    ") must match right rows (" +
                    std::to_string(b.row()) +
                    ")"
                );

            const size_t M = a.row();
            const size_t N = b.col();
            const size_t K = a.col();

            const size_t work = M * N * K;

            if (is_contiguous(a) && is_contiguous(b)) {
                #if NUMPP_USE_BLAS
                if constexpr (std::same_as<T, float> || std::same_as<T, double>) {
                    if (work > 125'000'000ULL/(8*sizeof(T))) {
                        if (a.rowstride() == b.rowstride() || a.colstride() == b.colstride())
                            return detail::matmul_blas_same_strides(a,b);
                        else
                            return detail::matmul_blas_different_strides(a,b);
                    }
                }
                #endif
                if (a.rowstride() == b.rowstride() && a.colstride() == b.colstride()) {
                    return detail::matmul_native_contiguous_same_strides(a, b);
                }
            }

            if (work > 12'500'000ULL) {
                if (!is_contiguous(a) && !is_contiguous(b)) {
                    return ::numpp::linalg::matmul(matrix<T>(a), matrix<T>(b));
                } else if (!is_contiguous(a)) {
                    return ::numpp::linalg::matmul(matrix<T>(a), b);
                } else {
                    return ::numpp::linalg::matmul(a, matrix<T>(b));
                }
            }
            return detail::matmul_native_general(a, b);
        }
    }
    
    template<matrix_like A, matrix_like B>
    requires (std::same_as<
        typename A::value_type,
        typename B::value_type> &&
        is_numeric<typename A::value_type>
    )
    auto matmul(const A& a, const B& b) {
        return linalg::matmul(a,b);
    }
}

#endif //NUMPP_MATRIX_LINALG_MATMUL_HPP