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
            template<strided_matrix A, strided_matrix B>
            auto matmul_blas_same_layout(const A& a, const B& b) {
                using T = typename A::value_type;
                matrix<T> out = matrix<T>::empty(a.row(), b.col());

                const bool trans_a = is_transpose(a);
                const bool trans_b = is_transpose(b);

                if (a.order() == layout::rowmajor) {
                    ::numpp::detail::blas_gemm<T>(
                        CblasRowMajor,
                        trans_a ? CblasTrans : CblasNoTrans,
                        trans_b ? CblasTrans : CblasNoTrans,

                        static_cast<int>(a.row()),
                        static_cast<int>(b.col()),
                        static_cast<int>(a.col()),

                        T{1},

                        a.data(),
                        ::numpp::detail::blas_leading_dimension(a, CblasRowMajor, trans_a ? CblasTrans : CblasNoTrans),
                        b.data(),

                        ::numpp::detail::blas_leading_dimension(b, CblasRowMajor, trans_b ? CblasTrans : CblasNoTrans),
                        T{0},
                        out.data(),
                        static_cast<int>(out.rowstride())
                    );
                } else {
                    const auto trans_bt = trans_b ? CblasNoTrans : CblasTrans;
                    const auto trans_at = trans_a ? CblasNoTrans : CblasTrans;

                    ::numpp::detail::blas_gemm<T>(
                        CblasColMajor,

                        trans_bt,
                        trans_at,

                        static_cast<int>(b.col()),
                        static_cast<int>(a.row()),
                        static_cast<int>(a.col()),

                        T{1},

                        b.data(),
                        ::numpp::detail::blas_leading_dimension(b, CblasColMajor, trans_bt),

                        a.data(),
                        ::numpp::detail::blas_leading_dimension(a, CblasColMajor, trans_at),

                        T{0},

                        out.data(),
                        static_cast<int>(out.rowstride())
                    );
                }

                return out;
            }

            template<strided_matrix A, strided_matrix B>
            auto matmul_blas_different_layout(const A& a, const B& b) {
                using T = typename A::value_type;
                matrix<T> out = matrix<T>::empty(a.row(), b.col());

                const bool trans_a = is_transpose(a);
                const bool trans_b = is_transpose(b);

                const int M = static_cast<int>(a.row());
                const int N = static_cast<int>(b.col());
                const int K = static_cast<int>(a.col());

                if (a.order() == layout::rowmajor) {
                    if (!trans_a && !trans_b) {

                        ::numpp::detail::blas_gemm<T>(
                            CblasRowMajor,

                            CblasNoTrans,
                            CblasTrans,

                            M,
                            N,
                            K,

                            T{1},

                            a.data(),
                            ::numpp::detail::blas_leading_dimension(
                                a,
                                CblasRowMajor,
                                CblasNoTrans
                            ),

                            b.data(),
                            ::numpp::detail::blas_leading_dimension(
                                b,
                                CblasRowMajor,
                                CblasTrans
                            ),

                            T{0},

                            out.data(),
                            static_cast<int>(out.rowstride())
                        );
                    }

                    else if (!trans_a && trans_b) {

                        ::numpp::detail::blas_gemm<T>(
                            CblasRowMajor,

                            CblasNoTrans,
                            CblasNoTrans,

                            M,
                            N,
                            K,

                            T{1},

                            a.data(),
                            ::numpp::detail::blas_leading_dimension(
                                a,
                                CblasRowMajor,
                                CblasNoTrans
                            ),

                            b.data(),
                            ::numpp::detail::blas_leading_dimension(
                                b,
                                CblasRowMajor,
                                CblasNoTrans
                            ),

                            T{0},

                            out.data(),
                            static_cast<int>(out.rowstride())
                        );
                    }

                    else if (trans_a && !trans_b) {
                        ::numpp::detail::blas_gemm<T>(
                            CblasRowMajor,

                            CblasTrans,
                            CblasTrans,

                            M,
                            N,
                            K,

                            T{1},

                            a.data(),
                            ::numpp::detail::blas_leading_dimension(
                                a,
                                CblasRowMajor,
                                CblasTrans
                            ),

                            b.data(),
                            ::numpp::detail::blas_leading_dimension(
                                b,
                                CblasRowMajor,
                                CblasTrans
                            ),

                            T{0},

                            out.data(),
                            static_cast<int>(out.rowstride())
                        );
                    }

                    else {
                        ::numpp::detail::blas_gemm<T>(
                            CblasRowMajor,

                            CblasTrans,
                            CblasNoTrans,

                            M,
                            N,
                            K,

                            T{1},

                            a.data(),
                            ::numpp::detail::blas_leading_dimension(
                                a,
                                CblasRowMajor,
                                CblasTrans
                            ),

                            b.data(),
                            ::numpp::detail::blas_leading_dimension(
                                b,
                                CblasRowMajor,
                                CblasNoTrans
                            ),

                            T{0},

                            out.data(),
                            static_cast<int>(out.rowstride())
                        );
                    }
                }

                else {
                    if (!trans_a && !trans_b) {

                        ::numpp::detail::blas_gemm<T>(
                            CblasColMajor,

                            CblasNoTrans,
                            CblasTrans,

                            N,
                            M,
                            K,

                            T{1},

                            b.data(),
                            ::numpp::detail::blas_leading_dimension(
                                b,
                                CblasColMajor,
                                CblasNoTrans
                            ),

                            a.data(),
                            ::numpp::detail::blas_leading_dimension(
                                a,
                                CblasColMajor,
                                CblasTrans
                            ),

                            T{0},

                            out.data(),
                            static_cast<int>(out.rowstride())
                        );
                    }
                    else if (!trans_a && trans_b) {
                        ::numpp::detail::blas_gemm<T>(
                            CblasColMajor,

                            CblasTrans,
                            CblasTrans,

                            N,
                            M,
                            K,

                            T{1},

                            b.data(),
                            ::numpp::detail::blas_leading_dimension(
                                b,
                                CblasColMajor,
                                CblasTrans
                            ),

                            a.data(),
                            ::numpp::detail::blas_leading_dimension(
                                a,
                                CblasColMajor,
                                CblasTrans
                            ),

                            T{0},

                            out.data(),
                            static_cast<int>(out.rowstride())
                        );
                    }
                    else if (trans_a && !trans_b) {
                        ::numpp::detail::blas_gemm<T>(
                            CblasColMajor,

                            CblasNoTrans,
                            CblasNoTrans,

                            N,
                            M,
                            K,

                            T{1},

                            b.data(),
                            ::numpp::detail::blas_leading_dimension(
                                b,
                                CblasColMajor,
                                CblasNoTrans
                            ),

                            a.data(),
                            ::numpp::detail::blas_leading_dimension(
                                a,
                                CblasColMajor,
                                CblasNoTrans
                            ),

                            T{0},

                            out.data(),
                            static_cast<int>(out.rowstride())
                        );
                    }

                    else {
                        ::numpp::detail::blas_gemm<T>(
                            CblasColMajor,

                            CblasTrans,
                            CblasNoTrans,

                            N,
                            M,
                            K,

                            T{1},

                            b.data(),
                            ::numpp::detail::blas_leading_dimension(
                                b,
                                CblasColMajor,
                                CblasTrans
                            ),

                            a.data(),
                            ::numpp::detail::blas_leading_dimension(
                                a,
                                CblasColMajor,
                                CblasNoTrans
                            ),

                            T{0},

                            out.data(),
                            static_cast<int>(out.rowstride())
                        );
                    }
                }

                return out;
            }
            
            #endif

            template<strided_matrix A, strided_matrix B>
            auto matmul_native_contiguous_notrans_samelayout(const A& a, const B& b) {
                using T = typename A::value_type;
                matrix<T> out = matrix<T>::zeros(a.row(), b.col());

                const T* adata = a.data();
                const T* bdata = b.data();
                T* cdata = out.data();

                const size_t M = a.row();
                const size_t N = b.col();
                const size_t K = a.col();

                if (a.order() == layout::rowmajor) {
                    for (size_t i = 0; i < M; ++i) {
                        for (size_t k = 0; k < K; ++k) {
                            const T aik = adata[i * K + k];

                            for (size_t j = 0; j < N; ++j)
                                cdata[i * N + j] +=
                                    aik * bdata[k * N + j];
                        }
                    }
                }
                else {
                    for (size_t j = 0; j < N; ++j) {
                        for (size_t k = 0; k < K; ++k) {
                            const T bkj = bdata[k + j * K];

                            for (size_t i = 0; i < M; ++i)
                                cdata[i * N + j] +=
                                    adata[i + k * M] * bkj;
                        }
                    }
                }

                return out;
            }

            template<strided_matrix A, strided_matrix B>
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
        
        template<strided_matrix A, strided_matrix B>
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

            if (is_contiguous(a) && is_contiguous(b)) {
                #if NUMPP_USE_BLAS
                if constexpr (std::same_as<T, float> || std::same_as<T, double>) {
                    const size_t work = M * N * K;

                    if (work > 5'250'000) {
                        if (a.order() == b.order())
                            return detail::matmul_blas_same_layout(a, b);
                        else
                            return detail::matmul_blas_different_layout(a, b);
                    }
                }
                #endif

                if (a.order() == b.order() && !is_transpose(a) && !is_transpose(b)) {
                    return detail::matmul_native_contiguous_notrans_samelayout(a, b);
                }
            }

            return detail::matmul_native_general(a, b);
        }
    }

    template<strided_matrix A, strided_matrix B>
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