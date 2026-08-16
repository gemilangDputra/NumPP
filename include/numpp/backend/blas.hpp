#ifndef NUMPP_BACKEND_BLAS_HPP
#define NUMPP_BACKEND_BLAS_HPP

#include <numpp/numpp_blas_config.hpp>

#if NUMPP_USE_BLAS
#include NUMPP_BLAS_HEADER

#include <numpp/matrix/tool.hpp>

namespace numpp::detail {
    template<typename T>
    void blas_gemm(
        CBLAS_ORDER order,
        CBLAS_TRANSPOSE trans_a,
        CBLAS_TRANSPOSE trans_b,
        int m,
        int n,
        int k,
        T alpha,
        const T* a,
        int lda,
        const T* b,
        int ldb,
        T beta,
        T* c,
        int ldc
    );

    template<>
    inline void blas_gemm<float>(
        CBLAS_ORDER order,
        CBLAS_TRANSPOSE trans_a,
        CBLAS_TRANSPOSE trans_b,
        int m,
        int n,
        int k,
        float alpha,
        const float* a,
        int lda,
        const float* b,
        int ldb,
        float beta,
        float* c,
        int ldc
    ) {
        cblas_sgemm(
            order,
            trans_a,
            trans_b,
            m, n, k,
            alpha,
            a, lda,
            b, ldb,
            beta,
            c, ldc
        );
    }

    template<>
    inline void blas_gemm<double>(
        CBLAS_ORDER order,
        CBLAS_TRANSPOSE trans_a,
        CBLAS_TRANSPOSE trans_b,
        int m,
        int n,
        int k,
        double alpha,
        const double* a,
        int lda,
        const double* b,
        int ldb,
        double beta,
        double* c,
        int ldc
    ) {
        cblas_dgemm(
            order,
            trans_a,
            trans_b,
            m, n, k,
            alpha,
            a, lda,
            b, ldb,
            beta,
            c, ldc
        );
    }
}
#endif

#endif //NUMPP_BACKEND_BLAS_HPP