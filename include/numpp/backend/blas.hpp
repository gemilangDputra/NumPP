#ifndef NUMPP_BACKEND_BLAS_HPP
#define NUMPP_BACKEND_BLAS_HPP

#if NUMPP_USE_BLAS
#ifdef NUMPP_BLAS_HEADR
    #include NUMPP_BLAS_HEADER
#else
    #include <cblas.h>
#endif

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

    template<typename T>
    T blas_dot(
        const int n,
        const T* a,
        const int incA,
        const T* b,
        const int incB
    );

    template<>
    inline float blas_dot<float>(
        const int n,
        const float* a,
        const int incA,
        const float* b,
        const int incB
    ) {
        return
        cblas_sdot(
            n,
            a, incA,
            b, incB
        );
    }
    
    template<>
    inline double blas_dot<double>(
        const int n,
        const double* a,
        const int incA,
        const double* b,
        const int incB
    ) {
        return
        cblas_ddot(
            n,
            a, incA,
            b, incB
        );
    }
}
#endif

#endif //NUMPP_BACKEND_BLAS_HPP