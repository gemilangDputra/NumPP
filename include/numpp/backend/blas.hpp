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

    template<typename T>
    void blas_ger(
        CBLAS_LAYOUT layout,
        int M,
        int N,
        T alpha,
        const T* a,
        int incA,
        const T* b,
        int incB,
        T* c,
        int ldc
    );

    template<>
    void blas_ger<float>(
        CBLAS_LAYOUT layout,
        int M,
        int N,
        float alpha,
        const float* a,
        int incA,
        const float* b,
        int incB,
        float* c,
        int ldc
    ) {
        cblas_sger(
            layout,
            M, N,
            alpha,
            a, incA,
            b, incB,
            c, ldc
        );
    }

    template<>
    void blas_ger<double>(
        CBLAS_LAYOUT layout,
        int M,
        int N,
        double alpha,
        const double* a,
        int incA,
        const double* b,
        int incB,
        double* c,
        int ldc
    ) {
        cblas_dger(
            layout,
            M, N,
            alpha,
            a, incA,
            b, incB,
            c, ldc
        );
    }
}
#endif

namespace numpp::threshold {
    inline size_t matmul_blas   = 125'000'000ULL;
    inline size_t matmul_repack = 12'500'000ULL;
    inline size_t dot_blas      = 15'500'000ULL;
    inline size_t outer_blas    = 15'500'000ULL;
}

#endif //NUMPP_BACKEND_BLAS_HPP