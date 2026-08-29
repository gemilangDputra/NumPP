#ifndef NUMPP_VECTOR_OUTER_HPP
#define NUMPP_VECTOR_OUTER_HPP

#include <numpp/vector/tool.hpp>
#include <numpp/vector/vector.hpp>
#include <numpp/vector/core.hpp>
#include <numpp/backend/blas.hpp>
#include <numpp/matrix/matrix.hpp>
#include <string>

namespace numpp {
    namespace linalg {
        namespace detail {
            #if NUMPP_USE_BLAS
            template<general_vector_like A, general_vector_like B>
            auto outer_blas(const A& a, const B& b) {
                using T = typename A::value_type;
                matrix<T> out = numpp::matrix<T>::zeros(a.size(), b.size());
                int incA = 1;
                int incB = 1;
                const T* adata = a.data();
                const T* bdata = b.data();
                if constexpr (vector_like<A>) { incA = static_cast<int>(a.stride()); adata = adata + a.offset(); }
                if constexpr (vector_like<B>) { incB = static_cast<int>(b.stride()); bdata = bdata + b.offset(); }
                ::numpp::detail::blas_ger<T>(CblasRowMajor, a.size(), b.size(), T{1}, adata, incA, bdata, incB, out.data(), out.rowstride());
                return out;
            }
            #endif
            
            template<general_vector_like A, general_vector_like B>
            auto outer_native(const A& a, const B& b) {
                using T = typename A::value_type;
                const size_t M = a.size();
                const size_t N = b.size();
                matrix<T> out = numpp::matrix<T>::empty(M, N);
                T* cdata = out.data();
                for (size_t i = 0; i < M; ++i) {
                    const T& ai = a[i];
                    for (size_t j = 0; j < N; ++j)
                        cdata[i * N + j] = ai * b[j];
                }
                return out;
            }
        }

        template<general_vector_like A, general_vector_like B>
        requires (
            std::same_as<
                typename A::value_type,
                typename B::value_type
            > &&
            is_numeric<typename A::value_type>
        )
        auto outer(const A& a, const B& b) {
            #if NUMPP_USE_BLAS
            if (a.size()*b.size() > ::numpp::threshold::outer_blas) return detail::outer_blas(a,b);
            #endif
            return detail::outer_native(a,b); 
        }
    }
}

#endif //NUMPP_VECTOR_OUTER_HPP