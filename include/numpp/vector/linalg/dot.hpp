#ifndef NUMPP_VECTOR_DOT_HPP
#define NUMPP_VECTOR_DOT_HPP

#include <numpp/vector/tool.hpp>
#include <numpp/vector/vector.hpp>
#include <numpp/vector/core.hpp>
#include <numpp/backend/blas.hpp>
#include <string>

namespace numpp {
    namespace linalg {
        namespace detail {
            #if NUMPP_USE_BLAS
            template<general_vector_like A, general_vector_like B>
            typename A::value_type dot_blas(const A& a, const B& b) {
                using T = typename A::value_type;
                int incA = 1;
                int incB = 1;
                const T* adata = a.data();
                const T* bdata = a.data();
                if constexpr (vector_like<A>) { incA = static_cast<int>(a.stride()); adata = adata + a.offset(); }
                if constexpr (vector_like<B>) { incB = static_cast<int>(b.stride()); bdata = bdata + b.offset(); }
                return ::numpp::detail::blas_dot(a.size(), adata, incA, bdata, incB);
            }
            #endif
            
            template<general_vector_like A, general_vector_like B>
            typename A::value_type dot_native(const A& a, const B& b) {
                using T = typename A::value_type;
                int incA = 1;
                int incB = 1;
                const T* adata = a.data();
                const T* bdata = a.data();
                if constexpr (vector_like<A>) { incA = static_cast<int>(a.stride()); adata = adata + a.offset(); }
                if constexpr (vector_like<B>) { incB = static_cast<int>(b.stride()); bdata = bdata + b.offset(); }
                T out = T{0};
                if (incA == 1 && incB == 1) { for (size_t i = 0; i < a.size(); ++i) out += adata[i] * bdata[i];          }
                else if (incB == 1        ) { for (size_t i = 0; i < a.size(); ++i) out += adata[i*incA] * bdata[i];     }
                else if (incA == 1        ) { for (size_t i = 0; i < a.size(); ++i) out += adata[i] * bdata[i*incB];     }
                else                        { for (size_t i = 0; i < a.size(); ++i) out += adata[i*incA] * bdata[i*incB];}
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
        auto dot(const A& a, const B& b) {
            if (a.size() != b.size())
                throw std::invalid_argument(
                    "numpp::operation<dot error: "
                    "vector size mismatch:"
                    "lhs has size " + std::to_string(a.size()) +
                    ", rhs has size " + std::to_string(b.size()) +
                    "; vectors must have the same size"
                );
            
            if (a.size() > 6144) return detail::dot_blas(a,b);
            else return detail::dot_native(a,b); 
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
    auto dot(const A& a, const B& b) {
        return linalg::dot(a,b);
    }
}

#endif //NUMPP_VECTOR_DOT_HPP