#ifndef NUMPP_MATRIX_LINALG_SOLVE_HPP
#define NUMPP_MATRIX_LINALG_SOLVE_HPP

#include <numpp/matrix/tool.hpp>
#include <numpp/matrix/matrix.hpp>
#include <numpp/matrix/core.hpp>
#include <numpp/backend/lapack.hpp>
#include <string>

namespace numpp {
    namespace linalg {
        namespace detail {
            #if NUMPP_USE_LAPACK
            template<matrix_like A, matrix_like B>
            auto solve_lapack_same_strides(const A& a, const B& b);
            #endif
        }
    }
}

#endif //NUMPP_MATRIX_LINALG_SOLVE_HPP