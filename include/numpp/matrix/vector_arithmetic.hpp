#ifndef NUMPP_MATRIX_VECTOR_ARITHMETIC_HPP
#define NUMPP_MATRIX_VECTOR_ARITHMETIC_HPP

#include <numpp/matrix/core.hpp>
#include <numpp/matrix/tool.hpp>
#include <numpp/matrix/matrix.hpp>
#include <numpp/matrix/view/broadcast.hpp>
#include <utility>
#include <string>

namespace numpp {
    template<numpp_matrix A, matrix_like B>
    requires (
        std::same_as<
            typename A::value_type,
            typename B::value_type
        > &&
        can_add<typename A::value_type>
    )
}

#endif //NUMPP_MATRIX_VECTOR_ARITHMETIC_HPP