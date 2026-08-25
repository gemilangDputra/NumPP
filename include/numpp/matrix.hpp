#ifndef NUMPP_MATRIX_HPP
#define NUMPP_MATRIX_HPP

#include <numpp/matrix/core.hpp>
#include <numpp/matrix/matrix.hpp>
#include <numpp/matrix/constructor.hpp>

#include <numpp/matrix/forward.hpp>
#include <numpp/matrix/io.hpp>
#include <numpp/matrix/tool.hpp>

#include <numpp/matrix/view/view.hpp>
#include <numpp/matrix/view/transpose.hpp>
#include <numpp/matrix/view/slice.hpp>
#include <numpp/matrix/view/reshape.hpp>
#include <numpp/matrix/view/broadcast.hpp>

#include <numpp/matrix/arithmetic.hpp>
#include <numpp/matrix/scalar_arithmetic.hpp>
#include <numpp/matrix/comparison.hpp>
#include <numpp/matrix/scalar_comparison.hpp>

#include <numpp/matrix/math.hpp>

#include <numpp/matrix/linalg/matmul.hpp>

#include <cstddef>

namespace numpp {
    using matrixf   = matrix<float>;
    using matrixd   = matrix<double>;
    using matrixi   = matrix<int>;
    using matrixb   = matrix<bool>;
    using matrixld  = matrix<long double>;
    using matrixi32 = matrix<int32_t>;
    using matrixi64 = matrix<int64_t>;
    using matrixi16 = matrix<int16_t>;
    using matrixi8  = matrix<int8_t>;
    
    using matrix_viewf   = matrix_view<float>;
    using matrix_viewd   = matrix_view<double>;
    using matrix_viewi   = matrix_view<int>;
    using matrix_viewb   = matrix_view<bool>;
    using matrix_viewld  = matrix_view<long double>;
    using matrix_viewi32 = matrix_view<int32_t>;
    using matrix_viewi64 = matrix_view<int64_t>;
    using matrix_viewi16 = matrix_view<int16_t>;
    using matrix_viewi8  = matrix_view<int8_t>;
}

#endif //NUMPP_MATRIX_HPP