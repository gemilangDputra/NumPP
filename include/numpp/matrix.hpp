#ifndef NUMPP_MATRIX_HPP
#define NUMPP_MATRIX_HPP

#include <numpp/matrix/core.hpp>
#include <numpp/matrix/matrix.hpp>
#include <numpp/matrix/constructor.hpp>

#include <numpp/matrix/arithmetic.hpp>
#include <numpp/matrix/scalar_arithmetic.hpp>
#include <numpp/matrix/comparison.hpp>
#include <numpp/matrix/scalar_comparison.hpp>

#include <numpp/matrix/view/view.hpp>
#include <numpp/matrix/view/transpose.hpp>
#include <numpp/matrix/view/slice.hpp>
#include <numpp/matrix/view/reshape.hpp>
#include <numpp/matrix/view/broadcast.hpp>

#include <numpp/matrix/forward.hpp>
#include <numpp/matrix/io.hpp>
#include <numpp/matrix/tool.hpp>

namespace numpp {
    using matrixf = matrix<float>;
    using matrixd = matrix<double>;
    using matrixi = matrix<int>;
    using matrixb = matrix<bool>;
}

#endif //NUMPP_MATRIX_HPP