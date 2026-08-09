#ifndef NUMPP_MATRIX_VIEW_BROADCAST_HPP
#define NUMPP_MATRIX_VIEW_BROADCAST_HPP

#include <numpp/matrix/core.hpp>
#include <numpp/matrix/view/view.hpp>
#include <numpp/matrix/tool.hpp>
#include <algorithm>
#include <utility>

namespace numpp::detail {
    template<strided_matrix A, strided_matrix B>
    bool is_broadcastable(const A& a, const B& b) {
        return
            (a.row() == b.row() || a.row() == 1 || b.row() == 1) &&
            (a.col() == b.col() || a.col() == 1 || b.col() == 1);
    }

    template<strided_matrix A, strided_matrix B>
    requires std::same_as<
        typename A::value_type,
        typename B::value_type
    >
    std::pair<
        matrix_view<const typename A::value_type>,
        matrix_view<const typename B::value_type>
    >
    broadcast(const A& a, const B& b) {
        using T = typename A::value_type;
        const size_t rows = std::max(a.row(), b.row());
        const size_t cols = std::max(a.col(), b.col());

        const size_t a_rowstride =
            a.row() == 1 && rows != 1
                ? 0
                : a.rowstride();

        const size_t a_colstride =
            a.col() == 1 && cols != 1
                ? 0
                : a.colstride();

        const size_t b_rowstride =
            b.row() == 1 && rows != 1
                ? 0
                : b.rowstride();

        const size_t b_colstride =
            b.col() == 1 && cols != 1
                ? 0
                : b.colstride();

        return {
            matrix_view<const T>(
                a.data(),
                a.size(),
                rows,
                cols,
                a_rowstride,
                a_colstride,
                a.order(),
                a.offset()
            ),

            matrix_view<const T>(
                b.data(),
                b.size(),
                rows,
                cols,
                b_rowstride,
                b_colstride,
                b.order(),
                b.offset()
            )
        };
    }
}

#endif //NUMPP_MATRIX_VIEW_BROADCAST_HPP