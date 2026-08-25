#ifndef NUMPP_MATRIX_TOOL_HPP
#define NUMPP_MATRIX_TOOL_HPP

#include <concepts>
#include <type_traits>
#include <stdexcept>

#include <numpp/backend/tool.hpp>
#include <numpp/matrix/forward.hpp>

namespace numpp {
    template<class EXPR>
    concept matrix_like =
    requires(const EXPR& mat) {
        typename EXPR::value_type;

        { mat.data() } -> std::convertible_to<const typename EXPR::value_type*>;
        { mat.row() } -> std::convertible_to<size_t>;
        { mat.col() } -> std::convertible_to<size_t>;
        { mat.size() } -> std::convertible_to<size_t>;
        { mat.rowstride() } -> std::convertible_to<size_t>;
        { mat.colstride() } -> std::convertible_to<size_t>;
        { mat.offset() } -> std::convertible_to<size_t>;
        { mat.order() } -> std::same_as<numpp::layout>;
        { mat(0, 0) } -> std::convertible_to<typename EXPR::value_type>;
    };

    template<matrix_like EXPR>
    bool is_transpose(const EXPR& mat) {
        if (mat.order() == layout::rowmajor)
            return mat.rowstride() == 1 &&
                   mat.colstride() == mat.row();

        return mat.rowstride() == mat.col() &&
               mat.colstride() == 1;
    }
    
    template<matrix_like EXPR>
    bool is_contiguous(const EXPR& mat) {
        if (mat.offset() != 0)
            return false;

        const bool trans = is_transpose(mat);

        if (mat.order() == layout::rowmajor) {
            return trans
                ? mat.rowstride() == 1 &&
                mat.colstride() == mat.row()
                : mat.rowstride() == mat.col() &&
                mat.colstride() == 1;
        }

        return trans
            ? mat.rowstride() == mat.col() &&
            mat.colstride() == 1
            : mat.rowstride() == 1 &&
            mat.colstride() == mat.row();
    }

    template<class T>
    concept numpp_matrix = std::derived_from< T,matrix_base<T, typename T::value_type>>;
}

#endif //NUMPP_MATRIX_TOOL_HPP