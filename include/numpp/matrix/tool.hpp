#ifndef NUMPP_BACKEND_CONCEPT_HPP
#define NUMPP_BACKEND_CONCEPT_HPP

#include <concepts>
#include <type_traits>
#include <stdexcept>

#include <numpp/matrix/forward.hpp>

namespace numpp {
    template<class EXPR>
    concept is_matrix = requires(EXPR mat) { mat.col(); mat.row(); };

    template<class EXPR>
    concept strided_matrix =
    requires(const EXPR& mat) {
        typename EXPR::value_type;

        { mat.data() } -> std::convertible_to<const typename EXPR::value_type*>;
        { mat.row() } -> std::convertible_to<std::size_t>;
        { mat.col() } -> std::convertible_to<std::size_t>;
        { mat.size() } -> std::convertible_to<std::size_t>;
        { mat.rowstride() } -> std::convertible_to<std::size_t>;
        { mat.colstride() } -> std::convertible_to<std::size_t>;
        { mat.offset() } -> std::convertible_to<std::size_t>;
        { mat.order() } -> std::same_as<numpp::layout>;
        { mat(0, 0) } -> std::convertible_to<typename EXPR::value_type>;
    };

    template<typename EXPR>
    concept vector_1d =
    requires(const EXPR& v, std::size_t i) {
        typename EXPR::value_type;

        { v.size() } -> std::convertible_to<std::size_t>;
        { v.data() } -> std::convertible_to<const typename EXPR::value_type*>;
        { v[i] } -> std::convertible_to<typename EXPR::value_type>;
        }
        && (!requires(const EXPR& v) { v.row(); })
        && (!requires(const EXPR& v) { v.col(); });
    
    template<strided_matrix EXPR>
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

    template<strided_matrix EXPR>
    bool is_transpose(const EXPR& mat) {
        if (mat.order() == layout::rowmajor)
            return mat.rowstride() == 1 &&
                   mat.colstride() == mat.row();

        return mat.rowstride() == mat.col() &&
               mat.colstride() == 1;
    }

    template<typename T>
    concept is_numeric =
        requires(T a, T b) {
            { T{0} } -> std::same_as<T>;
            { T{1} } -> std::same_as<T>;

            { a + b } -> std::same_as<T>;
            { a - b } -> std::same_as<T>;
            { a * b } -> std::same_as<T>;
            { a / b } -> std::same_as<T>;

            { a += b } -> std::same_as<T&>;
            { a -= b } -> std::same_as<T&>;
            { a *= b } -> std::same_as<T&>;
            { a /= b } -> std::same_as<T&>;
        };

    template<class EXPR>
    concept matrix_derived = std::derived_from<EXPR, matrix_base<EXPR, typename EXPR::value_type>>;

    template<typename T>
    concept can_add =
        requires(T a, T b) {
            { a + b } -> std::same_as<T>;
        };

    template<typename T>
    concept can_sub =
        requires(T a, T b) {
            { a - b } -> std::same_as<T>;
        };

    template<typename T>
    concept can_mul =
        requires(T a, T b) {
            { a * b } -> std::same_as<T>;
        };

    template<typename T>
    concept can_div =
        requires(T a, T b) {
            { a / b } -> std::same_as<T>;
        };
    
    template<typename T>
    concept can_add_assign =
        requires(T a, T b) {
            { a += b } -> std::same_as<T&>;
        };

    template<typename T>
    concept can_sub_assign =
        requires(T a, T b) {
            { a -= b } -> std::same_as<T&>;
        };

    template<typename T>
    concept can_mul_assign =
        requires(T a, T b) {
            { a *= b } -> std::same_as<T&>;
        };

    template<typename T>
    concept can_div_assign =
        requires(T a, T b) {
            { a /= b } -> std::same_as<T&>;
        };
}

#endif //NUMPP_BACKEND_CONCEPT_HPP