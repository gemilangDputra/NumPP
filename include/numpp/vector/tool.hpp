#ifndef NUMPP_VECTOR_TOOL_HPP
#define NUMPP_VECTOR_TOOL_HPP

#include <concepts>
#include <type_traits>
#include <stdexcept>

#include <numpp/backend/tool.hpp>
#include <numpp/vector/forward.hpp>

namespace numpp {
    template<class EXPR>
    concept vector_like = requires(const EXPR& vec) {
        typename EXPR::value_type;
        { vec.data() } -> std::convertible_to<const typename EXPR::value_type*>;
        { vec.size() } -> std::convertible_to<size_t>;
        { vec.stride() } -> std::convertible_to<size_t>;
        { vec.offset() } -> std::convertible_to<size_t>;
        { vec[0] } -> std::convertible_to<typename EXPR::value_type>;
    };
    
    template<class EXPR>
    concept general_vector_like = requires(const EXPR& vec) {
        typename EXPR::value_type;
        { vec.data() } -> std::convertible_to<const typename EXPR::value_type*>;
        { vec.size() } -> std::convertible_to<size_t>;
        { vec[0] } -> std::convertible_to<typename EXPR::value_type>;
    };

    template<class T>
    concept numpp_vector = std::derived_from< T,vector_base<T, typename T::value_type>>;
}

#endif //NUMPP_VECTOR_TOOL_HPP