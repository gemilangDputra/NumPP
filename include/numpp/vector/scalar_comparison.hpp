#ifndef NUMPP_VECTOR_SCALAR_COMPARISON_HPP
#define NUMPP_VECTOR_SCALAR_COMPARISON_HPP

#include <numpp/vector/core.hpp>
#include <numpp/vector/tool.hpp>
#include <numpp/vector/vector.hpp>
#include <numpp/vector/op_expr.hpp>

namespace numpp {
    template<numpp_vector EXPR> requires (can_add<typename EXPR::value_type>)
    vector<bool> operator<(const EXPR& vec, const typename EXPR::value_type& scalar) {
        return detail::vector_scalar_expr(vec, scalar,[](auto x, auto y) { return x < y; });
    }
    
    template<numpp_vector EXPR> requires (can_add<typename EXPR::value_type>)
    vector<bool> operator>(const EXPR& vec, const typename EXPR::value_type& scalar) {
        return detail::vector_scalar_expr(vec, scalar,[](auto x, auto y) { return x > y; });
    }
    
    template<numpp_vector EXPR> requires (can_add<typename EXPR::value_type>)
    vector<bool> operator<=(const EXPR& vec, const typename EXPR::value_type& scalar) {
        return detail::vector_scalar_expr(vec, scalar,[](auto x, auto y) { return x <= y; });
    }
    
    template<numpp_vector EXPR> requires (can_add<typename EXPR::value_type>)
    vector<bool> operator>=(const EXPR& vec, const typename EXPR::value_type& scalar) {
        return detail::vector_scalar_expr(vec, scalar,[](auto x, auto y) { return x >= y; });
    }
    
    template<numpp_vector EXPR> requires (can_add<typename EXPR::value_type>)
    vector<bool> operator==(const EXPR& vec, const typename EXPR::value_type& scalar) {
        return detail::vector_scalar_expr(vec, scalar,[](auto x, auto y) { return x == y; });
    }
    
    template<numpp_vector EXPR> requires (can_add<typename EXPR::value_type>)
    vector<bool> operator!=(const EXPR& vec, const typename EXPR::value_type& scalar) {
        return detail::vector_scalar_expr(vec, scalar,[](auto x, auto y) { return x != y; });
    }
    
    template<numpp_vector EXPR> requires (can_add<typename EXPR::value_type>)
    vector<bool> operator&&(const EXPR& vec, const typename EXPR::value_type& scalar) {
        return detail::vector_scalar_expr(vec, scalar,[](auto x, auto y) { return x && y; });
    }
    
    template<numpp_vector EXPR> requires (can_add<typename EXPR::value_type>)
    vector<bool> operator||(const EXPR& vec, const typename EXPR::value_type& scalar) {
        return detail::vector_scalar_expr(vec, scalar,[](auto x, auto y) { return x || y; });
    }

    template<numpp_vector EXPR> requires (can_add<typename EXPR::value_type>)
    vector<bool> operator<(const typename EXPR::value_type& scalar, const EXPR& vec) {
        return detail::vector_scalar_expr(vec, scalar,[](auto x, auto y) { return y < x; });
    }
    
    template<numpp_vector EXPR> requires (can_add<typename EXPR::value_type>)
    vector<bool> operator>(const typename EXPR::value_type& scalar, const EXPR& vec) {
        return detail::vector_scalar_expr(vec, scalar,[](auto x, auto y) { return y > x; });
    }
    
    template<numpp_vector EXPR> requires (can_add<typename EXPR::value_type>)
    vector<bool> operator<=(const typename EXPR::value_type& scalar, const EXPR& vec) {
        return detail::vector_scalar_expr(vec, scalar,[](auto x, auto y) { return y <= x; });
    }
    
    template<numpp_vector EXPR> requires (can_add<typename EXPR::value_type>)
    vector<bool> operator>=(const typename EXPR::value_type& scalar, const EXPR& vec) {
        return detail::vector_scalar_expr(vec, scalar,[](auto x, auto y) { return y >= x; });
    }
    
    template<numpp_vector EXPR> requires (can_add<typename EXPR::value_type>)
    vector<bool> operator==(const typename EXPR::value_type& scalar, const EXPR& vec) {
        return detail::vector_scalar_expr(vec, scalar,[](auto x, auto y) { return y == x; });
    }
    
    template<numpp_vector EXPR> requires (can_add<typename EXPR::value_type>)
    vector<bool> operator!=(const typename EXPR::value_type& scalar, const EXPR& vec) {
        return detail::vector_scalar_expr(vec, scalar,[](auto x, auto y) { return y != x; });
    }
    
    template<numpp_vector EXPR> requires (can_add<typename EXPR::value_type>)
    vector<bool> operator&&(const typename EXPR::value_type& scalar, const EXPR& vec) {
        return detail::vector_scalar_expr(vec, scalar,[](auto x, auto y) { return y && x; });
    }
    
    template<numpp_vector EXPR> requires (can_add<typename EXPR::value_type>)
    vector<bool> operator||(const typename EXPR::value_type& scalar, const EXPR& vec) {
        return detail::vector_scalar_expr(vec, scalar,[](auto x, auto y) { return y || x; });
    }
}

#endif //NUMPP_VECTOR_SCALAR_COMPARISON_HPP