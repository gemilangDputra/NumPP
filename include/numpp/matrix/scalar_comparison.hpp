#ifndef NUMPP_MATRIX_SCALAR_COMPARISON_HPP
#define NUMPP_MATRIX_SCALAR_COMPARISON_HPP

#include <numpp/matrix/core.hpp>
#include <numpp/matrix/tool.hpp>
#include <numpp/matrix/matrix.hpp>
#include <numpp/matrix/op_expr.hpp>

namespace numpp {
    template<numpp_matrix EXPR> requires (can_add<typename EXPR::value_type>)
    matrix<bool> operator<(const EXPR& mat, const typename EXPR::value_type& scalar) {
        return detail::matrix_scalar_expr(mat, scalar,[](auto x, auto y) { return x < y; });
    }
    
    template<numpp_matrix EXPR> requires (can_add<typename EXPR::value_type>)
    matrix<bool> operator>(const EXPR& mat, const typename EXPR::value_type& scalar) {
        return detail::matrix_scalar_expr(mat, scalar,[](auto x, auto y) { return x > y; });
    }
    
    template<numpp_matrix EXPR> requires (can_add<typename EXPR::value_type>)
    matrix<bool> operator<=(const EXPR& mat, const typename EXPR::value_type& scalar) {
        return detail::matrix_scalar_expr(mat, scalar,[](auto x, auto y) { return x <= y; });
    }
    
    template<numpp_matrix EXPR> requires (can_add<typename EXPR::value_type>)
    matrix<bool> operator>=(const EXPR& mat, const typename EXPR::value_type& scalar) {
        return detail::matrix_scalar_expr(mat, scalar,[](auto x, auto y) { return x >= y; });
    }
    
    template<numpp_matrix EXPR> requires (can_add<typename EXPR::value_type>)
    matrix<bool> operator==(const EXPR& mat, const typename EXPR::value_type& scalar) {
        return detail::matrix_scalar_expr(mat, scalar,[](auto x, auto y) { return x == y; });
    }
    
    template<numpp_matrix EXPR> requires (can_add<typename EXPR::value_type>)
    matrix<bool> operator!=(const EXPR& mat, const typename EXPR::value_type& scalar) {
        return detail::matrix_scalar_expr(mat, scalar,[](auto x, auto y) { return x != y; });
    }
    
    template<numpp_matrix EXPR> requires (can_add<typename EXPR::value_type>)
    matrix<bool> operator&&(const EXPR& mat, const typename EXPR::value_type& scalar) {
        return detail::matrix_scalar_expr(mat, scalar,[](auto x, auto y) { return x && y; });
    }
    
    template<numpp_matrix EXPR> requires (can_add<typename EXPR::value_type>)
    matrix<bool> operator||(const EXPR& mat, const typename EXPR::value_type& scalar) {
        return detail::matrix_scalar_expr(mat, scalar,[](auto x, auto y) { return x || y; });
    }

    template<numpp_matrix EXPR> requires (can_add<typename EXPR::value_type>)
    matrix<bool> operator<(const typename EXPR::value_type& scalar, const EXPR& mat) {
        return detail::matrix_scalar_expr(mat, scalar,[](auto x, auto y) { return y < x; });
    }
    
    template<numpp_matrix EXPR> requires (can_add<typename EXPR::value_type>)
    matrix<bool> operator>(const typename EXPR::value_type& scalar, const EXPR& mat) {
        return detail::matrix_scalar_expr(mat, scalar,[](auto x, auto y) { return y > x; });
    }
    
    template<numpp_matrix EXPR> requires (can_add<typename EXPR::value_type>)
    matrix<bool> operator<=(const typename EXPR::value_type& scalar, const EXPR& mat) {
        return detail::matrix_scalar_expr(mat, scalar,[](auto x, auto y) { return y <= x; });
    }
    
    template<numpp_matrix EXPR> requires (can_add<typename EXPR::value_type>)
    matrix<bool> operator>=(const typename EXPR::value_type& scalar, const EXPR& mat) {
        return detail::matrix_scalar_expr(mat, scalar,[](auto x, auto y) { return y >= x; });
    }
    
    template<numpp_matrix EXPR> requires (can_add<typename EXPR::value_type>)
    matrix<bool> operator==(const typename EXPR::value_type& scalar, const EXPR& mat) {
        return detail::matrix_scalar_expr(mat, scalar,[](auto x, auto y) { return y == x; });
    }
    
    template<numpp_matrix EXPR> requires (can_add<typename EXPR::value_type>)
    matrix<bool> operator!=(const typename EXPR::value_type& scalar, const EXPR& mat) {
        return detail::matrix_scalar_expr(mat, scalar,[](auto x, auto y) { return y != x; });
    }
    
    template<numpp_matrix EXPR> requires (can_add<typename EXPR::value_type>)
    matrix<bool> operator&&(const typename EXPR::value_type& scalar, const EXPR& mat) {
        return detail::matrix_scalar_expr(mat, scalar,[](auto x, auto y) { return y && x; });
    }
    
    template<numpp_matrix EXPR> requires (can_add<typename EXPR::value_type>)
    matrix<bool> operator||(const typename EXPR::value_type& scalar, const EXPR& mat) {
        return detail::matrix_scalar_expr(mat, scalar,[](auto x, auto y) { return y || x; });
    }
}

#endif //NUMPP_MATRIX_SCALAR_COMPARISON_HPP