#ifndef NUMPP_VECTOR_OP_EXPR_HPP
#define NUMPP_VECTOR_OP_EXPR_HPP

#include <numpp/vector/core.hpp>
#include <numpp/vector/tool.hpp>
#include <numpp/vector/vector.hpp>
#include <utility>
#include <string>
#include <string_view>

namespace numpp::detail {
    template<general_vector_like A, general_vector_like B, typename Op>
    requires std::same_as<typename A::value_type, typename B::value_type>
    vector<typename A::value_type>
    vector_binary_expr(const A& a, const B& b, Op op, std::string_view operation_name) {
        using T = typename A::value_type;

        if (a.size() != b.size())
            throw std::invalid_argument(
                "numpp::operation<" +
                std::string(operation_name) +
                "> error: vector size mismatch:"
                "lhs has size " + std::to_string(a.size()) +
                ", rhs has size " + std::to_string(b.size()) +
                "; vectors must have the same size"
            );

        vector<T> out = vector<T>::empty_like(a);
        for (size_t i = 0; i < out.size(); ++i)
            out[i] = op(a[i], b[i]);
        return out;
    }
    
    template<general_vector_like A, general_vector_like B, typename Op>
    requires std::same_as<typename A::value_type, typename B::value_type>
    vector<typename A::value_type>&
    vector_binary_assign_expr(A& a, const B& b, Op op, std::string_view operation_name) {
        using T = typename A::value_type;
        if (a.size() != b.size())
            throw std::invalid_argument(
                "numpp::operation<" +
                std::string(operation_name) +
                "> error: vector size mismatch:"
                "lhs has size " + std::to_string(a.size()) +
                ", rhs has size " + std::to_string(b.size()) +
                "; vectors must have the same size"
            );
        
        for (size_t i = 0; i < a.size(); ++i)
            op(a[i], b[i]);
        return a;
    }
    
    template<general_vector_like EXPR, typename Op>
    vector<typename EXPR::value_type>
    vector_scalar_expr(const EXPR& vec, const typename EXPR::value_type scalar, Op op) {
        using T = typename EXPR::value_type;
        vector<T> out = vector<T>::empty_like(vec);
        for (size_t i = 0; i < out.size(); ++i)
            out[i] = op(vec[i], scalar);
        return out;
    }
    
    template<general_vector_like EXPR, typename Op>
    vector<typename EXPR::value_type>&
    vector_scalar_assign_expr(EXPR& vec, const typename EXPR::value_type scalar, Op op) {
        using T = typename EXPR::value_type;
        for (size_t i = 0; i < vec.size(); ++i)
            op(vec[i], scalar);
        return vec;
    }
    
    template<general_vector_like EXPR, typename Op>
    vector<typename EXPR::value_type>
    vector_unary_expr(const EXPR& vec, Op op) {
        using T = typename EXPR::value_type;
        vector<T> out = vector<T>::empty_like(vec);
        for (size_t i = 0; i < out.size(); ++i)
            out[i] = op(vec[i]);
        return out;
    }
    
    template<general_vector_like EXPR, typename Op>
    typename EXPR::value_type
    vector_reduction_expr(const EXPR& vec, Op op, std::string_view operation_name) {
        using T = typename EXPR::value_type;

        if (vec.size() == 0)
            throw std::invalid_argument(
                "numpp::operation<" +
                std::string(operation_name) +
                "> error: cannot reduce an empty vector"
            );

        T out = vec[0];
        for (size_t i = 1; i < vec.size(); ++i)
            op(out, vec[i]);
        return out;
    }
}

#endif //NUMPP_VECTOR_OP_EXPR_HPP