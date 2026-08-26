#ifndef NUMPP_VECTOR_HPP
#define NUMPP_VECTOR_HPP

#include <numpp/vector/forward.hpp>
#include <numpp/vector/tool.hpp>
#include <numpp/vector/core.hpp>
#include <numpp/vector/constructor.hpp>
#include <numpp/vector/io.hpp>
#include <numpp/vector/view/view.hpp>
#include <numpp/vector/view/slice.hpp>

namespace numpp {
    using vectorf   = vector<float>;
    using vectord   = vector<double>;
    using vectori   = vector<int>;
    using vectorb   = vector<bool>;
    using vectorld  = vector<long double>;
    using vectori32 = vector<int32_t>;
    using vectori64 = vector<int64_t>;
    using vectori16 = vector<int16_t>;
    using vectori8  = vector<int8_t>;
    
    using vector_viewf   = vector_view<float>;
    using vector_viewd   = vector_view<double>;
    using vector_viewi   = vector_view<int>;
    using vector_viewb   = vector_view<bool>;
    using vector_viewld  = vector_view<long double>;
    using vector_viewi32 = vector_view<int32_t>;
    using vector_viewi64 = vector_view<int64_t>;
    using vector_viewi16 = vector_view<int16_t>;
    using vector_viewi8  = vector_view<int8_t>;
}

#endif //NUMPP_VECTOR_HPP