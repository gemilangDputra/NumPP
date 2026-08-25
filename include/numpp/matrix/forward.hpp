#ifndef NUMPP_MATRIX_FORWARD_HPP
#define NUMPP_MATRIX_FORWARD_HPP

namespace numpp {
    template<typename T>
    class matrix;
    template<typename T>
    class matrix_view;
    template<class Derived, typename T>
    class matrix_base;

    struct slice_range;

    namespace detail { 
        struct all_t {};
        struct transflag_t{};
        inline constexpr transflag_t transflag{};
    }
    
    inline constexpr detail::all_t all{};

    enum class layout {
        rowmajor,
        colmajor
    };
}


#endif //NUMPP_MATRIX_FORWARD_HPP