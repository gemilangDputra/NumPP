#ifndef NUMPP_MATRIX_FORWARD_HPP
#define NUMPP_MATRIX_FORWARD_HPP

namespace numpp {
    template<typename T>
    class matrix;
    template<typename T>
    class matrix_view;
    template<class Derived, typename T>
    class matrix_base;

    namespace detail {
        struct transflag_t{};
        inline constexpr transflag_t transflag{};
    }

    enum class layout {
        rowmajor,
        colmajor
    };
}


#endif //NUMPP_MATRIX_FORWARD_HPP