#ifndef NUMPP_BACKEND_TOOL_HPP
#define NUMPP_BACKEND_TOOL_HPP

#include <concepts>
#include <random>

namespace numpp {
    template<typename RNG>
    concept RandomEngine = std::uniform_random_bit_generator<RNG>;
    
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
            
            { -a } -> std::same_as<T>;
            { +a } -> std::same_as<T>;
        };
    
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
    
    namespace detail {
        struct all_t {};
    }
    
    inline constexpr detail::all_t all{};

    struct slice_range {
        size_t start;
        size_t stop;
        size_t step=1;

        slice_range(size_t start_, size_t stop_, size_t step_=1) : start(start_), stop(stop_), step(step_) {}
    };
}

#endif //NUMPP_BACKEND_TOOL_HPP