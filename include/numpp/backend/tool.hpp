#ifndef NUMPP_BACKEND_TOOL_HPP
#define NUMPP_BACKEND_TOOL_HPP

#include <concepts>
#include <random>

namespace {
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
}

#endif //NUMPP_BACKEND_TOOL_HPP