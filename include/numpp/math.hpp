#ifndef NUMPP_MATH_HPP
#define NUMPP_MATH_HPP

#include <cmath>

namespace numpp {
    namespace detail {
        template<typename T>
        auto sqrt(const T& a) {
            using std::sqrt;
            return sqrt(a);
        }

        template<typename T>
        auto cbrt(const T& a) {
            using std::cbrt;
            return cbrt(a);
        }

        template<typename T>
        auto pow(const T& a, const T& b) {
            using std::pow;
            return pow(a,b);
        }
        
        template<typename T>
        auto hypot(const T& a, const T& b) {
            using std::hypot;
            return hypot(a,b);
        }
        
        template<typename T>
        auto cos(const T& a) {
            using std::cos;
            return cos(a);
        }
        
        template<typename T>
        auto sin(const T& a) {
            using std::sin;
            return sin(a);
        }
        
        template<typename T>
        auto tan(const T& a) {
            using std::tan;
            return tan(a);
        }
        
        template<typename T>
        auto acos(const T& a) {
            using std::acos;
            return acos(a);
        }
        
        template<typename T>
        auto asin(const T& a) {
            using std::asin;
            return asin(a);
        }
        
        template<typename T>
        auto atan(const T& a) {
            using std::atan;
            return atan(a);
        }

        template<typename T>
        auto cosh(const T& a) {
            using std::cosh;
            return cosh(a);
        }
        
        template<typename T>
        auto sinh(const T& a) {
            using std::sinh;
            return sinh(a);
        }
        
        template<typename T>
        auto tanh(const T& a) {
            using std::tanh;
            return tanh(a);
        }
        
        template<typename T>
        auto exp(const T& a) {
            using std::exp;
            return exp(a);
        }
        
        template<typename T>
        auto log(const T& a) {
            using std::log;
            return log(a);
        }
        
        template<typename T>
        auto log2(const T& a) {
            using std::log2;
            return log2(a);
        }
        
        template<typename T>
        auto log10(const T& a) {
            using std::log10;
            return log10(a);
        }
        
        template<typename T>
        auto ceil(const T& a) {
            using std::ceil;
            return ceil(a);
        }
        
        template<typename T>
        auto floor(const T& a) {
            using std::floor;
            return floor(a);
        }
        
        template<typename T>
        auto round(const T& a) {
            using std::round;
            return round(a);
        }
        
        template<typename T>
        auto trunc(const T& a) {
            using std::trunc;
            return trunc(a);
        }
        
        template<typename T>
        auto max(const T& a, const T& b) {
            using std::max;
            return max(a, b);
        }
        
        template<typename T>
        auto min(const T& a, const T& b) {
            using std::min;
            return min(a, b);
        }
        
        template<typename T>
        auto abs(const T& a) {
            return +a;
        }
        
        template<typename T>
        auto neg(const T& a) {
            return -a;
        }
    }

    template<typename T>
    auto sqrt(const T& a) {
        return detail::sqrt(a);
    }

    template<typename T>
    auto cbrt(const T& a) {
        return detail::cbrt(a);
    }

    template<typename T>
    auto pow(const T& a, const T& b) {
        return detail::pow(a, b);
    }

    template<typename T>
    auto hypot(const T& a, const T& b) {
        return detail::hypot(a, b);
    }

    template<typename T>
    auto cos(const T& a) {
        return detail::cos(a);
    }

    template<typename T>
    auto sin(const T& a) {
        return detail::sin(a);
    }

    template<typename T>
    auto tan(const T& a) {
        return detail::tan(a);
    }

    template<typename T>
    auto acos(const T& a) {
        return detail::acos(a);
    }

    template<typename T>
    auto asin(const T& a) {
        return detail::asin(a);
    }

    template<typename T>
    auto atan(const T& a) {
        return detail::atan(a);
    }

    template<typename T>
    auto cosh(const T& a) {
        return detail::cosh(a);
    }

    template<typename T>
    auto sinh(const T& a) {
        return detail::sinh(a);
    }

    template<typename T>
    auto tanh(const T& a) {
        return detail::tanh(a);
    }

    template<typename T>
    auto exp(const T& a) {
        return detail::exp(a);
    }

    template<typename T>
    auto log(const T& a) {
        return detail::log(a);
    }

    template<typename T>
    auto log2(const T& a) {
        return detail::log2(a);
    }

    template<typename T>
    auto log10(const T& a) {
        return detail::log10(a);
    }

    template<typename T>
    auto ceil(const T& a) {
        return detail::ceil(a);
    }

    template<typename T>
    auto floor(const T& a) {
        return detail::floor(a);
    }

    template<typename T>
    auto round(const T& a) {
        return detail::round(a);
    }

    template<typename T>
    auto trunc(const T& a) {
        return detail::trunc(a);
    }

    template<typename T>
    auto max(const T& a, const T& b) {
        return detail::max(a, b);
    }
    
    template<typename T>
    auto min(const T& a, const T& b) {
        return detail::min(a, b);
    }
    
    template<typename T>
     requires requires(T a) { +a; }
    auto abs(const T& a) {
        return detail::abs(a);
    }
    
    template<typename T>
     requires requires(T a) { -a; }
    auto neg(const T& a) {
        return detail::neg(a);
    }
}

#endif //NUMPP_MATH_HPP