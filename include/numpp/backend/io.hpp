#ifndef NUMPP_BACKEND_PRINT_HPP
#define NUMPP_BACKEND_PRINT_HPP

#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

namespace numpp {
    struct print_option {
        enum class format {
            fixed,
            scientific,
            defaultfloat
        };

        int precision = 3;
        int threshold = 1000;
        int linewidth = 75;
        int edgeitems = 3;
        format floatformat = format::defaultfloat;
    };

    namespace detail {
    inline int option_index = std::ios_base::xalloc();
    inline void callback(
        std::ios_base::event ev,
        std::ios_base& ios,
        int idx) {
        auto*& p = ios.pword(idx);

        switch (ev) {
        case std::ios_base::erase_event:
            delete static_cast<print_option*>(p);
            p = nullptr;
            break;

        case std::ios_base::copyfmt_event:
            if (p)
                p = new print_option(
                    *static_cast<print_option*>(p));
            break;

        default:
            break;
        }
    }

    inline print_option& get_option(std::ostream& os) {
        auto*& ptr = reinterpret_cast<print_option*&>( os.pword(option_index));
        if (!ptr) {
            ptr = new print_option;
            os.register_callback(callback, option_index);
        }
        return *ptr;
    }

    inline const print_option& option(std::ostream& os) {
        return get_option(os);
    }

    template<typename T>
    std::string format_value(std::ostringstream& oss, const T& value) {
        oss.str("");
        oss.clear();
        oss << value;
        return oss.str();
    }

    inline bool need_newline(size_t linewidth, size_t current_width, size_t next_width) {
        if (linewidth <= 2)
            return false;

        return current_width > 1 &&
            current_width + next_width > linewidth - 2;
    }

    template<typename String>
    void print_element(
        std::ostream& os, const String& text, size_t index, size_t width, size_t linewidth, size_t& current_width) {
        const size_t extra =
            width + (index ? 1 : 0);

        if (need_newline(linewidth, current_width, extra)) {
            os << "\n  ";
            current_width = 1;
        }

        if (index) {
            os << ' ';
            ++current_width;
        }
        os << std::setw(width) << text;
        current_width += width;
    }

    }
    inline std::ostream& operator<<(std::ostream& os, const print_option& opt) {
        detail::get_option(os) = opt;
        return os;
    }

}

#endif //NUMPP_BACKEND_PRINT_HPP