#ifndef NUMPP_VECTOR_IO_HPP
#define NUMPP_VECTOR_IO_HPP

#include <numpp/vector/tool.hpp>
#include <numpp/vector/core.hpp>
#include <numpp/backend/io.hpp>

#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <algorithm>
#include <vector>

namespace numpp {
    template<class Derived, typename T>
    std::ostream& operator<<(
        std::ostream& os,
        const vector_base<Derived, T>& vec) {
        const auto& opt = detail::option(os);
        const size_t size = vec.size();
        const size_t edgeitems = opt.edgeitems;
        const bool truncate = size > static_cast<size_t>(opt.threshold);
        const bool truncate_element = truncate && size > 2 * edgeitems;
        std::ostringstream oss;
        
        if (opt.debug_mode) {
            os
            << "vector(size=" << vec.size()
            << ", stride=" << vec.stride()
            << ", offset=" << vec.offset()
            << ')';
            return os;
        }

        switch (opt.floatformat) {
        case print_option::format::fixed:
            oss << std::fixed;
            break;

        case print_option::format::scientific:
            oss << std::scientific;
            break;

        default:
            oss << std::defaultfloat;
            break;
        }

        oss << std::setprecision(opt.precision);
        if (vec.size() == 0) {
            os << "vector(0) = []";
            return os;
        }

        if (vec.size() == 1) {
            os << detail::format_value(oss, vec[0]);
            return os;
        }

        struct element {
            size_t index;
            std::string text;
        };

        std::vector<element> elements;
        size_t width = 0;
        auto add_element = [&](size_t i) {
            auto text = detail::format_value(oss, vec[i]);
            width = std::max(width, text.size());
            elements.push_back({i, std::move(text)});
        };

        if (!truncate_element) {
            for (size_t i = 0; i < size; ++i)
                add_element(i);
        } else {
            for (size_t i = 0; i < edgeitems; ++i)
                add_element(i);
            for (size_t i = size - edgeitems; i < size; ++i)
                add_element(i);
        }

        auto get_text = [&](size_t i) -> const std::string& {
            for (auto& e : elements) {
                if (e.index == i)
                    return e.text;
            }
            static const std::string dots = "...";
            return dots;
        };

        auto print_row = [&]() {
            os << "   ";
            size_t current_width = 0;
            size_t index = 0;
            auto print_col = [&](size_t i) {
                detail::print_element(
                    os,
                    get_text(i),
                    index++,
                    width,
                    opt.linewidth,
                    current_width);
            };

            if (!truncate_element) {
                for (size_t i = 0; i < size; ++i)
                    print_col(i);
            } else {
                for (size_t i = 0; i < edgeitems; ++i)
                    print_col(i);
                detail::print_element(
                    os,
                    "...",
                    index++,
                    width,
                    opt.linewidth,
                    current_width);
                for (size_t i = size - edgeitems; i < size; ++i)
                    print_col(i);
            }
        };

        os << "vector("
        << size
        << ") =\n[\n";
        print_row();
        os << "\n]";
        return os;
    }
}

#endif //NUMPP_VECTOR_IO_HPP