#ifndef NUMPP_MATRIX_PRINT_HPP
#define NUMPP_MATRIX_PRINT_HPP

#include <numpp/matrix/tool.hpp>
#include <numpp/backend/io.hpp>

#include <numpp/matrix/core.hpp>

#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <algorithm>
#include <vector>

namespace numpp {
    template<class Derived, typename T>
    std::ostream& operator<<(std::ostream& os, const matrix_base<Derived, T>& mat) {
        const auto& opt = detail::option(os);
        const bool truncate = mat.size() > static_cast<size_t>(opt.threshold);
        const bool truncate_row = truncate && mat.row() > 2 * opt.edgeitems;
        const bool truncate_col = truncate && mat.col() > 2 * opt.edgeitems;
        std::ostringstream oss;

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
        struct element {
            size_t row;
            size_t col;
            std::string text;
        };

        std::vector<element> elements;
        size_t width = 0;

        auto add_element = [&](size_t r, size_t c) {
            auto text = detail::format_value(
                oss,
                mat(r, c));
            width = std::max(width, text.size());
            elements.push_back({
                r,
                c,
                std::move(text)
            });
        };

        if (!truncate) {
            for (size_t r = 0; r < mat.row(); ++r) {
                for (size_t c = 0; c < mat.col(); ++c) {
                    add_element(r, c);
                }
            }

        } else {
            for (size_t r = 0; r < (truncate_row ? opt.edgeitems : mat.row()); ++r) {
                for (size_t c = 0; c < (truncate_col ? opt.edgeitems : mat.col()); ++c) {
                    add_element(r, c);
                }

                if (truncate_col) {
                    for (size_t c = mat.col() - opt.edgeitems; c < mat.col(); ++c) {
                        add_element(r, c);
                    }
                }
            }

            if (truncate_row) {
                for (size_t r = mat.row() - opt.edgeitems; r < mat.row(); ++r) {
                    for (size_t c = 0; c < (truncate_col ? opt.edgeitems : mat.col()); ++c) {
                        add_element(r, c);
                    }

                    if (truncate_col) {
                        for (size_t c = mat.col() - opt.edgeitems;
                            c < mat.col();
                            ++c)
                        {
                            add_element(r, c);
                        }
                    }
                }
            }
        }


        auto get_text = [&](size_t r, size_t c) -> const std::string {
            for (auto& e : elements) {
                if (e.row == r && e.col == c)
                    return e.text;
            }
            static const std::string dots = "...";
            return dots;
        };

        auto print_row = [&](size_t r) {
            os << "  [";

            size_t current_width = 3;
            size_t index = 0;

            auto print_col = [&](size_t c) {
                detail::print_element(
                    os,
                    get_text(r, c),
                    index++,
                    width,
                    opt.linewidth,
                    current_width);
            };


            if (!truncate_col) {
                for (size_t c = 0; c < mat.col(); ++c)
                    print_col(c);

            } else {
                for (size_t c = 0; c < opt.edgeitems; ++c)
                    print_col(c);

                detail::print_element(
                    os,
                    "...",
                    index++,
                    width,
                    opt.linewidth,
                    current_width);

                for (size_t c = mat.col() - opt.edgeitems;
                    c < mat.col();
                    ++c) {
                    print_col(c);
                }
            }

            os << ']';
        };

        os << "matrix("
        << mat.row()
        << "x"
        << mat.col()
        << ", layout="
        << (mat.order() == layout::rowmajor
                ? "row major"
                : "col major")
        << ") =\n[\n";

        if (!truncate_row) {
            for (size_t r = 0; r < mat.row(); ++r) {
                print_row(r);
                if (r + 1 != mat.row())
                    os << ",\n";
            }

        } else {
            for (size_t r = 0; r < opt.edgeitems; ++r) {
                print_row(r);
                os << ",\n";
            }

            os << "  ...,\n";
            for (size_t r = mat.row() - opt.edgeitems; r < mat.row(); ++r) {
                print_row(r);

                if (r + 1 != mat.row())
                    os << ",\n";
            }
        }

        os << "\n]";
        return os;
    }
}

#endif //NUMPP_MATRIX_PRINT_HPP