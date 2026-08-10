#include <numpp/matrix.hpp>
#include <iostream>
#include <stdexcept>

int main() {
    numpp::matrix<double> a({
        {1.0, 2.0, 3.0, 4.0, 5.0},
        {6.0, 7.0, 8.0, 9.0, 0.1},
        {1.1, 2.1, 3.1, 4.1, 5.1},
        {6.1, 7.1, 8.1, 9.1, 0.2},
        {1.2, 2.2, 3.2, 4.2, 5.2}
    });

    numpp::matrix<double> b({
        {1.0, 2.0, 3.0, 4.0, 5.0},
        {6.0, 7.0, 8.0, 9.0, 0.1},
        {1.1, 2.1, 3.1, 4.1, 5.1},
        {6.1, 7.1, 8.1, 9.1, 0.2},
        {1.2, 2.2, 3.2, 4.2, 5.2}
    });

    // ============================================================
    // operator+
    // ============================================================

    // Normal path: same shape
    auto add_normal = a + b;

    std::cout << "operator+ normal:\n";
    std::cout << add_normal << '\n';


    // Broadcast path: 5x5 + 1x5
    numpp::matrix<double> row_vector({
        {10.0, 20.0, 30.0, 40.0, 50.0}
    });

    auto add_broadcast = a + row_vector;

    std::cout << "operator+ broadcast:\n";
    std::cout << add_broadcast << '\n';


    // Error path
    numpp::matrix<double> invalid_add({
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    });

    try {
        auto add_error = a + invalid_add;
        std::cout << add_error << '\n';
    }
    catch (const std::exception& e) {
        std::cout << "operator+ exception:\n";
        std::cout << e.what() << '\n';
    }

    // Normal path: same shape
    auto add_assign_normal = numpp::matrix<double>::zeros_like(a);
    add_assign_normal += a;

    std::cout << "operator+= normal:\n";
    std::cout << add_assign_normal << '\n';


    // Broadcast path: 5x5 += 5x1
    numpp::matrix<double> col_vector({
        {10.0},
        {20.0},
        {30.0},
        {40.0},
        {50.0}
    });

    auto add_assign_broadcast = numpp::matrix<double>::zeros_like(a);
    add_assign_broadcast += col_vector;

    std::cout << "operator+= broadcast:\n";
    std::cout << add_assign_broadcast << '\n';


    // Error path
    try {
        auto add_assign_error = numpp::matrix<double>::zeros_like(a);
        add_assign_error += invalid_add;
    }
    catch (const std::exception& e) {
        std::cout << "operator+= exception:\n";
        std::cout << e.what() << '\n';
    }
}