#include <numpp/matrix.hpp>
#include <numpp/matrix/linalg/matmul.hpp>
#include <iostream>

int main() {
    numpp::matrix<double> a({
        {100.0, 101.0, 102.0, 103.0, 104.0, 105.0},
        {110.0, 111.0, 112.0, 113.0, 114.0, 115.0},
        {120.0, 121.0, 122.0, 123.0, 124.0, 125.0},
        {130.0, 131.0, 132.0, 133.0, 134.0, 135.0},
        {140.0, 141.0, 142.0, 143.0, 144.0, 145.0},
        {150.0, 151.0, 152.0, 153.0, 154.0, 155.0}
    });

    numpp::matrix<double> b = numpp::matrix<double>::full(6,6, 0.111);

    auto b_slice = b.slice(
        numpp::slice_range(0,4),
        numpp::slice_range(0,4)
    );

    auto a_slice = a.slice(
        numpp::slice_range(1, 5),
        numpp::slice_range(1, 5)
    );

    numpp::print_option print_option;
    print_option.floatformat = numpp::print_option::format::fixed;
    
    std::cout << print_option;
    std::cout << "before:\n" << a << '\n';
    a_slice += b_slice;
    std::cout << "after:\n"  << a << '\n';
}