#include <numpp/matrix.hpp>
#include <numpp/vector.hpp>

int main() {
    numpp::vectorf a = numpp::vectorf::arange(50);
    numpp::vectorf a1 = numpp::vectorf::arange(250);
    numpp::matrixf b = numpp::reshape(a1, 5, 50);

    std::cout << a << '\n';
    std::cout << b << '\n';

    numpp::print_option debug;
    debug.debug_mode = true;
    
    std::cout << debug;
    std::cout << a << '\n';
    std::cout << b << '\n';
}