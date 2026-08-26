#include <numpp/matrix.hpp>
#include <numpp/vector.hpp>
#include <assert.h>

int main() {
    const numpp::vectorf vec = numpp::vectorf::arange(50);

    auto a = vec.slice(25,50);
    auto b = vec.slice(0,50,2);
    auto c = vec.slice(0,25);
    
    std::cout << a << '\n';
    std::cout << b << '\n';
    std::cout << c << '\n';

    numpp::print_option debug;
    debug.debug_mode = true;
    
    std::cout << debug;
    std::cout << a << '\n';
    std::cout << b << '\n';
    std::cout << c << '\n';
}