#include <numpp/matrix.hpp>
#include <numpp/vector.hpp>
#include <assert.h>

int main() {
    const numpp::vectorf a = numpp::vectorf::arange(50);
    auto b = a.view();

    auto c = b - a;
    c += a;
    c[0] = 10;
    c += 100;

    auto mask = (a < c) && (b == a) && (c > b) && (c < 500);

    if (mask.all())
        std::cout << "1\n";

    if (mask.any())
        std::cout << "2\n";
    
    if (mask[5])
        std::cout << "3\n";
    
    std::cout << c;
    std::cout << a.mean();
}