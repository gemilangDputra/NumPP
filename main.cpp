#include <numpp/matrix.hpp>

int main() {
    numpp::matrixf a = numpp::matrixf::rand(5,5, 0.0, 1.0);
    numpp::matrixf b = numpp::matrixf::randint(5,5, 0, 10);

    std::cout << a << '\n';
    std::cout << b << '\n';
}