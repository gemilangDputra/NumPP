#include <numpp/matrix.hpp>
#include <numpp/vector.hpp>

#include <numpp/matrix/linalg/dot.hpp>
#include <numpp/vector/linalg/dot.hpp>

int main() {
    auto vec50a = numpp::vectorf::linspace(0.1, 5.0f, 50);
    auto vec50b = numpp::vectorf::linspace(1, 50.0f, 50);

    auto mat50x5 = numpp::reshape(numpp::vectorf::linspace(0.1f, 25.0f, 250), 50,5).to_matrix();
    auto mat5x50 = numpp::reshape(numpp::vectorf::linspace(0.1f, 25.0f, 250), 5,50).to_matrix();

    numpp::print_option print_option;
    print_option.floatformat = numpp::print_option::format::fixed;
    print_option.precision = 2;

    std::cout << print_option;
    std::cout << numpp::linalg::dot(vec50a, vec50b) << '\n';
    std::cout << numpp::linalg::dot(mat50x5, mat5x50) << '\n'; //nama lain matrix multiplication/matmul
    std::cout << numpp::linalg::dot(vec50a, mat50x5) << '\n';
    std::cout << numpp::linalg::dot(mat5x50, vec50b) << '\n';
}