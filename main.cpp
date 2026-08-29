#include <numpp/matrix.hpp>
#include <numpp/vector.hpp>
#include <numpp/vector/linalg/dot.hpp>
#include <numpp/vector/linalg/outer.hpp>

int main() {
    const numpp::vectorf a = numpp::vectorf::arange(50);
    const numpp::vectorf b = numpp::vectorf::arange(0,100,2);

    const numpp::matrix_view a_m = numpp::reshape(a, 5,10);
    const numpp::matrix_view b_m = numpp::reshape(b, 10,5);

    auto c   = numpp::linalg::dot(a,b);
    auto c_m = numpp::linalg::dot(a_m,b_m);

    std::cout << c   << '\n';
    std::cout << c_m << '\n';

    std::cout << numpp::linalg::outer(a,b) << '\n';
}