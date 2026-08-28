#include <numpp/matrix.hpp>
#include <numpp/vector.hpp>
#include <numpp/vector/linalg/dot.hpp>

int main() {
    const numpp::vectorf a = numpp::vectorf::arange(50);
    auto b = a.view();

    auto c = numpp::linalg::dot(a,b);
    std::cout << c;
}