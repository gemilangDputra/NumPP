#ifndef NUMPP_MATRIX_MATRIXC_HPP
#define NUMPP_MATRIX_MATRIXC_HPP

#include <numpp/matrix/forward.hpp>
#include <numpp/matrix/core.hpp>
#include <cstddef>

namespace numpp {
    template<typename T>
    class matrix : public matrix_base<matrix<T>, T> {
        private:
            void alloc(bool zero_fill=false) {
                if (zero_fill) {
                    this->data_ = new T[this->size_]{};
                } else {
                    this->data_ = new T[this->size_];
                }
            }

            void del() {
                if (this->data_) {
                    delete[] this->data_;
                    this->data_ = nullptr;
                }
            }
        public:
            matrix() = default;

            matrix(std::initializer_list<std::initializer_list<T>> list, layout order=layout::rowmajor);

            template<matrix_like EXPR>
            matrix(const EXPR& other);

            matrix(const matrix& other);
            matrix(matrix&& other) noexcept;

            template<matrix_like EXPR>
            matrix<T>& operator=(const EXPR& other);

            matrix<T>& operator=(const matrix& other);
            matrix<T>& operator=(matrix&&) noexcept;

            static matrix empty(size_t row, size_t col, layout order=layout::rowmajor);
            static matrix zeros(size_t row, size_t col, layout order=layout::rowmajor);
            static matrix ones(size_t row, size_t col, layout order=layout::rowmajor);
            static matrix full(size_t row, size_t col, const T& value, layout order=layout::rowmajor);

            template<RandomEngine RNG>
            static matrix rand(size_t row, size_t col, const T& min, const T& max, layout order, RNG& rng);
            static matrix rand(size_t row, size_t col, const T& min, const T& max, layout order=layout::rowmajor);

            template<RandomEngine RNG>
            static matrix randint(size_t row, size_t col, int64_t min, int64_t max, layout order, RNG& rng);
            static matrix randint(size_t row, size_t col, int64_t min, int64_t max, layout order=layout::rowmajor);

            template<RandomEngine RNG>
            static matrix randn(size_t row, size_t col, const T& mean, const T& stddev, layout order, RNG& rng);
            static matrix randn(size_t row, size_t col, const T& mean=T{0}, const T& stddev=T{1}, layout order=layout::rowmajor);

            template<matrix_like EXPR>
            static matrix<T> empty_like(const EXPR& other);
            template<matrix_like EXPR>
            static matrix<T> zeros_like(const EXPR& other);
            template<matrix_like EXPR>
            static matrix<T> ones_like(const EXPR& other);
            template<matrix_like EXPR>
            static matrix<T> full_like(const EXPR& other, const T& value);

            template<matrix_like EXPR, RandomEngine RNG>
            static matrix rand_like(const EXPR& other, const T& min, const T& max, RNG& rng);
            template<matrix_like EXPR, RandomEngine RNG>
            static matrix randint_like(const EXPR& other, int64_t min, int64_t max, RNG& rng);
            template<matrix_like EXPR, RandomEngine RNG>
            static matrix randn_like(const EXPR& other, const T& mean, const T& stddev, RNG& rng);

            template<matrix_like EXPR>
            static matrix rand_like(const EXPR& other, const T& min, const T& max);
            template<matrix_like EXPR>
            static matrix randint_like(const EXPR& other, int64_t min, int64_t max);
            template<matrix_like EXPR>
            static matrix randn_like(const EXPR& other, const T& mean=T{0}, const T& stddev=T{1});

            static matrix eye(size_t row, size_t col, int k=0, layout order=layout::rowmajor);
            static matrix identity(size_t row, size_t col, int k=0, layout order=layout::rowmajor);
            static matrix diag(std::initializer_list<T> list, int k=0, layout order=layout::rowmajor);

            ~matrix() {
                del();
            }
    };
}

#endif //NUMPP_MATRIX_MATRIXC_HPP