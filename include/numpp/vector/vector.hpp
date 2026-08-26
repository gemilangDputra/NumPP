#ifndef NUMPP_VECTOR_VECTORC_HPP
#define NUMPP_VECTOR_VECTORC_HPP

#include <numpp/vector/forward.hpp>
#include <numpp/vector/tool.hpp>
#include <numpp/vector/core.hpp>

#include <initializer_list>
#include <cstdint>

namespace numpp {
    template<typename T>
    class vector : public vector_base<vector<T>, T> {
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
            vector() = default;

            vector(std::initializer_list<T> list);

            template<general_vector_like EXPR>
            vector(const EXPR& other);

            vector(const vector& other);
            vector(vector&& other) noexcept;
            
            template<general_vector_like EXPR>
            vector& operator=(const EXPR& other);

            vector& operator=(const vector& other);
            vector& operator=(vector&& other) noexcept;

            static vector empty(size_t size);
            static vector zeros(size_t size);
            static vector ones(size_t size);
            static vector full(size_t size, const T& value);

            template<RandomEngine RNG>
            static vector rand(size_t size, const T& min, const T& max, RNG& rng);
            static vector rand(size_t size, const T& min=T{0}, const T& max=T{1});

            template<RandomEngine RNG>
            static vector randint(size_t size, int64_t min, int64_t max, RNG& rng);
            static vector randint(size_t size, int64_t min=0, int64_t max=1);
            
            template<RandomEngine RNG>
            static vector randn(size_t size, const T& mean, const T& stddev, RNG& rng);
            static vector randn(size_t size, const T& mean=T{0}, const T& stddev=T{1});

            template<general_vector_like EXPR>
            static vector empty_like(const EXPR& other);
            template<general_vector_like EXPR>
            static vector zeros_like(const EXPR& other);
            template<general_vector_like EXPR>
            static vector ones_like(const EXPR& other);
            template<general_vector_like EXPR>
            static vector full_like(const EXPR& other, const T& value);

            template<general_vector_like EXPR, RandomEngine RNG>
            static vector rand_like(const EXPR& other, const T& min, const T& max, RNG& rng);
            template<general_vector_like EXPR, RandomEngine RNG>
            static vector randint_like(const EXPR& other, int64_t min, int64_t max, RNG& rng);
            template<general_vector_like EXPR, RandomEngine RNG>
            static vector randn_like(const EXPR& other, const T& mean, const T& stddev, RNG& rng);

            template<general_vector_like EXPR>
            static vector rand(const EXPR& other, const T& min=T{0}, const T& max=T{1});
            template<general_vector_like EXPR>
            static vector randint(const EXPR& other, int64_t min=0, int64_t max=1);
            template<general_vector_like EXPR>
            static vector randn(const EXPR& other, const T& mean=T{0}, const T& stddev=T{1});

            static vector arange(const T& start, const T& stop, const T& step=T{1});
            static vector arange(const T& stop);

            static vector linspace(const T& start, const T& stop, size_t num, bool endpoint=true);
            static vector linspace(const T& stop, size_t num, bool endpoint=true);

            ~vector() {
                del();
            }
    };
}

#endif //NUMPP_VECTOR_VECTORC_HPP