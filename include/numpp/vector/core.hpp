#ifndef NUMPP_VECTOR_CORE_HPP
#define NUMPP_VECTOR_CORE_HPP

#include <numpp/vector/forward.hpp>
#include <numpp/vector/tool.hpp>
#include <cstdint>

namespace numpp {
    template<class Derived, typename T>
    class vector_base {
        protected:
            T* data_;
            size_t size_;

            size_t stride_;
            size_t offset_;
        public:
            using value_type = std::remove_const_t<T>;

            size_t index_translate(size_t i) const {
                return offset_ + i * stride_;
            }
            
            const T* data() const { return data_; }
            T* data() { return data_; }

            size_t size() const { return size_; }
            size_t stride() const { return stride_; }
            size_t offset() const { return offset_; }

            const T& operator[](size_t i) const { return data_[index_translate(i)]; }
            T& operator[](size_t i) { return data_[index_translate(i)]; }
            
            vector_view<T> view();
            vector_view<T> slice(slice_range slice);
            vector_view<T> slice(size_t start, size_t stop, size_t step=1);
            vector_view<T> slice(detail::all_t);
            
            vector_view<const T> view() const;
            vector_view<const T> slice(slice_range slice) const;
            vector_view<const T> slice(size_t start, size_t stop, size_t step=1) const;
            vector_view<const T> slice(detail::all_t) const;

            template<general_vector_like B>
            requires (std::same_as<value_type, typename B::value_type> && can_add_assign<T>)
            Derived& operator+=(const B& b);
            template<general_vector_like B>
            requires (std::same_as<value_type, typename B::value_type> && can_sub_assign<T>)
            Derived& operator-=(const B& b);
            template<general_vector_like B>
            requires (std::same_as<value_type, typename B::value_type> && can_mul_assign<T>)
            Derived& operator*=(const B& b);
            template<general_vector_like B>
            requires (std::same_as<value_type, typename B::value_type> && can_div_assign<T>)
            Derived& operator/=(const B& b);

            Derived& operator+=(const T& scalar) requires (can_add_assign<T>);
            Derived& operator-=(const T& scalar) requires (can_sub_assign<T>);
            Derived& operator*=(const T& scalar) requires (can_mul_assign<T>);
            Derived& operator/=(const T& scalar) requires (can_div_assign<T>);

            
            bool all() requires(std::same_as<T, bool>) {
                for (size_t i = 0; i < size_; ++i) {
                    if (!(*this)[i]) return false;
                }
                return true;
            }

            bool any() requires(std::same_as<T, bool>) {
                for (size_t i = 0; i < size_; ++i) {
                    if ((*this)[i]) return true;
                }
                 return false;
            }
            
            vector<T> sqrt() const;
            vector<T> cbrt() const;

            vector<T> cos() const;
            vector<T> sin() const;
            vector<T> tan() const;
            
            vector<T> acos() const;
            vector<T> asin() const;
            vector<T> atan() const;

            vector<T> cosh() const;
            vector<T> sinh() const;
            vector<T> tanh() const;

            vector<T> log() const;
            vector<T> log2() const;
            vector<T> log10() const;

            vector<T> exp() const;

            vector<T> ceil() const;
            vector<T> floor() const;
            vector<T> round() const;
            vector<T> trunc() const;
            vector<T> abs() const;
            vector<T> neg() const;

            T max() const;
            T min() const;
            size_t argmax() const;
            size_t argmin() const;

            T sum() const requires(can_add_assign<T>);
            T prod() const requires(can_mul_assign<T>);
            T mean() const requires(can_add_assign<T> && can_div<T>);
        private:
            Derived& derived() {
                return static_cast<Derived&>(*this);
            }

            const Derived& derived() const {
                return static_cast<const Derived&>(*this);
            }
        };
}

#endif //NUMPP_VECTOR_CORE_HPP