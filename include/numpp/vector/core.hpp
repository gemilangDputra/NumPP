#ifndef NUMPP_VECTOR_CORE_HPP
#define NUMPP_VECTOR_CORE_HPP

#include <numpp/vector/forward.hpp>
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