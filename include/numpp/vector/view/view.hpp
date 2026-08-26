#ifndef NUMPP_VECTOR_VIEW_HPP
#define NUMPP_VECTOR_VIEW_HPP

#include <numpp/vector/forward.hpp>
#include <numpp/vector/core.hpp>
#include <numpp/vector/tool.hpp>
#include <numpp/matrix/tool.hpp>
#include <string>

namespace numpp {
    template<typename T>
    class vector_view : public vector_base<vector_view<T>, T> {
        public:
            template<general_vector_like EXPR>
            vector_view(EXPR& other) {
                if constexpr (vector_like<EXPR>) {
                    this->data_ = other.data();
                    this->size_ = other.size();
                    this->stride_ = other.stride();
                    this->offset_ = other.offset();
                }
                else {
                    this->data_ = other.data();
                    this->size_ = other.size();
                    this->stride_ = 1;
                    this->offset_ = 0;
                }
            }

            template<general_vector_like EXPR>
            vector_view(EXPR& other, slice_range slice);
            template<general_vector_like EXPR>
            vector_view(EXPR& other, detail::all_t);
    };

    template<class Derived, typename T>
    vector_view<T> vector_base<Derived,T>::view() {
        return vector_view<T>(derived());
    }
    
    template<class Derived, typename T>
    vector_view<const T> vector_base<Derived,T>::view() const {
        return vector_view<const T>(derived());
    }

    template<general_vector_like EXPR>
    vector_view<typename EXPR::value_type> view(EXPR& other) {
        return vector_view<typename EXPR::value_type>(other);
    }
    
    template<general_vector_like EXPR>
    vector_view<const typename EXPR::value_type> view(const EXPR& other) {
        return vector_view<const typename EXPR::value_type>(other);
    }
}

#endif //NUMPP_VECTOR_VIEW_HPP