#ifndef NUMPP_VECTOR_VIEW_SLICE_HPP
#define NUMPP_VECTOR_VIEW_SLICE_HPP

#include <numpp/vector/forward.hpp>
#include <numpp/vector/core.hpp>
#include <numpp/vector/tool.hpp>
#include <numpp/matrix/tool.hpp>
#include <string>
#include <stdexcept>

namespace numpp {
    template<typename T>
    template<general_vector_like EXPR>
    vector_view<T>::vector_view(EXPR& other, slice_range slice) {
        if (slice.step == 0)
            throw std::invalid_argument(
                "numpp::operation<slice> error: "
                "slice step must be greater than zero"
            );

        if (slice.start > slice.stop)
            throw std::invalid_argument(
                "numpp::operation<slice> error:"
                "slice start (" +
                std::to_string(slice.start) +
                ") cannot be greater than stop (" +
                std::to_string(slice.stop) + ")"
            );
        
        if  (slice.stop > other.size())
            throw std::invalid_argument(
                "numpp::operation<slice> error: "
                "slice stop (" +
                std::to_string(slice.stop) +
                ") exceeds vector size count (" +
                std::to_string(other.size()) + ")"
            );
            
        size_t sizes = ((slice.stop - slice.start) + slice.step - 1) / slice.step;
        
        if constexpr (vector_like<EXPR>) {
            this->data_ = other.data();
            this->size_ = sizes;
            this->stride_ = other.stride() * slice.step;
            this->offset_ = other.offset() + slice.start * other.stride();
        }
        else {
            this->data_ = other.data();
            this->size_ = sizes;
            this->stride_ = slice.step;
            this->offset_ = slice.start * other.stride();
        }
    }

    template<typename T>
    template<general_vector_like EXPR>
    vector_view<T>::vector_view(EXPR& other, detail::all_t) {
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
    
    template<class Derived, typename T>
    vector_view<T> vector_base<Derived,T>::slice(slice_range slice) {
        return vector_view<T>(derived(), slice);
    }
    
    template<class Derived, typename T>
    vector_view<const T> vector_base<Derived,T>::slice(slice_range slice) const {
        return vector_view<const T>(derived(), slice);
    }
    
    template<general_vector_like EXPR>
    vector_view<typename EXPR::value_type> slice(EXPR& other, slice_range slice) {
        return vector_view<typename EXPR::value_type>(other, slice);
    }
    
    template<general_vector_like EXPR>
    vector_view<const typename EXPR::value_type> slice(const EXPR& other, slice_range slice) {
        return vector_view<const typename EXPR::value_type>(other, slice);
    }

    template<class Derived, typename T>
    vector_view<T> vector_base<Derived,T>::slice(size_t start, size_t stop, size_t step) {
        return vector_view<T>(derived(), slice_range(start, stop, step));
    }
    
    template<class Derived, typename T>
    vector_view<const T> vector_base<Derived,T>::slice(size_t start, size_t stop, size_t step) const {
        return vector_view<const T>(derived(), slice_range(start, stop, step));
    }
    
    template<general_vector_like EXPR>
    vector_view<typename EXPR::value_type> slice(EXPR& other, size_t start, size_t stop, size_t step) {
        return vector_view<typename EXPR::value_type>(other, slice_range(start, stop, step));
    }
    
    template<general_vector_like EXPR>
    vector_view<const typename EXPR::value_type> slice(const EXPR& other, size_t start, size_t stop, size_t step) {
        return vector_view<const typename EXPR::value_type>(other, slice_range(start, stop, step));
    }
    
    template<class Derived, typename T>
    vector_view<T> vector_base<Derived,T>::slice(detail::all_t) {
        return vector_view<T>(derived(), all);
    }
    
    template<class Derived, typename T>
    vector_view<const T> vector_base<Derived,T>::slice(detail::all_t) const {
        return vector_view<const T>(derived(), all);
    }
    
    template<general_vector_like EXPR>
    vector_view<typename EXPR::value_type> slice(EXPR& other, detail::all_t) {
        return vector_view<typename EXPR::value_type>(other, all);
    }
    
    template<general_vector_like EXPR>
    vector_view<const typename EXPR::value_type> slice(const EXPR& other, detail::all_t) {
        return vector_view<const typename EXPR::value_type>(other, all);
    }
}

#endif //NUMPP_VECTOR_VIEW_SLICE_HPP