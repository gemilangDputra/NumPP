#ifndef NUMPP_VECTOR_CONSTRUCTOR_HPP
#define NUMPP_VECTOR_CONSTRUCTOR_HPP

#include <numpp/vector/forward.hpp>
#include <numpp/vector/tool.hpp>
#include <numpp/vector/vector.hpp>

#include <stdexcept>
#include <initializer_list>
#include <random>

namespace numpp {
    template<typename T>
    vector<T> vector<T>::empty(size_t size) {
        vector<T> out;

        out.size_ = size;
        out.stride_ = 1;
        out.offset_ = 0;
        out.alloc();
        return out;
    }
    
    template<typename T>
    vector<T> vector<T>::zeros(size_t size) {
        vector<T> out;

        out.size_ = size;
        out.stride_ = 1;
        out.offset_ = 0;
        out.alloc(true);
        return out;
    }
    
    template<typename T>
    vector<T> vector<T>::ones(size_t size) {
        vector<T> out;

        out.size_ = size;
        out.stride_ = 1;
        out.offset_ = 0;
        out.alloc();
        for (size_t i = 0; i < size; ++i) out.data_[i] = T{1};
        return out;
    }
    
    template<typename T>
    vector<T> vector<T>::full(size_t size, const T& value) {
        vector<T> out;

        out.size_ = size;
        out.stride_ = 1;
        out.offset_ = 0;
        out.alloc();
        for (size_t i = 0; i < size; ++i) out.data_[i] = value;
        return out;
    }
    
    template<typename T>
    template<RandomEngine RNG>
    vector<T> vector<T>::rand(size_t size, const T& min, const T& max, RNG& rng) {
        vector<T> out;

        out.size_ = size;
        out.stride_ = 1;
        out.offset_ = 0;
        out.alloc();

        std::uniform_real_distribution<T> dist(min, max);
        for (size_t i = 0; i < size; ++i) out.data_[i] = dist(rng);
        return out;
    }
    
    template<typename T>
    template<RandomEngine RNG>
    vector<T> vector<T>::randint(size_t size, int64_t min, int64_t max, RNG& rng) {
        vector<T> out;

        out.size_ = size;
        out.stride_ = 1;
        out.offset_ = 0;
        out.alloc();

        std::uniform_int_distribution<int64_t> dist(min, max);
        for (size_t i = 0; i < size; ++i) out.data_[i] = static_cast<T>(dist(rng));
        return out;
    }
    
    template<typename T>
    template<RandomEngine RNG>
    vector<T> vector<T>::randn(size_t size, const T& mean, const T& stddev, RNG& rng) {
        vector<T> out;

        out.size_ = size;
        out.stride_ = 1;
        out.offset_ = 0;
        out.alloc();

        std::normal_distribution<T> dist(mean, stddev);
        for (size_t i = 0; i < size; ++i) out.data_[i] = dist(rng);
        return out;
    }

    template<typename T>
    vector<T> vector<T>::rand(size_t size, const T& min, const T& max) {
        std::mt19937 rng(std::random_device{}());
        return rand(size, min, max, rng);
    }
    
    template<typename T>
    vector<T> vector<T>::randint(size_t size, int64_t min, int64_t max) {
        std::mt19937 rng(std::random_device{}());
        return randint(size, min, max, rng);
    }
    
    template<typename T>
    vector<T> vector<T>::randn(size_t size, const T& mean, const T& stddev) {
        std::mt19937 rng(std::random_device{}());
        return randn(size, mean, stddev, rng);
    }
    
    template<typename T>
    template<general_vector_like EXPR>
    vector<T> vector<T>::empty_like(const EXPR& other) {
        vector<T> out;

        out.size_ = other.size();
        out.stride_ = 1;
        out.offset_ = 0;
        out.alloc();
        return out;
    }
    
    template<typename T>
    template<general_vector_like EXPR>
    vector<T> vector<T>::zeros_like(const EXPR& other) {
        vector<T> out;

        out.size_ = other.size();
        out.stride_ = 1;
        out.offset_ = 0;
        out.alloc(true);
        return out;
    }
    
    template<typename T>
    template<general_vector_like EXPR>
    vector<T> vector<T>::full_like(const EXPR& other, const T& value) {
        vector<T> out;

        out.size_ = other.size();
        out.stride_ = 1;
        out.offset_ = 0;
        out.alloc();
        for (size_t i = 0; i < out.size_; ++i) out.data_[i] = value;
        return out;
    }
    
    template<typename T>
    template<general_vector_like EXPR, RandomEngine RNG>
    vector<T> vector<T>::rand_like(const EXPR& other, const T& min, const T& max, RNG& rng) {
        vector<T> out;

        out.size_ = other.size();
        out.stride_ = 1;
        out.offset_ = 0;
        out.alloc();

        std::uniform_real_distribution<T> dist(min, max);
        for (size_t i = 0; i < out.size_; ++i) out.data_[i] = dist(rng);
        return out;
    }
    
    template<typename T>
    template<general_vector_like EXPR, RandomEngine RNG>
    vector<T> vector<T>::randint_like(const EXPR& other, int64_t min, int64_t max, RNG& rng) {
        vector<T> out;

        out.size_ = other.size();
        out.stride_ = 1;
        out.offset_ = 0;
        out.alloc();

        std::uniform_int_distribution<int64_t> dist(min, max);
        for (size_t i = 0; i < out.size_; ++i) out.data_[i] = static_cast<float>(dist(rng));
        return out;
    }
    
    template<typename T>
    template<general_vector_like EXPR, RandomEngine RNG>
    vector<T> vector<T>::randn_like(const EXPR& other, const T& mean, const T& stddev, RNG& rng) {
        vector<T> out;

        out.size_ = other.size();
        out.stride_ = 1;
        out.offset_ = 0;
        out.alloc();

        std::normal_distribution<T> dist(mean, stddev);
        for (size_t i = 0; i < out.size_; ++i) out.data_[i] = dist(rng);
        return out;
    }
    
    template<typename T>
    template<general_vector_like EXPR>
    vector<T> vector<T>::rand(const EXPR& other, const T& min, const T& max) {
        std::mt19937 rng(std::random_device{}());
        return rand_like(other, min, max);
    }
    
    template<typename T>
    template<general_vector_like EXPR>
    vector<T> vector<T>::randint(const EXPR& other, int64_t min, int64_t max) {
        std::mt19937 rng(std::random_device{}());
        return randint_like(other, min, max);
    }
    
    template<typename T>
    template<general_vector_like EXPR>
    vector<T> vector<T>::randn(const EXPR& other, const T& mean, const T& stddev) {
        std::mt19937 rng(std::random_device{}());
        return randn_like(other, mean, stddev);
    }

    template<typename T>
    vector<T>::vector(std::initializer_list<T> list) {
        this->size_ = list.size();
        this->stride_ = 1;
        this->offset_ = 0;
        alloc();

        size_t i = 0;
        for (const auto& value : list) {
            this->data_[i] = value;
            ++i;
        }
    }

    template<typename T>
    vector<T> vector<T>::arange(const T& start, const T& stop, const T& step) {
        if (step <= T{0})
            throw std::invalid_argument("numpp::operation<arange> error: step must be greater than zero");

        std::vector<T> temp;
        for (T value = start; value < stop; value += step)
            temp.push_back(value);
        
        vector<T> out;
        out.size_ = temp.size();
        out.stride_ = 1;
        out.offset_ = 0;
        out.alloc();

        for (size_t i = 0; i < out.size_; ++i)
            out.data_[i] = std::move(temp[i]);
        
        return out;
    }

    template<typename T>
    vector<T> vector<T>::arange(const T& stop) {
        return arange(T{0}, stop, T{1});
    }

    template<typename T>
    vector<T> vector<T>::linspace(const T& start, const T& stop, size_t num, bool endpoint) {
        vector<T> out;
        out.size_ = num;
        out.stride_ = 1;
        out.offset_ = 0;

        if (num == 0) {
            out.data_ = nullptr;
            return out;
        }

        out.alloc();
        if (num == 1) {
            out.data_[0] = start;
            return out;
        }

        const T step = endpoint ? (stop - start) / T{num - 1} : (stop - start) / T{num};
        for (size_t i = 0; i < num; ++i)
            out.data_[i] = start + T{i} * step;
        
        if (endpoint)
            out.data_[num - 1] = stop;
        
        return out;
    }

    template<typename T>
    vector<T> vector<T>::linspace(const T& stop, size_t num, bool endpoint) {
        return linspace(T{0}, stop, num, endpoint);
    }

    template<typename T>
    template<general_vector_like EXPR>
    vector<T>::vector(const EXPR& other) {
        this->size_ = other.size();
        this->stride_ = 1;
        this->offset_ = 0;
        alloc();
        for (size_t i = 0; i < other.size(); ++i) this->data_[i] = other[i];
    }
    
    template<typename T>
    template<general_vector_like EXPR>
    vector<T>& vector<T>::operator=(const EXPR& other) {
        del();
        this->size_ = other.size();
        this->stride_ = 1;
        this->offset_ = 0;
        alloc();
        for (size_t i = 0; i < other.size(); ++i) this->data_[i] = other[i];
        return *this;
    }
    
    template<typename T>
    vector<T>::vector(const vector<T>& other) {
        this->size_ = other.size_;
        this->stride_ = 1;
        this->offset_ = 0;
        alloc();
        for (size_t i = 0; i < other.size_; ++i) this->data_[i] = other[i];
    }
    
    template<typename T>
    vector<T>& vector<T>::operator=(const vector<T>& other) {
        if (this != &other) {
            del();
            this->size_ = other.size_;
            this->stride_ = 1;
            this->offset_ = 0;
            alloc();
            for (size_t i = 0; i < other.size_; ++i) this->data_[i] = other[i];
        }
        return *this;
    }
    
    template<typename T>
    vector<T>::vector(vector<T>&& other) noexcept {
        this->size_ = other.size();
        this->stride_ = 1;
        this->offset_ = 0;
        this->data_ = other.data_;
        other.data_ = nullptr;
    }
    
    template<typename T>
    vector<T>& vector<T>::operator=(vector<T>&& other) noexcept {
        if (this != &other) {
            this->size_ = other.size();
            this->stride_ = 1;
            this->offset_ = 0;
            this->data_ = other.data_;
            other.data_ = nullptr;
        }
        return *this;
    }
}

#endif //NUMPP_VECTOR_CONSTRUCTOR_HPP