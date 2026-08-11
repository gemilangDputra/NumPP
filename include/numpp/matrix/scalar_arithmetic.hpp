#ifndef NUMPP_MATRIX_SCALAR_ARITHMETIC_HPP
#define NUMPP_MATRIX_SCALAR_ARITHMETIC_HPP

#include <numpp/matrix/core.hpp>
#include <numpp/matrix/tool.hpp>
#include <numpp/matrix/matrix.hpp>
#include <numpp/matrix/view/broadcast.hpp>
#include <utility>
#include <string>

namespace numpp {
    template<matrix_derived EXPR> requires (can_add<typename EXPR::value_type>)
    matrix<typename EXPR::value_type> operator+(const EXPR& mat, const typename EXPR::value_type& scalar) {
        matrix<typename EXPR::value_type> out = matrix<typename EXPR::value_type>::empty_like(mat);
        if (is_contiguous(mat) && mat.offset() == 0) {
            const auto adata = mat.data();
            auto cdata = out.data();
            for (size_t i=0; i < out.size(); ++i) {
                cdata[i] = adata[i] + scalar;
            }
        } else {
            for (size_t i=0; i < mat.row(); ++i) {
                for (size_t j=0; j < mat.col(); ++j) {
                    out(i,j) = mat(i,j) + scalar;
                }
            }
        }
        return out;
    }
    
    template<matrix_derived EXPR> requires (can_sub<typename EXPR::value_type>)
    matrix<typename EXPR::value_type> operator-(const EXPR& mat, const typename EXPR::value_type& scalar) {
        matrix<typename EXPR::value_type> out = matrix<typename EXPR::value_type>::empty_like(mat);
        if (is_contiguous(mat) && mat.offset() == 0) {
            const auto adata = mat.data();
            auto cdata = out.data();
            for (size_t i=0; i < out.size(); ++i) {
                cdata[i] = adata[i] - scalar;
            }
        } else {
            for (size_t i=0; i < mat.row(); ++i) {
                for (size_t j=0; j < mat.col(); ++j) {
                    out(i,j) = mat(i,j) - scalar;
                }
            }
        }
        return out;
    }
    
    template<matrix_derived EXPR> requires (can_mul<typename EXPR::value_type>)
    matrix<typename EXPR::value_type> operator*(const EXPR& mat, const typename EXPR::value_type& scalar) {
        matrix<typename EXPR::value_type> out = matrix<typename EXPR::value_type>::empty_like(mat);
        if (is_contiguous(mat) && mat.offset() == 0) {
            const auto adata = mat.data();
            auto cdata = out.data();
            for (size_t i=0; i < out.size(); ++i) {
                cdata[i] = adata[i] * scalar;
            }
        } else {
            for (size_t i=0; i < mat.row(); ++i) {
                for (size_t j=0; j < mat.col(); ++j) {
                    out(i,j) = mat(i,j) * scalar;
                }
            }
        }
        return out;
    }

    
    template<matrix_derived EXPR> requires (can_div<typename EXPR::value_type>)
    matrix<typename EXPR::value_type> operator/(const EXPR& mat, const typename EXPR::value_type& scalar) {
        matrix<typename EXPR::value_type> out = matrix<typename EXPR::value_type>::empty_like(mat);
        if (is_contiguous(mat) && mat.offset() == 0) {
            const auto adata = mat.data();
            auto cdata = out.data();
            for (size_t i=0; i < out.size(); ++i) {
                cdata[i] = adata[i] / scalar;
            }
        } else {
            for (size_t i=0; i < mat.row(); ++i) {
                for (size_t j=0; j < mat.col(); ++j) {
                    out(i,j) = mat(i,j) / scalar;
                }
            }
        }
        return out;
    }

    template<class Derived, typename T>
    matrix_base<Derived, T>& matrix_base<Derived, T>::operator+=(const T& scalar) requires (can_add_assign<T>) {
        if (is_contiguous(derived()) && offset_ == 0) {
            for (size_t i=0; i < size_; ++i) {
                data_[i] += scalar;
            }
        } else {
            for (size_t i=0; i < row_; ++i) {
                for (size_t j=0; j < col_; ++j) {
                    (*this)(i,j) += scalar;
                }
            }
        }
        return *this;
    }

    template<class Derived, typename T>
    matrix_base<Derived, T>& matrix_base<Derived, T>::operator-=(const T& scalar) requires (can_sub_assign<T>) {
        if (is_contiguous(derived()) && offset_ == 0) {
            for (size_t i=0; i < size_; ++i) {
                data_[i] -= scalar;
            }
        } else {
            for (size_t i=0; i < row_; ++i) {
                for (size_t j=0; j < col_; ++j) {
                    (*this)(i,j) -= scalar;
                }
            }
        }
        return *this;
    }
    
    template<class Derived, typename T>
    matrix_base<Derived, T>& matrix_base<Derived, T>::operator*=(const T& scalar) requires (can_mul_assign<T>) {
        if (is_contiguous(derived()) && offset_ == 0) {
            for (size_t i=0; i < size_; ++i) {
                data_[i] *= scalar;
            }
        } else {
            for (size_t i=0; i < row_; ++i) {
                for (size_t j=0; j < col_; ++j) {
                    (*this)(i,j) *= scalar;
                }
            }
        }
        return *this;
    }

    template<class Derived, typename T>
    matrix_base<Derived, T>& matrix_base<Derived, T>::operator/=(const T& scalar) requires (can_div_assign<T>) {
        if (is_contiguous(derived()) && offset_ == 0) {
            for (size_t i=0; i < size_; ++i) {
                data_[i] /= scalar;
            }
        } else {
            for (size_t i=0; i < row_; ++i) {
                for (size_t j=0; j < col_; ++j) {
                    (*this)(i,j) /= scalar;
                }
            }
        }
        return *this;
    }
}

#endif //NUMPP_MATRIX_SCALAR_ARITHMETIC_HPP