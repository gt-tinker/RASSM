#ifndef MATRIX_H
#define MATRIX_H

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>

#include "utils/util.h"

template<typename T = double, typename ITYPE = int64_t>
class Matrix {
public:
    T *_data;
    ITYPE nrows;
    ITYPE ncols;

    Matrix() : _data(nullptr), nrows(0), ncols(0) { }

    ~Matrix() {
        if (_data != nullptr) {
            std::free(_data);
        }
    }

    Matrix(ITYPE nrows, ITYPE ncols) : nrows(nrows), ncols(ncols) {
        size_t bytes_allocated = (nrows * ncols * sizeof(T)) + (nrows * ncols * sizeof(T)) % ALIGNED_ACCESS;

        _data = (T *) std::aligned_alloc( ALIGNED_ACCESS, bytes_allocated );

        std::memset( _data, 0, (nrows * ncols * sizeof(T)) );
    }

    void init(ITYPE nrows, ITYPE ncols) {
        this->nrows = nrows;
        this->ncols = ncols;

        size_t bytes_allocated = (nrows * ncols * sizeof(T)) + (nrows * ncols * sizeof(T)) % ALIGNED_ACCESS;

        _data = (T *) std::aligned_alloc( ALIGNED_ACCESS, bytes_allocated );

        std::memset( _data, 0, (nrows * ncols * sizeof(T)) );
    }

    void free() {
        if (_data) {
            std::free(_data);
        }
        _data = nullptr;
    }

    inline T& operator() (ITYPE r, ITYPE c) {
        return _data[ r * ncols + c ];
    }

    T& get_element(ITYPE r, ITYPE c) { return _data[r * ncols + c]; }
    void set_element(T val, ITYPE r, ITYPE c) { _data[r * ncols + c] = val; }
    ITYPE get_addr(ITYPE r, ITYPE c) { return (ITYPE) ( &_data[r * ncols + c] ) ; }
    T* get_addr_raw(ITYPE r, ITYPE c) { return (&_data[r * ncols + c]); }


    void print_matrix();
    void reset_matrix();
    void reorder(Matrix<T, ITYPE> m, ITYPE *order);
};

template<typename T, typename ITYPE>
void Matrix<T, ITYPE>::reset_matrix()
{
    if (_data) {
        std::memset(_data, 0, sizeof(T) * nrows * ncols);
    }
}

template<typename T, typename ITYPE>
void Matrix<T, ITYPE>::print_matrix()
{
    std::cerr << std::endl;

    for (ITYPE r = 0; r < nrows; r++) {
        for (ITYPE c = 0; c < ncols; c++) {
            std::cerr << get_element(r, c) << " ";
        }
        std::cerr << std::endl;
    }

    std::cerr << std::endl;
}

template <typename T, typename ITYPE>
void Matrix<T, ITYPE>::reorder(Matrix<T, ITYPE> m, ITYPE *order)
{
    ITYPE feature = m.ncols;
    for ( ITYPE r = 0; r < m.nrows; r++ ) {
        for ( ITYPE c = 0; c < feature; c++ ) {
            this->_data[ (order[r] * feature) + c ] = m._data[ (r * feature) + c ];
        }
    }
}

#endif // MATRIX_H