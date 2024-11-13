#ifndef CSC_H
#define CSC_H

#include <cstdlib>
#include <iostream>
#include <memory>

#include "utils/util.h"
#include "config.h"

template<typename T = uint64_t, typename ITYPE = size_t>
class CSC {

public:
    ITYPE nrows;    // number of rows
    ITYPE ncols;    // number of columns
    ITYPE nnzs;     // number of non-zeroes

    T* vals;
    ITYPE *col_ptr;
    ITYPE *rows;

    ITYPE get_nrows() const { return nrows; }
    ITYPE get_ncols() const { return ncols; }
    ITYPE get_nnzs() const { return nnzs; }

    CSC() : nrows(0), ncols(0), nnzs(0) {}
    ~CSC() {
        if (vals) { std::free(vals); }
        if (col_ptr) { std::free(col_ptr); }
        if (rows) { std::free(rows); }

    }
    CSC(ITYPE nr, ITYPE nc, ITYPE nz, std::pair<ITYPE, ITYPE> *pairs, T *vals);

    void print_matrix();

    size_t get_size_bytes() {
        return sizeof(ITYPE) * (nrows + 1) + sizeof(ITYPE) * (nnzs) + sizeof(T) * (nnzs);
    }
};

template<typename T, typename ITYPE>
CSC<T, ITYPE>::CSC(ITYPE nr, ITYPE nc, ITYPE nz, std::pair<ITYPE, ITYPE> *pairs, T *vals)
{
    ITYPE nnz = nz;

    // Copy in to temp structure for the current time being
    struct v_struct *temp = new struct v_struct[nz];
    for (ITYPE i = 0; i < nnz; i++) {
        temp[i].row = pairs[i].first;
        temp[i].col = pairs[i].second;
        temp[i].grp = 0;
        temp[i].val = vals[i];
    }

    this->nnzs = nnz;
    this->nrows = nr;
    this->ncols = nc;

    // Sort in column major order
    std::qsort(temp, nnzs, sizeof(struct v_struct), compare2);

    this->vals = (T*) std::aligned_alloc(ALLOC_ALIGNMENT, nnz * sizeof(T));
    std::memset(vals, 0, nnz * sizeof(T));
    this->rows = (ITYPE*) std::aligned_alloc(ALLOC_ALIGNMENT, nnz * sizeof(ITYPE));
    std::memset(rows, 0, nnz * sizeof(ITYPE));
    this->col_ptr = (ITYPE *) std::aligned_alloc(ALLOC_ALIGNMENT, (ncols + 1) * sizeof(ITYPE));
    std::memset(col_ptr, 0, (ncols + 1) * sizeof(ITYPE));

    for (ITYPE i = 0; i < nnz; i++) {
        col_ptr[ temp[i].col + 1 ]++;
        rows[i] = temp[i].row;
        vals[i] = temp[i].val;
    }

    // setup col_ptr values
    for (ITYPE i = 1; i < ncols + 1; i++) {
        col_ptr[i] += col_ptr[i - 1];
    }

    delete[] temp;
}

#endif // CSC_H
