#ifndef SDDMM_SIMPLE_H
#define SDDMM_SIMPLE_H

#include "matrices/CSR.h"

template <typename T, typename ITYPE>
long long sddmm_simple(CSR<T, ITYPE> &S, T *D1, T *D2, T *O, ITYPE K)
{
    long long start_cycle = readTSC();
    // First loop does the dot product of the dense matrices
    for ( ITYPE i = 0; i < S.nrows; i++ ) {
        ITYPE row_start = S.row_ptr[i];
        ITYPE row_end = S.row_ptr[i + 1];

        for ( ITYPE ptr = row_start; ptr < row_end; ptr++ ) {
            for ( ITYPE k = 0; k < K; k++ ) {
                O[ptr] += D1[ (S.cols[ptr] * K) + k] * D2[i * K + k];
            }
            O[ptr] *= S.vals[ptr];
        }
    }

    long long end_cycle = readTSC();
    return (end_cycle - start_cycle);
}

template <typename T, typename ITYPE>
long long sddmm_simple_parallel(CSR<T, ITYPE> &S, T *D1, T *D2, CSR<T, ITYPE> &O, ITYPE K, ITYPE chunk_size)
{
    long long start_cycle = readTSC();

    // First loop does the dot product of the dense matrices
    #pragma omp parallel for schedule(OMP_SCHEDULE, chunk_size)
    for ( ITYPE i = 0; i < S.nrows; i++ ) {
        ITYPE row_start = S.row_ptr[i];
        ITYPE row_end = S.row_ptr[i + 1];

        for ( ITYPE ptr = row_start; ptr < row_end; ptr++ ) {
            for ( ITYPE k = 0; k < K; k++ ) {
                O.vals[ptr] += D1[i * K + k] * D2[ (S.cols[ptr] * K) + k];
            }
        }
    }

    // Second loop does the scaling by the sparse input
    #pragma omp parallel for schedule(OMP_SCHEDULE, chunk_size)
    for ( ITYPE i = 0; i < S.nrows; i++ ) {
        ITYPE row_start = S.row_ptr[i];
        ITYPE row_end = S.row_ptr[i + 1];
        for ( ITYPE ptr = row_start; ptr < row_end; ptr++ ) {
            O.vals[ptr] *= S.vals[ptr];
        }
    }

    long long end_cycle = readTSC();
    return (end_cycle - start_cycle);
}

#endif // SDMM_SIMPLE_H

