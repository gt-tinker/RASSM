#ifndef SIMPLE_H
#define SIMPLE_H

#include "matrices/CSR.h"
#include "matrices/Matrix.h"

#include "utils/util.h"

#include <omp.h>

// Simple single core sparse-dense matrix multiplication implementation
template<typename T, typename ITYPE>
void spmm_simple(CSR<T, ITYPE> &S, Matrix<T, ITYPE> &I, Matrix<T, ITYPE> &O)
{
    ITYPE feature = I.ncols;
    T *I_data = I._data;
    T *O_data = O._data;

    for (ITYPE i = 0; i < S.nrows; i++) {
        ITYPE row_start = S.row_ptr[i];
        ITYPE row_end = S.row_ptr[i + 1];

        for (ITYPE ptr = row_start; ptr < row_end; ptr++) {
            ITYPE j = S.cols[ptr];
            for (ITYPE k = 0; k < feature; k++) {
                O_data[i * feature + k] += S.vals[ptr] * I_data[j * feature + k];
            }
        }
    }
}

template<typename T, typename ITYPE>
void spmm_simple_parallel(CSR<T, ITYPE> &S, Matrix<T, ITYPE> &I, Matrix<T, ITYPE> &O, ITYPE chunk_size = 1)
{
    ITYPE feature = I.ncols;
    ITYPE num_rows = S.nrows;
    T *I_data = I._data;
    T *O_data = O._data;

    #pragma omp parallel for schedule(static, chunk_size)
    for (ITYPE i = 0; i < num_rows; i++) {
        ITYPE row_start = S.row_ptr[i];
        ITYPE row_end = S.row_ptr[i + 1];

        for (ITYPE ptr = row_start; ptr < row_end; ptr++) {
            ITYPE j = S.cols[ptr];
            for (ITYPE k = 0; k < feature; k++) {
                O_data[i * feature + k] += S.vals[ptr] * I_data[j * feature + k];
            }
        }
    }

}

template<typename T, typename ITYPE>
void spmm_simple(CSR<T, ITYPE> &S, T *I, T *O, ITYPE feature)
{
    for ( ITYPE row = 0; row < S.nrows; row++ ) {
        ITYPE row_start = S.row_ptr[row];
        ITYPE row_end = S.row_ptr[row + 1];
        for ( ITYPE ptr = row_start; ptr < row_end; ptr++ ) {
            ITYPE col = S.cols[ptr];
            for ( ITYPE k = 0; k < feature; k++ ) {
                O[ row * (feature + PADDING_C) + k ] += I[col * (feature + PADDING_B) + k] * S.vals[ptr];
            }
        }
    }
}

template<typename T, typename ITYPE>
void spmm_simple_parallel(CSR<T, ITYPE> &S, T *I, T *O, ITYPE feature, ITYPE chunk_size = 1)
{
    ITYPE num_rows = S.nrows;

    #pragma omp parallel for schedule(static, chunk_size)
    for (ITYPE i = 0; i < num_rows; i++) {
        ITYPE row_start = S.row_ptr[i];
        ITYPE row_end = S.row_ptr[i + 1];

        for (ITYPE ptr = row_start; ptr < row_end; ptr++) {
            ITYPE j = S.cols[ptr];
            for (ITYPE k = 0; k < feature; k++) {
                O[i * (feature + PADDING_C) + k] += S.vals[ptr] * I[j * (feature + PADDING_B) + k];
            }
        }
    }
}

#define PRINT_DIFFERENCE
template<typename T, typename ITYPE>
bool check_simple(CSR<T, ITYPE> &S, T *I, T *O, ITYPE feature, T *to_check, bool early_terminate = true)
{
    spmm_simple(S, I, O, feature);

    print_status("Checking against output of spmm_simple\n");

    ITYPE num_differences = 0;

    for ( ITYPE i = 0; i < S.nrows; i++ ) {
        for ( ITYPE j = 0; j < feature; j++ ) {
            ITYPE index = (i * (feature + PADDING_C)) + j;
            T a = O[index] < 0 ? -1 * O[index] : O[index];
            T b = to_check[index] < 0 ? -1 * to_check[index] : to_check[index];

            T diff = ((a - b) < 0 ? (b - a) : (a - b));

            diff = diff / MAX(a, b);

            if ( (std::is_same<T, float>::value && diff > 0.02) || (std::is_same<T, double>::value && diff > 0.02) ) {
                #ifdef PRINT_DIFFERENCE
                    // std::cerr << "Difference at (" << i / feature << "," << i % feature << ") " << O[index] << " " << to_check[index] << std::endl;
                    std::cerr << "Difference at (" << i << "," << j << ") " << O[index] << " " << to_check[index] << std::endl;
                #endif
                num_differences++;

                if (early_terminate) {
                    return false;
                }
            }
            // assert( num_differences == 0 && "FAILURE : Difference in output" );
        }
    }

    return (num_differences == 0);
}

template <typename T, typename ITYPE>
void spmm_simple_vectorized_parallel( CSR<T, ITYPE> &S, Matrix<T, ITYPE> &I, Matrix<T, ITYPE> &O, ITYPE chunk_size = 1 )
{

    ITYPE nrows = S.get_nrows();
    ITYPE feature = I.ncols;

    T* csr_v = S.vals;
    ITYPE *csr_c = S.cols;
    ITYPE *csr_r = S.row_ptr;
    T *I_data = I._data;
    T *O_data = O._data;

    #ifdef INTEL_COMPILER
        __assume_aligned(csr_v, 64);
        __assume_aligned(csr_c, 64);
        __assume_aligned(csr_r, 64);
        __assume_aligned(I_data, 64);
        __assume_aligned(O_data, 64);
    #endif


    #pragma ivdep
    #pragma vector aligned
    #pragma omp parallel for schedule(static, chunk_size)
    for ( ITYPE row = 0; row < nrows; row++ ) {
        ITYPE row_start = csr_r[row];
        ITYPE row_end = csr_r[row + 1];
        for ( ITYPE r = row_start; r < row_end; r++ ) {
            T* O_base_addr = O_data + (row * feature);
            ITYPE col = csr_c[r];
            T val = csr_v[r];

            rtype Areg = vset( val );
            T *I_base_addr = I_data + (col * feature);

            // Inner loop streaming over the I and O arrays -- vectorized in the K dimension with AVX2
            for (ITYPE kk = 0; kk < feature; kk += 32) {

                rtype Breg1 = vload( I_base_addr + kk + 0 );
                rtype Breg2 = vload( I_base_addr + kk + 4 );
                rtype Breg3 = vload( I_base_addr + kk + 8 );
                rtype Breg4 = vload( I_base_addr + kk + 12 );
                rtype Breg5 = vload( I_base_addr + kk + 16 );
                rtype Breg6 = vload( I_base_addr + kk + 20 );
                rtype Breg7 = vload( I_base_addr + kk + 24 );
                rtype Breg8 = vload( I_base_addr + kk + 28 );

                rtype Creg1 = vload( O_base_addr + kk + 0 );
                Creg1 = vfma( Areg, Breg1, Creg1 );
                vstore( O_base_addr + kk + 0, Creg1 );

                rtype Creg2 = vload( O_base_addr + kk + 4 );
                Creg2 = vfma( Areg, Breg2, Creg2 );
                vstore( O_base_addr + kk + 4, Creg2 );

                rtype Creg3 = vload( O_base_addr + kk + 8 );
                Creg3 = vfma( Areg, Breg3, Creg3 );
                vstore( O_base_addr + kk + 8, Creg3 );

                rtype Creg4 = vload( O_base_addr + kk + 12 );
                Creg4 = vfma( Areg, Breg4, Creg4 );
                vstore( O_base_addr + kk + 12, Creg4 );

                // reuse C registers since we only have 16 and have used 13
                Creg1 = vload( O_base_addr + kk + 16 );
                Creg1 = vfma( Areg, Breg5, Creg1 );
                vstore( O_base_addr + kk + 16, Creg1 );

                Creg2 = vload( O_base_addr + kk + 20 );
                Creg2 = vfma( Areg, Breg6, Creg2 );
                vstore( O_base_addr + kk + 20, Creg2 );

                Creg3 = vload( O_base_addr + kk + 24 );
                Creg3 = vfma( Areg, Breg7, Creg3 );
                vstore( O_base_addr + kk + 24, Creg3 );

                Creg4 = vload( O_base_addr + kk + 28 );
                Creg4 = vfma( Areg, Breg8, Creg4 );
                vstore( O_base_addr + kk + 28, Creg4 );
            }
        }
    }
}


template <typename T, typename ITYPE>
long long spmm_simple_vectorized_parallel( CSR<T, ITYPE> &S, Matrix<T, ITYPE> &I, Matrix<T, ITYPE> &O, ITYPE chunk_size = 1 )
{

    ITYPE nrows = S.get_nrows();
    ITYPE feature = I.ncols;

    T* csr_v = S.vals;
    ITYPE *csr_c = S.cols;
    ITYPE *csr_r = S.row_ptr;
    T *I_data = I._data;
    T *O_data = O._data;

    #ifdef INTEL_COMPILER
        __assume_aligned(csr_v, 64);
        __assume_aligned(csr_c, 64);
        __assume_aligned(csr_r, 64);
        __assume_aligned(I_data, 64);
        __assume_aligned(O_data, 64);
    #endif

    long long start_cycle = readTSC();

    #pragma ivdep
    #pragma vector aligned
    #pragma omp parallel for schedule(OMP_SCHEDULE, chunk_size)
    for ( ITYPE row = 0; row < nrows; row++ ) {
        ITYPE row_start = csr_r[row];
        ITYPE row_end = csr_r[row + 1];
        for ( ITYPE r = row_start; r < row_end; r++ ) {
            T* O_base_addr = O_data + (row * feature);
            ITYPE col = csr_c[r];
            T val = csr_v[r];

            rtype Areg = vset( val );
            T *I_base_addr = I_data + (col * feature);

            // Inner loop streaming over the I and O arrays -- vectorized in the K dimension with AVX2
            for (ITYPE kk = 0; kk < feature; kk += 32) {

                rtype Breg1 = vload( I_base_addr + kk + 0 );
                rtype Breg2 = vload( I_base_addr + kk + 4 );
                rtype Breg3 = vload( I_base_addr + kk + 8 );
                rtype Breg4 = vload( I_base_addr + kk + 12 );
                rtype Breg5 = vload( I_base_addr + kk + 16 );
                rtype Breg6 = vload( I_base_addr + kk + 20 );
                rtype Breg7 = vload( I_base_addr + kk + 24 );
                rtype Breg8 = vload( I_base_addr + kk + 28 );

                rtype Creg1 = vload( O_base_addr + kk + 0 );
                Creg1 = vfma( Areg, Breg1, Creg1 );
                vstore( O_base_addr + kk + 0, Creg1 );

                rtype Creg2 = vload( O_base_addr + kk + 4 );
                Creg2 = vfma( Areg, Breg2, Creg2 );
                vstore( O_base_addr + kk + 4, Creg2 );

                rtype Creg3 = vload( O_base_addr + kk + 8 );
                Creg3 = vfma( Areg, Breg3, Creg3 );
                vstore( O_base_addr + kk + 8, Creg3 );

                rtype Creg4 = vload( O_base_addr + kk + 12 );
                Creg4 = vfma( Areg, Breg4, Creg4 );
                vstore( O_base_addr + kk + 12, Creg4 );

                // reuse C registers since we only have 16 and have used 13
                Creg1 = vload( O_base_addr + kk + 16 );
                Creg1 = vfma( Areg, Breg5, Creg1 );
                vstore( O_base_addr + kk + 16, Creg1 );

                Creg2 = vload( O_base_addr + kk + 20 );
                Creg2 = vfma( Areg, Breg6, Creg2 );
                vstore( O_base_addr + kk + 20, Creg2 );

                Creg3 = vload( O_base_addr + kk + 24 );
                Creg3 = vfma( Areg, Breg7, Creg3 );
                vstore( O_base_addr + kk + 24, Creg3 );

                Creg4 = vload( O_base_addr + kk + 28 );
                Creg4 = vfma( Areg, Breg8, Creg4 );
                vstore( O_base_addr + kk + 28, Creg4 );
            }
        }
    }

    long long end_cycle = readTSC();

    return (end_cycle - start_cycle);
}

template <typename T, typename ITYPE>
long long spmm_simple_parallel_compiler_vectorized( CSR<T, ITYPE> &S, T *I, T *O, ITYPE feature, ITYPE Ti = 1, ITYPE Tk = 1, ITYPE chunk_size = 1 )
{
    ITYPE nrows = S.nrows;
    T* csr_v = S.vals;
    ITYPE *csr_c = S.cols;
    ITYPE *csr_r = S.row_ptr;
    T *I_data = I;
    T *O_data = O;

    #ifdef INTEL_COMPILER
        __assume_aligned(csr_v, 64);
        __assume_aligned(csr_c, 64);
        __assume_aligned(csr_r, 64);
        __assume_aligned(I_data, 64);
        __assume_aligned(O_data, 64);
    #endif


    long long start_cycle = readTSC();

    #pragma ivdep
    #pragma vector aligned
    #pragma omp parallel for schedule(OMP_SCHEDULE, chunk_size)
    for (ITYPE i = 0; i < nrows; i++) {
        ITYPE row_start = csr_r[i];
        ITYPE row_end = csr_r[i + 1];

        for (ITYPE ptr = row_start; ptr < row_end; ptr++) {
            ITYPE j = csr_c[ptr];
            for (ITYPE k = 0; k < feature; k++) {
                O_data[i * feature + k] += csr_v[ptr] * I_data[j * feature + k];
            }
        }
    }

    long long end_cycle = readTSC();

    return (end_cycle - start_cycle);
}


// Ti & Tk are just passed in for the sake of it -- not used in this function
template <typename T, typename ITYPE>
long long spmm_simple_vectorized_parallel( CSR<T, ITYPE> &S, T *I, T *O, ITYPE feature, ITYPE Ti = 1, ITYPE Tk = 1, ITYPE chunk_size = 1 )
{

    ITYPE nrows = S.get_nrows();

    T* csr_v = S.vals;
    ITYPE *csr_c = S.cols;
    ITYPE *csr_r = S.row_ptr;
    T *I_data = I;
    T *O_data = O;

    #ifdef INTEL_COMPILER
        __assume_aligned(csr_v, 64);
        __assume_aligned(csr_c, 64);
        __assume_aligned(csr_r, 64);
        __assume_aligned(I_data, 64);
        __assume_aligned(O_data, 64);
    #endif

    long long start_cycle = readTSC();

    #pragma ivdep
    #pragma vector aligned
    #pragma omp parallel for schedule(OMP_SCHEDULE, chunk_size)
    for ( ITYPE row = 0; row < nrows; row++ ) {
        ITYPE row_start = csr_r[row];
        ITYPE row_end = csr_r[row + 1];
        for ( ITYPE r = row_start; r < row_end; r++ ) {
            T* O_base_addr = O_data + (row * feature);
            ITYPE col = csr_c[r];
            T val = csr_v[r];

            rtype Areg = vset( val );
            T *I_base_addr = I_data + (col * feature);

            // Inner loop streaming over the I and O arrays -- vectorized in the K dimension with AVX2
            for (ITYPE kk = 0; kk < feature; kk += 32) {

                rtype Breg1 = vload( I_base_addr + kk + 0 );
                rtype Breg2 = vload( I_base_addr + kk + 4 );
                rtype Breg3 = vload( I_base_addr + kk + 8 );
                rtype Breg4 = vload( I_base_addr + kk + 12 );
                rtype Breg5 = vload( I_base_addr + kk + 16 );
                rtype Breg6 = vload( I_base_addr + kk + 20 );
                rtype Breg7 = vload( I_base_addr + kk + 24 );
                rtype Breg8 = vload( I_base_addr + kk + 28 );

                rtype Creg1 = vload( O_base_addr + kk + 0 );
                Creg1 = vfma( Areg, Breg1, Creg1 );
                vstore( O_base_addr + kk + 0, Creg1 );

                rtype Creg2 = vload( O_base_addr + kk + 4 );
                Creg2 = vfma( Areg, Breg2, Creg2 );
                vstore( O_base_addr + kk + 4, Creg2 );

                rtype Creg3 = vload( O_base_addr + kk + 8 );
                Creg3 = vfma( Areg, Breg3, Creg3 );
                vstore( O_base_addr + kk + 8, Creg3 );

                rtype Creg4 = vload( O_base_addr + kk + 12 );
                Creg4 = vfma( Areg, Breg4, Creg4 );
                vstore( O_base_addr + kk + 12, Creg4 );

                // reuse C registers since we only have 16 and have used 13
                Creg1 = vload( O_base_addr + kk + 16 );
                Creg1 = vfma( Areg, Breg5, Creg1 );
                vstore( O_base_addr + kk + 16, Creg1 );

                Creg2 = vload( O_base_addr + kk + 20 );
                Creg2 = vfma( Areg, Breg6, Creg2 );
                vstore( O_base_addr + kk + 20, Creg2 );

                Creg3 = vload( O_base_addr + kk + 24 );
                Creg3 = vfma( Areg, Breg7, Creg3 );
                vstore( O_base_addr + kk + 24, Creg3 );

                Creg4 = vload( O_base_addr + kk + 28 );
                Creg4 = vfma( Areg, Breg8, Creg4 );
                vstore( O_base_addr + kk + 28, Creg4 );
            }
        }
    }

    long long end_cycle = readTSC();

    return (end_cycle - start_cycle);
}


#endif // SIMPLE_H
