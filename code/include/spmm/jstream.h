#ifndef JSTREAM_H
#define JSTREAM_H

#include "config.h"
#include "matrices/CSR.h"
#include "matrices/DCSC.h"
#include "matrices/Matrix.h"
#include "utils/util.h"

#include <omp.h>

#ifdef PER_PANEL_PERF_EVENTS
    #include "Perf.h"
#endif

#include "utils/util.h"


template <typename T, typename ITYPE>
void spmm_csr_j_stream( CSR<T, ITYPE> &S, Matrix<T, ITYPE> &I, Matrix<T, ITYPE> &O, ITYPE Ti, ITYPE Tk, ITYPE chunk_size = 1 )
{
    ITYPE num_row_panels = CEIL( S.get_nrows(), Ti );
    ITYPE num_threads = omp_get_num_threads();

    T* csr_v = S.vals;
    ITYPE *csr_c = S.cols;
    ITYPE *csr_ptr = S.row_ptr;
    T *I_data = I._data;
    T *O_data = O._data;

    __assume_aligned(csr_v, 64);
    __assume_aligned(csr_c, 64);
    __assume_aligned(csr_ptr, 64);
    __assume_aligned(I_data, 64);
    __assume_aligned(O_data, 64);


    ITYPE num_rows = S.nrows;
    ITYPE feature = I.ncols;

    #pragma ivdep
    #pragma vector aligned
    #pragma omp parallel for schedule(static, chunk_size)
    for ( ITYPE row_panel = 0; row_panel < num_row_panels; row_panel++ ) {

        for ( ITYPE k = 0; k < I.ncols; k += Tk ) {

            ITYPE row_start = row_panel * Ti;

            for ( ITYPE row = row_start; row < MIN(row_start + Ti, num_rows); row++ ) {
                ITYPE row_begin = csr_ptr[row];
                ITYPE row_end = csr_ptr[row + 1];

                T* O_base_addr = &O_data[row * feature];
                for ( ITYPE r = row_begin; r < row_end; r++ ) {

                    ITYPE col = csr_c[r];
                    T val = csr_v[r];
                    rtype Areg = vset( val );

                    T* I_base_addr = &I_data[col * feature];
                    // K loops vetorized with avx2 instructions
                    for ( ITYPE kk = k; kk < MIN(feature, k + Tk); kk += 32 ) {

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

                        // reuse C

                        #ifdef REUSE_C

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

                        #else // REUSE_C

                            rtype Creg5 = vload( O_base_addr + kk + 16 );
                            Creg5 = vfma( Areg, Breg5, Creg5 );
                            vstore( O_base_addr + kk + 16, Creg5 );

                            rtype Creg6 = vload( O_base_addr + kk + 20 );
                            Creg6 = vfma( Areg, Breg6, Creg6 );
                            vstore( O_base_addr + kk + 20, Creg6 );

                            rtype Creg7 = vload( O_base_addr + kk + 24 );
                            Creg7 = vfma( Areg, Breg7, Creg7 );
                            vstore( O_base_addr + kk + 24, Creg7 );

                            rtype Creg8 = vload( O_base_addr + kk + 28 );
                            Creg8 = vfma( Areg, Breg8, Creg8 );
                            vstore( O_base_addr + kk + 28, Creg8 );

                        #endif // REUSE_C
                    }
                }
            }
        }
    }
}

// #define TIMESTAMP_COUNTERS

// #define REUSE_OUTPUT_FROM_REGS

// Function to add perf counters per panel
template <typename T, typename ITYPE>
long long spmm_csr_jstream( CSR<T, ITYPE> &S, T *I, T *O, ITYPE feature, ITYPE Ti, ITYPE Tk, ITYPE chunk_size = 1, long long *per_core_runtime = nullptr)
{
    ITYPE num_row_panels = CEIL( S.get_nrows(), Ti );

    T* csr_v = S.vals;
    ITYPE *csr_c = S.cols;
    ITYPE *csr_ptr = S.row_ptr;
    T *I_data = I;
    T *O_data = O;

    #ifdef INTEL_COMPILER
        __assume_aligned(csr_v, 64);
        __assume_aligned(csr_c, 64);
        __assume_aligned(csr_ptr, 64);
        __assume_aligned(I_data, 64);
        __assume_aligned(O_data, 64);
    #endif

    ITYPE num_rows = S.nrows;

    long long start_cycle = readTSC();

    #ifdef TRACK_PER_CORE_RUNTIME
    #pragma omp parallel shared(per_core_runtime, csr_v, csr_c, csr_ptr, I_data, O_data, Ti, chunk_size, Tk, feature)
    {
        auto my_tid = omp_get_thread_num();
        per_core_runtime[my_tid] = readTSC();
    #endif

    #pragma ivdep
    #pragma vector aligned
    #ifdef TRACK_PER_CORE_RUNTIME
    #pragma omp for schedule(OMP_SCHEDULE, chunk_size), nowait
    // #pragma omp for schedule(dynamic, chunk_size), nowait
    #else
    #pragma omp parallel for schedule(OMP_SCHEDULE, chunk_size)
    #endif
    // #pragma omp parallel for, nowait
    for ( ITYPE row_panel = 0; row_panel < num_row_panels; row_panel++ ) {

        // TIle / Panel start
        for ( ITYPE k = 0; k < feature; k += Tk ) {

            ITYPE row_start = row_panel * Ti;
            ITYPE row_end = MIN(row_start + Ti, num_rows);

            // Tile start -- Iterate over the row panel
            for ( ITYPE row = row_start; row < row_end; row++ ) {
                ITYPE row_begin = csr_ptr[row];
                ITYPE row_end = csr_ptr[row + 1];

                // Output base register depends on the sparse row index - invariant in below loops
                T* O_base_addr = &O_data[row * (feature + PADDING_C)];

                #ifdef REUSE_OUTPUT_FROM_REGS

                /*
                    // Some funky code to prefetch the entire I row before use
                    for ( ITYPE kk = k; kk < MIN(feature, k + Tk); kk += 32 ) {
                        for ( ITYPE r = row_begin; r < row_end; r++) {
                            ITYPE col = csr_c[r];

                            T *input_ba = &I_data[col * feature];

                            _mm_prefetch( (input_ba + kk + 0), 1 );
                            _mm_prefetch( (input_ba + kk + 8), 1 );
                            _mm_prefetch( (input_ba + kk + 16), 1 );
                            _mm_prefetch( (input_ba + kk + 24), 1 );
                        }
                    }
                */

                // slice is 32 wide in this case
                ITYPE kk_end = MIN(k + Tk, feature);
                for ( ITYPE kk = k; kk < kk_end; kk += 32 ) {

                    // Load the output slice into registers
                    rtype Creg0 = vload( O_base_addr + kk + 0 );
                    rtype Creg1 = vload( O_base_addr + kk + 4 );
                    rtype Creg2 = vload( O_base_addr + kk + 8 );
                    rtype Creg3 = vload( O_base_addr + kk + 12 );
                    rtype Creg4 = vload( O_base_addr + kk + 16 );
                    rtype Creg5 = vload( O_base_addr + kk + 20 );
                    rtype Creg6 = vload( O_base_addr + kk + 24 );
                    rtype Creg7 = vload( O_base_addr + kk + 28 );

                    // Iterate over all non-zeroes of the row
                    for ( ITYPE r = row_begin; r < row_end; r++ ) {
                        ITYPE col = csr_c[r];
                        T val = csr_v[r];
                        rtype Areg = vset( val );

                        // The column index of the non-zero dictates the I base address
                        T *I_base_addr = &I_data[col * feature];

                        /*
                            // We might want to prefetch the slice of the next row
                            T *prefetch_base_addr = &I_data[csr_c[r + 1] * feature] + kk;
                            _mm_prefetch( (prefetch_base_addr + 0), 1 );
                            _mm_prefetch( (prefetch_base_addr + 8), 1 );
                            _mm_prefetch( (prefetch_base_addr + 16), 1 );
                            _mm_prefetch( (prefetch_base_addr + 24), 1 );
                        */

                        // Load the first half of the I slice
                        rtype Breg0 = vload( I_base_addr + kk + 0 );
                        rtype Breg1 = vload( I_base_addr + kk + 4 );
                        rtype Breg2 = vload( I_base_addr + kk + 8 );
                        rtype Breg3 = vload( I_base_addr + kk + 12 );

                        // First set of fused multiply adds performed in place
                        Creg0 = vfma( Areg, Breg0, Creg0 );
                        Creg1 = vfma( Areg, Breg1, Creg1 );
                        Creg2 = vfma( Areg, Breg2, Creg2 );
                        Creg3 = vfma( Areg, Breg3, Creg3 );

                        // Load the second half of the I slice
                        Breg0 = vload( I_base_addr + kk + 16 );
                        Breg1 = vload( I_base_addr + kk + 20 );
                        Breg2 = vload( I_base_addr + kk + 24 );
                        Breg3 = vload( I_base_addr + kk + 28 );


                        // Second set of fused multiply adds performed in place
                        Creg4 = vfma( Areg, Breg0, Creg4 );
                        Creg5 = vfma( Areg, Breg1, Creg5 );
                        Creg6 = vfma( Areg, Breg2, Creg6 );
                        Creg7 = vfma( Areg, Breg3, Creg7 );
                    }

                    // Store slice back into memory before processing the next sparse row
                    vstore( O_base_addr + kk + 0, Creg0 );
                    vstore( O_base_addr + kk + 4, Creg1 );
                    vstore( O_base_addr + kk + 8, Creg2 );
                    vstore( O_base_addr + kk + 12, Creg3 );
                    vstore( O_base_addr + kk + 16, Creg4 );
                    vstore( O_base_addr + kk + 20, Creg5 );
                    vstore( O_base_addr + kk + 24, Creg6 );
                    vstore( O_base_addr + kk + 28, Creg7 );
                }


                #else

                for ( ITYPE r = row_begin; r < row_end; r++ ) {

                    ITYPE col = csr_c[r];
                    T val = csr_v[r];
                    rtype Areg = vset( val );

                    T* I_base_addr = &I_data[col * (feature + PADDING_B)];
                    // K loops vetorized with avx2 instructions
                    for ( ITYPE kk = k; kk < MIN(feature, k + Tk); kk += 32 ) {

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

                        // reuse C
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

                #endif // REUSE_OUTPUT_FROM_REGS
            }   // End processinig row panel for Tk wide Input and Output

        }

        // Tile / Panel end
        #ifdef PER_PANEL_PERF_EVENTS

        #endif
    }

    #ifdef TRACK_PER_CORE_RUNTIME
        per_core_runtime[my_tid] = readTSC() - per_core_runtime[my_tid];

        #pragma omp barrier

    } // end omp parallel region

    // Single thread code -- maybe we should avoid printing from here
    // long long min_time = per_core_runtime[0];
    // for (ITYPE i = 0; i < 20; i++) {
    //     if (per_core_runtime[i] < min_time) {
    //         min_time = per_core_runtime[i];
    //     }
    // }

    // for (ITYPE i = 0; i < 20; i++) {
    //     std::cout << "Core: " << i << " -- " << per_core_runtime[i] / ((double) min_time) << std::endl;
    // }

    #endif

    long long end_cycle = readTSC();
    return (end_cycle - start_cycle);
}


template <typename T, typename ITYPE>
long long spmm_csr_jstream_hand_optimized( CSR<T, ITYPE> &S, T *I, T *O, ITYPE feature, ITYPE Ti, ITYPE Tk, ITYPE chunk_size = 1 ) {

    ITYPE num_panels = CEIL( S.nrows, Ti );
    ITYPE nrows = S.nrows;
    ITYPE *row_ptr = S.row_ptr;
    ITYPE *cols = S.cols;
    T *vals = S.vals;

    __assume_aligned( row_ptr, ALLOC_ALIGNMENT );
    __assume_aligned( cols, ALLOC_ALIGNMENT );
    __assume_aligned( vals, ALLOC_ALIGNMENT );
    __assume_aligned( I, ALLOC_ALIGNMENT );
    __assume_aligned( O, ALLOC_ALIGNMENT );

    long long start_cycle = readTSC();

    #pragma ivdep
    #pragma vector aligned
    #pragma omp parallel for schedule (OMP_SCHEDULE, chunk_size)
    for ( ITYPE panel = 0; panel < num_panels; panel++ ) {

        for ( ITYPE k = 0; k < feature; k += Tk ) {
            ITYPE panel_start = panel * Ti;
            ITYPE panel_end = MIN( panel_start + Ti, nrows );

            for ( ITYPE row = panel_start; row < panel_end; row++ ) {
                ITYPE row_start = row_ptr[row];
                ITYPE row_end = row_ptr[row + 1];
                T *O_base_addr = &O[ row * (feature + PADDING_C) ];
                for ( ITYPE ptr = row_start; ptr < row_end; ptr++ ) {
                    T val = vals[ptr];
                    rtype Areg = vset( val );
                    ITYPE col = cols[ptr];

                    T* I_base_addr = &I[col * (feature + PADDING_B)];
                    // K loops vetorized with avx2 instructions
                    for ( ITYPE kk = k; kk < MIN(feature, k + Tk); kk += 32 ) {

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

                        // reuse C
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
    }

    long long end_cycle = readTSC();
    return (end_cycle - start_cycle);
}



// Function to add perf counters per panel
template <typename T, typename ITYPE>
long long spmm_csr_jstream_compiler_vectorized( CSR<T, ITYPE> &S, T *I, T *O, ITYPE feature, ITYPE Ti, ITYPE Tk, ITYPE chunk_size = 1, long long *per_core_runtime = nullptr)
{
    ITYPE num_row_panels = CEIL( S.get_nrows(), Ti );

    T* csr_v = S.vals;
    ITYPE *csr_c = S.cols;
    ITYPE *csr_ptr = S.row_ptr;
    T *I_data = I;
    T *O_data = O;

    #ifdef INTEL_COMPILER
        __assume_aligned(csr_v, 64);
        __assume_aligned(csr_c, 64);
        __assume_aligned(csr_ptr, 64);
        __assume_aligned(I_data, 64);
        __assume_aligned(O_data, 64);
    #endif

    ITYPE num_rows = S.nrows;

    long long start_cycle = readTSC();

    #ifdef TRACK_PER_CORE_RUNTIME
    #pragma omp parallel shared(per_core_runtime, csr_v, csr_c, csr_ptr, I_data, O_data, Ti, chunk_size, Tk, feature)
    {
        auto my_tid = omp_get_thread_num();
        per_core_runtime[my_tid] = readTSC();
    #endif

    #pragma ivdep
    #pragma vector aligned
    #ifdef TRACK_PER_CORE_RUNTIME
        #pragma omp for schedule(OMP_SCHEDULE, chunk_size), nowait
    #else
        #pragma omp parallel for schedule(OMP_SCHEDULE, chunk_size)
    #endif
    for ( ITYPE row_panel = 0; row_panel < num_row_panels; row_panel++ ) {

        // TIle / Panel start
        for ( ITYPE k = 0; k < feature; k += Tk ) {

            ITYPE row_start = row_panel * Ti;
            ITYPE row_end = MIN(row_start + Ti, num_rows);

            // Tile start -- Iterate over the row panel
            for ( ITYPE row = row_start; row < row_end; row++ ) {
                ITYPE row_begin = csr_ptr[row];
                ITYPE row_end = csr_ptr[row + 1];

                // Output base register depends on the sparse row index - invariant in below loops
                T* O_base_addr = &O_data[row * (feature + PADDING_C)];

                for ( ITYPE ptr = row_begin; ptr < row_end; ptr++ ) {
                    T sparse_val = csr_v[ptr];
                    ITYPE col = csr_c[ptr];
                    T *I_base_addr = &I_data[ col * (feature + PADDING_B) ];

                    #pragma ivdep
                    #pragma vector aligned
                    for ( ITYPE kk = k; kk < MIN(feature, k + Tk); kk += 16 ) {
                        O_base_addr[ kk + 0 ] += sparse_val * I_base_addr[ kk + 0 ];
                        O_base_addr[ kk + 1 ] += sparse_val * I_base_addr[ kk + 1 ];
                        O_base_addr[ kk + 2 ] += sparse_val * I_base_addr[ kk + 2 ];
                        O_base_addr[ kk + 3 ] += sparse_val * I_base_addr[ kk + 3 ];
                        O_base_addr[ kk + 4 ] += sparse_val * I_base_addr[ kk + 4 ];
                        O_base_addr[ kk + 5 ] += sparse_val * I_base_addr[ kk + 5 ];
                        O_base_addr[ kk + 6 ] += sparse_val * I_base_addr[ kk + 6 ];
                        O_base_addr[ kk + 7 ] += sparse_val * I_base_addr[ kk + 7 ];

                        O_base_addr[ kk + 8 ] += sparse_val * I_base_addr[ kk + 8 ];
                        O_base_addr[ kk + 9 ] += sparse_val * I_base_addr[ kk + 9 ];
                        O_base_addr[ kk + 10 ] += sparse_val * I_base_addr[ kk + 10 ];
                        O_base_addr[ kk + 11 ] += sparse_val * I_base_addr[ kk + 11 ];
                        O_base_addr[ kk + 12 ] += sparse_val * I_base_addr[ kk + 12 ];
                        O_base_addr[ kk + 13 ] += sparse_val * I_base_addr[ kk + 13 ];
                        O_base_addr[ kk + 14 ] += sparse_val * I_base_addr[ kk + 14 ];
                        O_base_addr[ kk + 15 ] += sparse_val * I_base_addr[ kk + 15 ];

                    }
                }
            }   // End processinig row panel for Tk wide Input and Output
        }
    }

    #ifdef TRACK_PER_CORE_RUNTIME
        per_core_runtime[my_tid] = readTSC() - per_core_runtime[my_tid];
    } // end omp parallel region
    #endif

    long long end_cycle = readTSC();
    return (end_cycle - start_cycle);
}


#define SIMPLE_COMPILER

// Function to add perf counters per panel
template <typename T, typename ITYPE>
long long csr_row_panel_compiler_vectorized( CSR<T, ITYPE> &S, T *I, T *O, ITYPE feature, ITYPE Ti, ITYPE Tk, ITYPE chunk_size = 1, long long *per_core_runtime = nullptr, long long *per_panel_runtime = nullptr)
{
    ITYPE num_row_panels = CEIL( S.get_nrows(), Ti );

    T* csr_v = S.vals;
    ITYPE *csr_c = S.cols;
    ITYPE *csr_ptr = S.row_ptr;
    T *I_data = I;
    T *O_data = O;

    #if defined(INTEL_COMPILER) && !defined(SIMPLE_COMPILER)
        __assume_aligned(csr_v, 64);
        __assume_aligned(csr_c, 64);
        __assume_aligned(csr_ptr, 64);
        __assume_aligned(I_data, 64);
        __assume_aligned(O_data, 64);
    #endif

    // std::cout << "omp_num_threads: " << omp_get_max_threads() << std::endl;

    ITYPE num_rows = S.nrows;

    long long start_cycle = readTSC();


    #ifdef SIMPLE_COMPILER

        #ifdef RUN_SIMPLE_PANEL
            #pragma omp parallel for schedule(OMP_SCHEDULE, 1)
            for ( ITYPE panel = 0; panel < CEIL(num_rows, chunk_size); panel++ ) {
                ITYPE panel_start = panel * chunk_size;
                ITYPE panel_end = MIN( (panel + 1) * chunk_size, num_rows );
                for ( ITYPE row = panel_start; row < panel_end; row++) {
                    ITYPE row_start = csr_ptr[row];
                    ITYPE row_end = csr_ptr[row + 1];
                    for ( ITYPE ptr = row_start; ptr < row_end; ptr++ ) {
                        for ( ITYPE k = 0; k < feature; k++ ) {
                            O[ row * (feature + PADDING_C) + k ] += I[ csr_c[ptr] * (feature + PADDING_B) + k ] * csr_v[ptr];
                        }
                    }
                }
            }

        #else
            #pragma omp parallel for schedule(OMP_SCHEDULE, chunk_size)
            for ( ITYPE row = 0; row < num_rows; row++) {
                ITYPE row_start = csr_ptr[row];
                ITYPE row_end = csr_ptr[row + 1];
                for ( ITYPE ptr = row_start; ptr < row_end; ptr++ ) {
                    for ( ITYPE k = 0; k < feature; k++ ) {
                        O[ row * (feature + PADDING_C) + k ] += I[ csr_c[ptr] * (feature + PADDING_B) + k ] * csr_v[ptr];
                    }
                }
            }
        #endif

    #else
        #ifdef TRACK_PER_CORE_RUNTIME
        #pragma omp parallel shared(per_core_runtime, csr_v, csr_c, csr_ptr, I_data, O_data, Ti, chunk_size, Tk, feature)
        {
            auto my_tid = omp_get_thread_num();
            per_core_runtime[my_tid] = readTSC();
        #endif


        #pragma ivdep
        #pragma vector aligned
        #ifdef TRACK_PER_CORE_RUNTIME
            #pragma omp for schedule(OMP_SCHEDULE, chunk_size), nowait
        #else
            #pragma omp parallel for schedule(OMP_SCHEDULE, chunk_size)
        #endif
        for (ITYPE row = 0; row < num_rows; row++) {
            ITYPE row_start = csr_ptr[row];
            ITYPE row_end = csr_ptr[row + 1];

            ITYPE unroll_limit = row_start + (((row_end - row_start) >> 3) << 3);
            ITYPE ptr;

            for ( ptr = row_start; ptr < unroll_limit; ptr += 8 ) {

                #pragma ivdep
                for ( ITYPE k = 0; k < feature; k++ ) {
                    // O_data[ row * (feature + PADDING_C) + k ] += I_data[ csr_c[ptr + 0] * (feature + PADDING_B) + k  ] * csr_v[ptr + 0] +
                    //                                             I_data[ csr_c[ptr + 1] * (feature + PADDING_B) + k  ] * csr_v[ptr + 1] +
                    //                                             I_data[ csr_c[ptr + 2] * (feature + PADDING_B) + k  ] * csr_v[ptr + 2] +
                    //                                             I_data[ csr_c[ptr + 3] * (feature + PADDING_B) + k  ] * csr_v[ptr + 3] +
                    //                                             I_data[ csr_c[ptr + 4] * (feature + PADDING_B) + k  ] * csr_v[ptr + 4] +
                    //                                             I_data[ csr_c[ptr + 5] * (feature + PADDING_B) + k  ] * csr_v[ptr + 5] +
                    //                                             I_data[ csr_c[ptr + 6] * (feature + PADDING_B) + k  ] * csr_v[ptr + 6] +
                    //                                             I_data[ csr_c[ptr + 7] * (feature + PADDING_B) + k  ] * csr_v[ptr + 7];
                    // repeat the same loop but the multiplicaton is csr_v first instead of the above
                    O_data[ row * (feature + PADDING_C) + k ] += csr_v[ptr + 0] * I_data[ csr_c[ptr + 0] * (feature + PADDING_B) + k  ] +
                                                                csr_v[ptr + 1] * I_data[ csr_c[ptr + 1] * (feature + PADDING_B) + k  ] +
                                                                csr_v[ptr + 2] * I_data[ csr_c[ptr + 2] * (feature + PADDING_B) + k  ] +
                                                                csr_v[ptr + 3] * I_data[ csr_c[ptr + 3] * (feature + PADDING_B) + k  ] +
                                                                csr_v[ptr + 4] * I_data[ csr_c[ptr + 4] * (feature + PADDING_B) + k  ] +
                                                                csr_v[ptr + 5] * I_data[ csr_c[ptr + 5] * (feature + PADDING_B) + k  ] +
                                                                csr_v[ptr + 6] * I_data[ csr_c[ptr + 6] * (feature + PADDING_B) + k  ] +
                                                                csr_v[ptr + 7] * I_data[ csr_c[ptr + 7] * (feature + PADDING_B) + k  ];


                }
            }

            for ( ; ptr < row_end; ptr++ ) {
                #pragma ivdep
                // #pragma vector nontemporal (csr_v)
                // #pragma prefetch I_data:_MM_HINT_T1
                for ( ITYPE k = 0; k < feature; k++ ) {
                    // O_data[ row * (feature + PADDING_C) + k ] += I_data[ csr_c[ptr + 0] * (feature + PADDING_B) + k  ] * csr_v[ptr + 0];
                    O_data[ row * (feature + PADDING_C) + k ] += csr_v[ptr + 0] * I_data[ csr_c[ptr + 0] * (feature + PADDING_B) + k  ];
                }
            }
        }

        #ifdef TRACK_PER_CORE_RUNTIME
            per_core_runtime[my_tid] = readTSC() - per_core_runtime[my_tid];
        } // end omp parallel region
        #endif
    #endif

    long long end_cycle = readTSC();
    return (end_cycle - start_cycle);
}


// Function to add perf counters per panel
template <typename T, typename ITYPE>
long long spmm_csr_jstream_worklist( CSR<T, ITYPE> &S, T *I, T *O, ITYPE feature, ITYPE *num_panels_per_thread, struct workitem *worklist, ITYPE num_threads, ITYPE chunk_size = 1, long long *per_core_timing = nullptr, long long *per_panel_timing = nullptr )
{
    T* csr_v = S.vals;
    ITYPE *csr_c = S.cols;
    ITYPE *csr_ptr = S.row_ptr;
    T *I_data = I;
    T *O_data = O;

    #ifdef INTEL_COMPILER
        __assume_aligned(csr_v, 64);
        __assume_aligned(csr_c, 64);
        __assume_aligned(csr_ptr, 64);
        __assume_aligned(I_data, 64);
        __assume_aligned(O_data, 64);

        __assume_aligned(num_panels_per_thread, 64);
        __assume_aligned(worklist, 64);
    #endif

    ITYPE num_rows = S.nrows;

    long long start_cycle = readTSC();

    #pragma omp parallel shared(csr_v, csr_c, csr_ptr, I_data, O_data, feature, worklist, num_panels_per_thread)
    {
    auto my_tid = omp_get_thread_num();
    #ifdef TRACK_PER_CORE_RUNTIME
        per_core_timing[my_tid] = readTSC();
    #endif

    for ( ITYPE row_panel = 0; row_panel < num_panels_per_thread[my_tid]; row_panel++ ) {

        ITYPE offset = row_panel * num_threads + my_tid;
        #ifdef TRACK_PER_PANEL_RUNTIME
            per_panel_timing[ offset ] = readTSC();
        #endif

        // TIle / Panel start

        // if (worklist[offset].panel_id == 54) {continue;}

        ITYPE Tk = worklist[ offset ].Tk;

        for ( ITYPE k = 0; k < feature; k += Tk ) {

            ITYPE panel_start = worklist[ offset ].start_row;
            ITYPE panel_end = MIN( worklist[ offset ].end_row, num_rows);

            // Tile start -- Iterate over the row panel
            for ( ITYPE row = panel_start; row < panel_end; row++ ) {
                ITYPE row_begin = csr_ptr[row];
                ITYPE row_end = csr_ptr[row + 1];

                // Output base register depends on the sparse row index - invariant in below loops
		        T* O_base_addr = &O_data[row * feature];

                #ifdef REUSE_OUTPUT_FROM_REGS

                /*
                    // Some funky code to prefetch the entire I row before use
                    for ( ITYPE kk = k; kk < MIN(feature, k + Tk); kk += 32 ) {
                        for ( ITYPE r = row_begin; r < row_end; r++) {
                            ITYPE col = csr_c[r];

                            T *input_ba = &I_data[col * feature];

                            _mm_prefetch( (input_ba + kk + 0), 1 );
                            _mm_prefetch( (input_ba + kk + 8), 1 );
                            _mm_prefetch( (input_ba + kk + 16), 1 );
                            _mm_prefetch( (input_ba + kk + 24), 1 );
                        }
                    }
                */

                // slice is 32 wide in this case
                ITYPE kk_end = MIN(k + Tk, feature);
                for ( ITYPE kk = k; kk < kk_end; kk += 32 ) {

                    // Load the output slice into registers
                    rtype Creg0 = vload( O_base_addr + kk + 0 );
                    rtype Creg1 = vload( O_base_addr + kk + 4 );
                    rtype Creg2 = vload( O_base_addr + kk + 8 );
                    rtype Creg3 = vload( O_base_addr + kk + 12 );
                    rtype Creg4 = vload( O_base_addr + kk + 16 );
                    rtype Creg5 = vload( O_base_addr + kk + 20 );
                    rtype Creg6 = vload( O_base_addr + kk + 24 );
                    rtype Creg7 = vload( O_base_addr + kk + 28 );

                    // Iterate over all non-zeroes of the row
                    for ( ITYPE r = row_begin; r < row_end; r++ ) {
                        ITYPE col = csr_c[r];
                        T val = csr_v[r];
                        rtype Areg = vset( val );

                        // The column index of the non-zero dictates the I base address
                        T *I_base_addr = &I_data[col * feature];

                        /*
                            // We might want to prefetch the slice of the next row
                            T *prefetch_base_addr = &I_data[csr_c[r + 1] * feature] + kk;
                            _mm_prefetch( (prefetch_base_addr + 0), 1 );
                            _mm_prefetch( (prefetch_base_addr + 8), 1 );
                            _mm_prefetch( (prefetch_base_addr + 16), 1 );
                            _mm_prefetch( (prefetch_base_addr + 24), 1 );
                        */

                        // Load the first half of the I slice
                        rtype Breg0 = vload( I_base_addr + kk + 0 );
                        rtype Breg1 = vload( I_base_addr + kk + 4 );
                        rtype Breg2 = vload( I_base_addr + kk + 8 );
                        rtype Breg3 = vload( I_base_addr + kk + 12 );

                        // First set of fused multiply adds performed in place
                        Creg0 = vfma( Areg, Breg0, Creg0 );
                        Creg1 = vfma( Areg, Breg1, Creg1 );
                        Creg2 = vfma( Areg, Breg2, Creg2 );
                        Creg3 = vfma( Areg, Breg3, Creg3 );

                        // Load the second half of the I slice
                        Breg0 = vload( I_base_addr + kk + 16 );
                        Breg1 = vload( I_base_addr + kk + 20 );
                        Breg2 = vload( I_base_addr + kk + 24 );
                        Breg3 = vload( I_base_addr + kk + 28 );


                        // Second set of fused multiply adds performed in place
                        Creg4 = vfma( Areg, Breg0, Creg4 );
                        Creg5 = vfma( Areg, Breg1, Creg5 );
                        Creg6 = vfma( Areg, Breg2, Creg6 );
                        Creg7 = vfma( Areg, Breg3, Creg7 );
                    }

                    // Store slice back into memory before processing the next sparse row
                    vstore( O_base_addr + kk + 0, Creg0 );
                    vstore( O_base_addr + kk + 4, Creg1 );
                    vstore( O_base_addr + kk + 8, Creg2 );
                    vstore( O_base_addr + kk + 12, Creg3 );
                    vstore( O_base_addr + kk + 16, Creg4 );
                    vstore( O_base_addr + kk + 20, Creg5 );
                    vstore( O_base_addr + kk + 24, Creg6 );
                    vstore( O_base_addr + kk + 28, Creg7 );
                }


                #else

                for ( ITYPE r = row_begin; r < row_end; r++ ) {

                    ITYPE col = csr_c[r];
                    T val = csr_v[r];
                    rtype Areg = vset( val );

                    T* I_base_addr = &I_data[col * feature];
                    // K loops vetorized with avx2 instructions
                    for ( ITYPE kk = k; kk < MIN(feature, k + Tk); kk += 32 ) {

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

                        // reuse C -- because we don't want to spill out of the register file!
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

                #endif // REUSE_OUTPUT_FROM_REGS
            } // End processinig row panel for Tk wide Input and Output
        } // End K loop over the current panel

        #ifdef TRACK_PER_PANEL_RUNTIME
            per_panel_timing[ offset ] = readTSC() - per_panel_timing[ offset ];
        #endif

    } // End of panel (worklist) loop

    #ifdef TRACK_PER_CORE_RUNTIME
        per_core_timing[my_tid] = readTSC() - per_core_timing[my_tid];
    #endif


    } // End omp parallel region

    long long end_cycle = readTSC();
    return (end_cycle - start_cycle);
}

// Function to produce per panel statistics
template <typename T, typename ITYPE>
long long spmm_csr_jstream_panel_statistics( CSR<T, ITYPE> &S, T *I, T *O, ITYPE feature, ITYPE Ti, ITYPE Tk, ITYPE chunk_size, perf *p, stats_t<long long, ITYPE> **stats)
{
    ITYPE num_row_panels = CEIL( S.get_nrows(), Ti );

    T* csr_v = S.vals;
    ITYPE *csr_c = S.cols;
    ITYPE *csr_ptr = S.row_ptr;
    T *I_data = I;
    T *O_data = O;

    __assume_aligned(csr_v, 64);
    __assume_aligned(csr_c, 64);
    __assume_aligned(csr_ptr, 64);
    __assume_aligned(I_data, 64);
    __assume_aligned(O_data, 64);


    ITYPE num_rows = S.nrows;

    long long start_cycle = readTSC();

    #pragma ivdep
    #pragma vector aligned
    for ( ITYPE row_panel = 0; row_panel < num_row_panels; row_panel++ ) {

        // TIle / Panel start
        #ifdef PER_PANEL_PERF_EVENTS
            parallel_dram_start_helper(p);
        #endif

        for ( ITYPE k = 0; k < feature; k += Tk ) {

            ITYPE row_start = row_panel * Ti;

            // Tile start -- fake

            for ( ITYPE row = row_start; row < MIN(row_start + Ti, num_rows); row++ ) {
                ITYPE row_begin = csr_ptr[row];
                ITYPE row_end = csr_ptr[row + 1];

                T* O_base_addr = &O_data[row * feature];
                for ( ITYPE r = row_begin; r < row_end; r++ ) {

                    ITYPE col = csr_c[r];
                    T val = csr_v[r];
                    rtype Areg = vset( val );

                    T* I_base_addr = &I_data[col * feature];
                    // K loops vetorized with avx2 instructions
                    for ( ITYPE kk = k; kk < MIN(feature, k + Tk); kk += 32 ) {

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

                        // reuse C

                        #ifdef REUSE_C

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

                        #else // REUSE_C

                            rtype Creg5 = vload( O_base_addr + kk + 16 );
                            Creg5 = vfma( Areg, Breg5, Creg5 );
                            vstore( O_base_addr + kk + 16, Creg5 );

                            rtype Creg6 = vload( O_base_addr + kk + 20 );
                            Creg6 = vfma( Areg, Breg6, Creg6 );
                            vstore( O_base_addr + kk + 20, Creg6 );

                            rtype Creg7 = vload( O_base_addr + kk + 24 );
                            Creg7 = vfma( Areg, Breg7, Creg7 );
                            vstore( O_base_addr + kk + 24, Creg7 );

                            rtype Creg8 = vload( O_base_addr + kk + 28 );
                            Creg8 = vfma( Areg, Breg8, Creg8 );
                            vstore( O_base_addr + kk + 28, Creg8 );

                        #endif // REUSE_C
                    }
                }
            }

            // Tile end -- fake

        }

        // Tile / Panel end
        #ifdef PER_PANEL_PERF_EVENTS
            dram_stop_helper(p, stats[row_panel]);
        #endif
    }

    long long end_cycle = readTSC();
    return (end_cycle - start_cycle);
}



template <typename T, typename ITYPE>
void spmm_csr_j_stream_avx512( CSR<T, ITYPE> &S, Matrix<T, ITYPE> &I, Matrix<T, ITYPE> &O, ITYPE Ti, ITYPE Tk, ITYPE chunk_size = 1 )
{

    ITYPE num_row_panels = CEIL( S.get_nrows(), Ti );
    ITYPE num_threads = omp_get_num_threads();


    T* csr_v = S.vals;
    ITYPE *csr_c = S.cols;
    ITYPE *csr_ptr = S.row_ptr;
    T *I_data = I._data;
    T *O_data = O._data;

    __assume_aligned(csr_v, 64);
    __assume_aligned(csr_c, 64);
    __assume_aligned(csr_ptr, 64);
    __assume_aligned(I_data, 64);
    __assume_aligned(O_data, 64);


    ITYPE num_rows = S.nrows;
    ITYPE feature = I.ncols;

    #pragma ivdep
    #pragma vector aligned
    #pragma omp parallel for schedule(static, chunk_size)
    for ( ITYPE row_panel = 0; row_panel < num_row_panels; row_panel++ ) {

        for ( ITYPE k = 0; k < I.ncols; k += Tk ) {

            ITYPE row_start = row_panel * Ti;
            for ( ITYPE row = row_start; row < MIN(row_start + Ti, num_rows); row++ ) {
                ITYPE row_begin = csr_ptr[row];
                ITYPE row_end = csr_ptr[row + 1];
                T *O_base_addr = &O_data[row * feature];


                // Lets load the output before writing to it
                rtype512 Creg1 = vload512( O_base_addr + k + 0 );
                rtype512 Creg2 = vload512( O_base_addr + k + 8 );
                rtype512 Creg3 = vload512( O_base_addr + k + 16 );
                rtype512 Creg4 = vload512( O_base_addr + k + 24 );

                for ( ITYPE r = row_begin; r < row_end; r++ ) {

                    ITYPE col = csr_c[r];
                    T val = csr_v[r];
                    rtype512 Areg = vset512( val );

                    T *I_base_addr = &I_data[col * feature];
                    for ( ITYPE kk = k; kk < MIN(feature, k + Tk); kk += 32 ) {

                        rtype512 Breg1 = vload512( I_base_addr + kk + 0 );
                        rtype512 Breg2 = vload512( I_base_addr + kk + 8 );
                        rtype512 Breg3 = vload512( I_base_addr + kk + 16 );
                        rtype512 Breg4 = vload512( I_base_addr + kk + 24 );

                        // rtype512 Creg1 = vload512( O_base_addr + kk + 0 );
                        Creg1 = vfma512( Areg, Breg1, Creg1 );
                        // vstore512( O_base_addr + kk + 0, Creg1 );

                        // rtype512 Creg2 = vload512( O_base_addr + kk + 8 );
                        Creg2 = vfma512( Areg, Breg2, Creg2 );
                        // vstore512( O_base_addr + kk + 8, Creg2 );

                        // rtype512 Creg3 = vload512( O_base_addr + kk + 16 );
                        Creg3 = vfma512( Areg, Breg3, Creg3 );
                        // vstore512( O_base_addr + kk + 16, Creg3 );

                        // rtype512 Creg4 = vload512( O_base_addr + kk + 24 );
                        Creg4 = vfma512( Areg, Breg4, Creg4 );
                        // vstore512( O_base_addr + kk + 24, Creg4 );
                    }
                }

                // Store the output back to main memory
                vstore512( O_base_addr + k + 0, Creg1 );
                vstore512( O_base_addr + k + 8, Creg2 );
                vstore512( O_base_addr + k + 16, Creg3 );
                vstore512( O_base_addr + k + 24, Creg4 );
            }
        }
    }

}


/********************** J-Stream implementation based on column-wise traversals **********************/

// #define PRINT_THREAD_ID

template <typename T, typename ITYPE>
void spmm_dcsc_jstream( DCSC<T, ITYPE> &S, Matrix<T, ITYPE> &I, Matrix<T, ITYPE> &O, ITYPE Ti, ITYPE Tk, ITYPE chunk_size = 1 )
{
    ITYPE feature = I.ncols;

    T* dcsc_vals = S.vals;
    ITYPE *dcsc_rows = S.rows;
    ITYPE *dcsc_col_ptr = S.col_ptr;
    ITYPE *dcsc_cols = S.cols;
    ITYPE *panel_ptr = S.aux;
    T *I_data = I._data;
    T *O_data = O._data;

    __assume_aligned(dcsc_vals, ALLOC_ALIGNMENT);
    __assume_aligned(dcsc_rows, ALLOC_ALIGNMENT);
    __assume_aligned(dcsc_col_ptr, ALLOC_ALIGNMENT);
    __assume_aligned(dcsc_cols, ALLOC_ALIGNMENT);
    __assume_aligned(I_data, ALLOC_ALIGNMENT);
    __assume_aligned(O_data, ALLOC_ALIGNMENT);

    #pragma ivdep
    #pragma vector aligned
    #pragma omp parallel for schedule (static, chunk_size)
    for (ITYPE row_panel = 0; row_panel < S.num_segments; row_panel++) { // I loop
        if (panel_ptr[row_panel] == panel_ptr[row_panel + 1]) {
            continue;
        }

        for (ITYPE k = 0; k < feature; k += Tk) {  // K loop

            // Iterating over the active columns of a row panel
            for (ITYPE j = panel_ptr[row_panel]; j < panel_ptr[row_panel + 1]; j++) {
                ITYPE col = dcsc_cols[j];
                ITYPE nRows = dcsc_col_ptr[j + 1] - dcsc_col_ptr[j];
                T *input_base_addr = &I_data[feature * col];

                for (ITYPE kk = k; kk < MIN(feature, k + Tk); kk += 32) {

                    rtype Breg1 = vload( input_base_addr + kk + 0  );
                    rtype Breg2 = vload( input_base_addr + kk + 4  );
                    rtype Breg3 = vload( input_base_addr + kk + 8  );
                    rtype Breg4 = vload( input_base_addr + kk + 12  );
                    rtype Breg5 = vload( input_base_addr + kk + 16  );
                    rtype Breg6 = vload( input_base_addr + kk + 20  );
                    rtype Breg7 = vload( input_base_addr + kk + 24  );
                    rtype Breg8 = vload( input_base_addr + kk + 28  );

                    // Iterating over rows of an active column
                    for (ITYPE i = 0; i < nRows; i++) {

                        T *output_base_addr = &O_data[ dcsc_rows[i + dcsc_col_ptr[j]] * feature ];
                        T aval = dcsc_vals[ i + dcsc_col_ptr[j] ];
                        rtype Areg = vset( aval );

                        rtype Creg1 = vload( output_base_addr + kk + 0 );
                        Creg1 = vfma( Areg, Breg1, Creg1 );
                        vstore( output_base_addr + kk + 0, Creg1 );

                        rtype Creg2 = vload( output_base_addr + kk + 4 );
                        Creg2 = vfma( Areg, Breg2, Creg2 );
                        vstore( output_base_addr + kk + 4, Creg2 );

                        rtype Creg3 = vload( output_base_addr + kk + 8 );
                        Creg3 = vfma( Areg, Breg3, Creg3 );
                        vstore( output_base_addr + kk + 8, Creg3 );

                        rtype Creg4 = vload( output_base_addr + kk + 12 );
                        Creg4 = vfma( Areg, Breg4, Creg4 );
                        vstore( output_base_addr + kk + 12, Creg4 );

                        // Reuse the C register set
                        Creg1 = vload( output_base_addr + kk + 16 );
                        Creg1 = vfma( Areg, Breg5, Creg1 );
                        vstore( output_base_addr + kk + 16, Creg1 );

                        Creg2 = vload( output_base_addr + kk + 20 );
                        Creg2 = vfma( Areg, Breg6, Creg2 );
                        vstore( output_base_addr + kk + 20, Creg2 );

                        Creg3 = vload( output_base_addr + kk + 24 );
                        Creg3 = vfma( Areg, Breg7, Creg3 );
                        vstore( output_base_addr + kk + 24, Creg3 );

                        Creg4 = vload( output_base_addr + kk + 28 );
                        Creg4 = vfma( Areg, Breg8, Creg4 );
                        vstore( output_base_addr + kk + 28, Creg4 );
                    }
                }
            }
        }
    }
}


#define USE_JSTREAM_SC_LOOP_BODY

template <typename T, typename ITYPE>
long long spmm_dcsc_jstream( DCSC<T, ITYPE> &S, T *I, T *O, ITYPE feature, ITYPE Ti, ITYPE Tk, ITYPE chunk_size = 1, long long *per_core_timing = nullptr, long long *per_panel_timing = nullptr )
{
    ITYPE num_panels = S.num_segments;
    T* dcsc_vals = S.vals;
    ITYPE *dcsc_rows = S.rows;
    ITYPE *dcsc_col_ptr = S.col_ptr;
    ITYPE *dcsc_cols = S.cols;
    ITYPE *panel_ptr = S.aux;
    ITYPE *panel_Tk = S.panel_Tk;
    T *I_data = I;
    T *O_data = O;

    #ifdef INTEL_COMPILER
        __assume_aligned(dcsc_vals, ALLOC_ALIGNMENT);
        __assume_aligned(dcsc_rows, ALLOC_ALIGNMENT);
        __assume_aligned(dcsc_col_ptr, ALLOC_ALIGNMENT);
        __assume_aligned(dcsc_cols, ALLOC_ALIGNMENT);
        __assume_aligned(I_data, ALLOC_ALIGNMENT);
        __assume_aligned(O_data, ALLOC_ALIGNMENT);
    #endif

    #ifdef PRINT_THREAD_ID
        ITYPE row_panel_thread_map[S.num_segments];
    #endif

    long long start_cycle = readTSC();


#ifndef USE_JSTREAM_SC_LOOP_BODY

    #ifdef TRACK_PER_CORE_RUNTIME
    #pragma omp parallel shared(dcsc_vals, dcsc_rows, dcsc_col_ptr, dcsc_cols, I_data, Tk)
    {
        auto my_tid = omp_get_thread_num();
        #ifdef TRACK_PER_PANEL_RUNTIME
            auto num_threads = omp_get_num_threads();
        #endif
        per_core_timing[my_tid] = readTSC();
    #endif

    #pragma ivdep
    #pragma vector aligned
    #ifdef TRACK_PER_CORE_RUNTIME
    #pragma omp for schedule (OMP_SCHEDULE, chunk_size), nowait
    #else
    #pragma omp parallel for schedule (OMP_SCHEDULE, chunk_size)
    #endif
    for (ITYPE row_panel = 0; row_panel < num_panels; row_panel++) { // I loop

        #ifdef PRINT_THREAD_ID
            // auto my_tid = omp_get_thread_num();
            row_panel_thread_map[row_panel] = my_tid;
        #endif

        #ifdef TRACK_PER_PANEL_RUNTIME
            ITYPE offset = my_tid + num_threads * (row_panel / num_threads);
            per_panel_timing[offset] = readTSC();
        #endif

        if (panel_ptr[row_panel] == panel_ptr[row_panel + 1]) {
            continue;
        }

        // Panel kernel start
        #ifdef PER_PANEL_TK
            ITYPE my_Tk = panel_Tk[row_panel];
            for ( ITYPE k = 0; k < feature; k += my_Tk ) {
        #else
            for (ITYPE k = 0; k < feature; k += Tk) {  // K loop
        #endif

            // Iterating over the active columns of a row panel
            for (ITYPE j = panel_ptr[row_panel]; j < panel_ptr[row_panel + 1]; j++) {
                ITYPE col = dcsc_cols[j];
                ITYPE nRows = dcsc_col_ptr[j + 1] - dcsc_col_ptr[j];
                T *input_base_addr = &I_data[ (feature + PADDING_B) * col];

                #ifdef PER_PANEL_TK
                    for (ITYPE kk = k; kk < MIN(feature, k + my_Tk); kk += 32) {
                #else
                    for (ITYPE kk = k; kk < MIN(feature, k + Tk); kk += 32) {
                #endif
                    rtype Breg1 = vload( input_base_addr + kk + 0  );
                    rtype Breg2 = vload( input_base_addr + kk + 4  );
                    rtype Breg3 = vload( input_base_addr + kk + 8  );
                    rtype Breg4 = vload( input_base_addr + kk + 12  );
                    rtype Breg5 = vload( input_base_addr + kk + 16  );
                    rtype Breg6 = vload( input_base_addr + kk + 20  );
                    rtype Breg7 = vload( input_base_addr + kk + 24  );
                    rtype Breg8 = vload( input_base_addr + kk + 28  );

                    // Iterating over rows of an active column
                    for (ITYPE i = 0; i < nRows; i++) {

                        T *output_base_addr = &O_data[ dcsc_rows[i + dcsc_col_ptr[j]] * (feature + PADDING_C) ];
                        T aval = dcsc_vals[ i + dcsc_col_ptr[j] ];
                        rtype Areg = vset( aval );

                        rtype Creg1 = vload( output_base_addr + kk + 0 );
                        Creg1 = vfma( Areg, Breg1, Creg1 );
                        vstore( output_base_addr + kk + 0, Creg1 );

                        rtype Creg2 = vload( output_base_addr + kk + 4 );
                        Creg2 = vfma( Areg, Breg2, Creg2 );
                        vstore( output_base_addr + kk + 4, Creg2 );

                        rtype Creg3 = vload( output_base_addr + kk + 8 );
                        Creg3 = vfma( Areg, Breg3, Creg3 );
                        vstore( output_base_addr + kk + 8, Creg3 );

                        rtype Creg4 = vload( output_base_addr + kk + 12 );
                        Creg4 = vfma( Areg, Breg4, Creg4 );
                        vstore( output_base_addr + kk + 12, Creg4 );

                        // Reuse the C register set
                        Creg1 = vload( output_base_addr + kk + 16 );
                        Creg1 = vfma( Areg, Breg5, Creg1 );
                        vstore( output_base_addr + kk + 16, Creg1 );

                        Creg2 = vload( output_base_addr + kk + 20 );
                        Creg2 = vfma( Areg, Breg6, Creg2 );
                        vstore( output_base_addr + kk + 20, Creg2 );

                        Creg3 = vload( output_base_addr + kk + 24 );
                        Creg3 = vfma( Areg, Breg7, Creg3 );
                        vstore( output_base_addr + kk + 24, Creg3 );

                        Creg4 = vload( output_base_addr + kk + 28 );
                        Creg4 = vfma( Areg, Breg8, Creg4 );
                        vstore( output_base_addr + kk + 28, Creg4 );
                    }
                }
            }
        }

        // Panel kernel end

        #ifdef TRACK_PER_PANEL_RUNTIME
            per_panel_timing[offset] = readTSC() - per_panel_timing[offset];
        #endif
    }

    #ifdef TRACK_PER_CORE_RUNTIME
        per_core_timing[my_tid] = readTSC() - per_core_timing[my_tid];
    }
    #endif

#else


    #pragma ivdep
    #pragma vector aligned
    #pragma omp parallel for schedule (dynamic, chunk_size)
    for ( ITYPE row_panel = 0; row_panel < num_panels; row_panel++ ) {
        if (panel_ptr[row_panel] == panel_ptr[row_panel + 1]) {
            continue;
        }

        for ( ITYPE k = 0; k < feature; k += Tk ) {

            for ( ITYPE j = panel_ptr[row_panel]; j < panel_ptr[row_panel+1]; j++ ) {
                ITYPE col_idx = dcsc_cols[j];
                ITYPE num_rows = dcsc_col_ptr[j + 1] - dcsc_col_ptr[j];
                ITYPE IbaseIndex = col_idx * (feature + PADDING_B);

                for ( ITYPE kk = k; kk < MIN( feature, k + Tk ); kk += 32 ) {

                    rtype Breg1 = vload(&I[ IbaseIndex + kk + 0 ]);
                    rtype Breg2 = vload(&I[ IbaseIndex + kk + 4 ]);
                    rtype Breg3 = vload(&I[ IbaseIndex + kk + 8 ]);
                    rtype Breg4 = vload(&I[ IbaseIndex + kk + 12]);
                    rtype Breg5 = vload(&I[ IbaseIndex + kk + 16]);
                    rtype Breg6 = vload(&I[ IbaseIndex + kk + 20]);
                    rtype Breg7 = vload(&I[ IbaseIndex + kk + 24]);
                    rtype Breg8 = vload(&I[ IbaseIndex + kk + 28]);

                    for ( ITYPE i = 0; i < num_rows; i++ ) {

                        ITYPE cindex = dcsc_rows[ i + dcsc_col_ptr[j]] * (feature + PADDING_C);
                        T aval = dcsc_vals[i + dcsc_col_ptr[j]];
                        rtype Areg = vset(aval);

                        rtype Creg1 = vload(&O[ cindex + kk + 0 ]);
                        Creg1 = vfma(Areg, Breg1, Creg1);
                        vstore(&O[ cindex + kk + 0 ], Creg1);

                        rtype Creg2 = vload(&O[ cindex + kk + 4 ]);
                        Creg2 = vfma(Areg, Breg2, Creg2);
                        vstore(&O[ cindex + kk + 4  ], Creg2);

                        rtype Creg3 = vload(&O[ cindex + kk + 8 ]);
                        Creg3 = vfma(Areg, Breg3, Creg3);
                        vstore(&O[ cindex + kk + 8 ], Creg3);

                        rtype Creg4 = vload(&O[ cindex + kk + 12 ]);
                        Creg4 = vfma(Areg, Breg4, Creg4);
                        vstore(&O[ cindex + kk + 12  ], Creg4);

                        /// C reuse beg

                        Creg1 = vload(&O[ cindex + kk + 16 ]);
                        Creg1 = vfma(Areg, Breg5, Creg1);
                        vstore(&O[ cindex + kk + 16 ], Creg1);

                        Creg2 = vload(&O[ cindex + kk + 20 ]);
                        Creg2 = vfma(Areg, Breg6, Creg2);
                        vstore(&O[ cindex + kk + 20  ], Creg2);

                        Creg3 = vload(&O[ cindex + kk + 24 ]);
                        Creg3 = vfma(Areg, Breg7, Creg3);
                        vstore(&O[ cindex + kk + 24 ], Creg3);

                        Creg4 = vload(&O[ cindex + kk + 28 ]);
                        Creg4 = vfma(Areg, Breg8, Creg4);
                        vstore(&O[ cindex + kk + 28  ], Creg4);
                    }
                }
            }
        }
    }



#endif


    long long end_cycle = readTSC();

    #ifdef PRINT_THREAD_ID
        for (ITYPE panel = 0; panel < S.num_segments; panel++) {
            std::cout << "Panel: " << panel << " -- " << row_panel_thread_map[panel] << std::endl;
        }
    #endif

    return (end_cycle - start_cycle);
}





template <typename T, typename ITYPE>
long long spmm_dcsc_compiler_vectorized( DCSC<T, ITYPE> &S, T *I, T *O, ITYPE feature, ITYPE Ti, ITYPE Tk, ITYPE chunk_size = 1, long long *per_core_timing = nullptr, long long *per_panel_timing = nullptr )
{
    ITYPE num_panels = S.num_segments;
    T* dcsc_vals = S.vals;
    ITYPE *dcsc_rows = S.rows;
    ITYPE *dcsc_col_ptr = S.col_ptr;
    ITYPE *dcsc_cols = S.cols;
    ITYPE *panel_ptr = S.aux;
    ITYPE *panel_Tk = S.panel_Tk;
    T *I_data = I;
    T *O_data = O;

    #ifdef INTEL_COMPILER
        __assume_aligned(dcsc_vals, ALLOC_ALIGNMENT);
        __assume_aligned(dcsc_rows, ALLOC_ALIGNMENT);
        __assume_aligned(dcsc_col_ptr, ALLOC_ALIGNMENT);
        __assume_aligned(dcsc_cols, ALLOC_ALIGNMENT);
        __assume_aligned(I_data, ALLOC_ALIGNMENT);
        __assume_aligned(O_data, ALLOC_ALIGNMENT);
    #endif

    long long start_cycle = readTSC();

    #ifdef TRACK_PER_CORE_RUNTIME
    #pragma omp parallel shared(dcsc_vals, dcsc_rows, dcsc_col_ptr, dcsc_cols, I_data, Tk)
    {
        auto my_tid = omp_get_thread_num();
        per_core_timing[my_tid] = readTSC();
    #endif

    #pragma ivdep
    #pragma vector aligned
    #ifdef TRACK_PER_CORE_RUNTIME
        #pragma omp for schedule (OMP_SCHEDULE, chunk_size), nowait
    #else
        #pragma omp parallel for schedule (OMP_SCHEDULE, chunk_size)
    #endif
    for (ITYPE row_panel = 0; row_panel < num_panels; row_panel++) { // I loop

        if (panel_ptr[row_panel] == panel_ptr[row_panel + 1]) {
            continue;
        }

        for (ITYPE k = 0; k < feature; k += Tk) {  // K loop

            // Iterating over the active columns of a row panel
            for (ITYPE j = panel_ptr[row_panel]; j < panel_ptr[row_panel + 1]; j++) {
                ITYPE col = dcsc_cols[j];
                ITYPE col_start = dcsc_col_ptr[j];
                ITYPE col_end = dcsc_col_ptr[j + 1];

                for (ITYPE kk = k; kk < MIN(feature, k + Tk); kk += 8) {

                    // Iterating over rows of an active column
                    for (ITYPE ptr = col_start; ptr < col_end; ptr++) {

                        T sparse_val = dcsc_vals[ ptr ];

                        O_data[ dcsc_rows[ptr] * feature + 0 ] += I_data[ feature * col + 0 ] * sparse_val;
                        O_data[ dcsc_rows[ptr] * feature + 1 ] += I_data[ feature * col + 1 ] * sparse_val;
                        O_data[ dcsc_rows[ptr] * feature + 2 ] += I_data[ feature * col + 2 ] * sparse_val;
                        O_data[ dcsc_rows[ptr] * feature + 3 ] += I_data[ feature * col + 3 ] * sparse_val;
                        O_data[ dcsc_rows[ptr] * feature + 4 ] += I_data[ feature * col + 4 ] * sparse_val;
                        O_data[ dcsc_rows[ptr] * feature + 5 ] += I_data[ feature * col + 5 ] * sparse_val;
                        O_data[ dcsc_rows[ptr] * feature + 6 ] += I_data[ feature * col + 6 ] * sparse_val;
                        O_data[ dcsc_rows[ptr] * feature + 7 ] += I_data[ feature * col + 7 ] * sparse_val;
                    }
                }
            }
        }
    }

    #ifdef TRACK_PER_CORE_RUNTIME
        per_core_timing[my_tid] = readTSC() - per_core_timing[my_tid];
    }
    #endif

    long long end_cycle = readTSC();

    return (end_cycle - start_cycle);
}











// worklist balanced
template <typename T, typename ITYPE>
long long spmm_dcsc_jstream_worklist( DCSC<T, ITYPE> &S, T *I, T *O, ITYPE feature, ITYPE *per_core_num_panels, ITYPE *worklist, long long *per_core_timing = nullptr, long long *per_panel_timing = nullptr, ITYPE num_threads = 64 )
{
    ITYPE num_panels = S.num_segments;
    T* dcsc_vals = S.vals;
    ITYPE *dcsc_rows = S.rows;
    ITYPE *dcsc_col_ptr = S.col_ptr;
    ITYPE *dcsc_cols = S.cols;
    ITYPE *panel_ptr = S.aux;
    ITYPE *panel_Tk = S.panel_Tk;
    T *I_data = I;
    T *O_data = O;

    #ifdef INTEL_COMPILER
        __assume_aligned(dcsc_vals, ALLOC_ALIGNMENT);
        __assume_aligned(dcsc_rows, ALLOC_ALIGNMENT);
        __assume_aligned(dcsc_col_ptr, ALLOC_ALIGNMENT);
        __assume_aligned(dcsc_cols, ALLOC_ALIGNMENT);
        __assume_aligned(I_data, ALLOC_ALIGNMENT);
        __assume_aligned(O_data, ALLOC_ALIGNMENT);
    #endif

    #ifdef PRINT_THREAD_ID
        ITYPE row_panel_thread_map[S.num_segments];
    #endif

    long long start_cycle = readTSC();


    #pragma omp parallel shared(dcsc_vals, dcsc_rows, dcsc_col_ptr, dcsc_cols, I_data, O_data, panel_Tk, panel_ptr)
    {
        auto my_tid = omp_get_thread_num();

        #ifdef TRACK_PER_CORE_RUNTIME
            #ifdef TRACK_PER_PANEL_RUNTIME
                auto num_threads = omp_get_num_threads();
            #endif
            auto core_start_time = readTSC();
        #endif

        #pragma ivdep
        #pragma vector aligned
        for (ITYPE panel_count = 0; panel_count < per_core_num_panels[my_tid]; panel_count++) { // I loop

            // Panel being processed by the current thread
            ITYPE row_panel = worklist[my_tid + (panel_count * num_threads)];

            #ifdef PRINT_THREAD_ID
                // auto my_tid = omp_get_thread_num();
                row_panel_thread_map[row_panel] = my_tid;
            #endif

            #ifdef TRACK_PER_PANEL_RUNTIME
                ITYPE offset = my_tid + num_threads * (row_panel / num_threads);
                per_panel_timing[offset] = readTSC();
            #endif

            if (panel_ptr[row_panel] == panel_ptr[row_panel + 1]) {
                continue;
            }

            // Panel kernel start
            ITYPE my_Tk = panel_Tk[row_panel];
            for ( ITYPE k = 0; k < feature; k += my_Tk ) {

                // Iterating over the active columns of a row panel
                for (ITYPE j = panel_ptr[row_panel]; j < panel_ptr[row_panel + 1]; j++) {
                    ITYPE col = dcsc_cols[j];
                    ITYPE nRows = dcsc_col_ptr[j + 1] - dcsc_col_ptr[j];
                    T *input_base_addr = &I_data[feature * col];

                    for (ITYPE kk = k; kk < MIN(feature, k + my_Tk); kk += 32) {

                        rtype Breg1 = vload( input_base_addr + kk + 0  );
                        rtype Breg2 = vload( input_base_addr + kk + 4  );
                        rtype Breg3 = vload( input_base_addr + kk + 8  );
                        rtype Breg4 = vload( input_base_addr + kk + 12  );
                        rtype Breg5 = vload( input_base_addr + kk + 16  );
                        rtype Breg6 = vload( input_base_addr + kk + 20  );
                        rtype Breg7 = vload( input_base_addr + kk + 24  );
                        rtype Breg8 = vload( input_base_addr + kk + 28  );

                        // Iterating over rows of an active column
                        for (ITYPE i = 0; i < nRows; i++) {

                            T *output_base_addr = &O_data[ dcsc_rows[i + dcsc_col_ptr[j]] * feature ];
                            T aval = dcsc_vals[ i + dcsc_col_ptr[j] ];
                            rtype Areg = vset( aval );

                            rtype Creg1 = vload( output_base_addr + kk + 0 );
                            Creg1 = vfma( Areg, Breg1, Creg1 );
                            vstore( output_base_addr + kk + 0, Creg1 );

                            rtype Creg2 = vload( output_base_addr + kk + 4 );
                            Creg2 = vfma( Areg, Breg2, Creg2 );
                            vstore( output_base_addr + kk + 4, Creg2 );

                            rtype Creg3 = vload( output_base_addr + kk + 8 );
                            Creg3 = vfma( Areg, Breg3, Creg3 );
                            vstore( output_base_addr + kk + 8, Creg3 );

                            rtype Creg4 = vload( output_base_addr + kk + 12 );
                            Creg4 = vfma( Areg, Breg4, Creg4 );
                            vstore( output_base_addr + kk + 12, Creg4 );

                            // Reuse the C register set
                            Creg1 = vload( output_base_addr + kk + 16 );
                            Creg1 = vfma( Areg, Breg5, Creg1 );
                            vstore( output_base_addr + kk + 16, Creg1 );

                            Creg2 = vload( output_base_addr + kk + 20 );
                            Creg2 = vfma( Areg, Breg6, Creg2 );
                            vstore( output_base_addr + kk + 20, Creg2 );

                            Creg3 = vload( output_base_addr + kk + 24 );
                            Creg3 = vfma( Areg, Breg7, Creg3 );
                            vstore( output_base_addr + kk + 24, Creg3 );

                            Creg4 = vload( output_base_addr + kk + 28 );
                            Creg4 = vfma( Areg, Breg8, Creg4 );
                            vstore( output_base_addr + kk + 28, Creg4 );
                        }
                    }
                }
            }

            // Panel kernel end

            #ifdef TRACK_PER_PANEL_RUNTIME
                per_panel_timing[offset] = readTSC() - per_panel_timing[offset];
            #endif
        }

        #ifdef TRACK_PER_CORE_RUNTIME
            auto core_end_cycle = readTSC();
            // auto core_end_cycle = rdtscp();
            per_core_timing[my_tid] = core_end_cycle - core_start_time;
        #endif
    }

    long long end_cycle = readTSC();

    #ifdef PRINT_THREAD_ID
        for (ITYPE panel = 0; panel < S.num_segments; panel++) {
            std::cout << "Panel: " << panel << " -- " << row_panel_thread_map[panel] << std::endl;
        }
    #endif

    return (end_cycle - start_cycle);
}


template<typename T, typename ITYPE>
void spmm_dcsc_jstream_compiler_vectorized( DCSC<T, ITYPE> &S, Matrix<T, ITYPE> &I, Matrix<T, ITYPE> &O, ITYPE Ti, ITYPE Tk, ITYPE chunk_size = 1 )
{
    ITYPE feature = I.ncols;

    T* dcsc_vals = S.vals;
    ITYPE *dcsc_rows = S.rows;
    ITYPE *dcsc_col_ptr = S.col_ptr;
    ITYPE *dcsc_cols = S.cols;
    ITYPE *panel_ptr = S.aux;
    T *I_data = I._data;
    T *O_data = O._data;

    __assume_aligned(dcsc_vals, ALLOC_ALIGNMENT);
    __assume_aligned(dcsc_rows, ALLOC_ALIGNMENT);
    __assume_aligned(dcsc_col_ptr, ALLOC_ALIGNMENT);
    __assume_aligned(dcsc_cols, ALLOC_ALIGNMENT);
    __assume_aligned(I_data, ALLOC_ALIGNMENT);
    __assume_aligned(O_data, ALLOC_ALIGNMENT);

    #pragma ivdep
    #pragma vector aligned
    #pragma vector temporal
    #pragma omp parallel for schedule (static, chunk_size)
    for (ITYPE row_panel = 0; row_panel < S.num_segments; row_panel++) { // I loop
        if (panel_ptr[row_panel] == panel_ptr[row_panel + 1]) {
            continue;
        }

        for (ITYPE k = 0; k < feature; k += Tk) {  // K loop

            for (ITYPE j = panel_ptr[row_panel]; j < panel_ptr[row_panel + 1]; j++) { // J loop
                ITYPE col = dcsc_cols[j]; // column number of the sparse matrix
                ITYPE nRows = dcsc_col_ptr[j + 1] - dcsc_col_ptr[j];
                ITYPE nRows_unrolled = ((nRows >> 3) << 3);

                for (ITYPE kk = k; kk < MIN(feature, k + Tk); kk ++) {

                    int i = 0;
                    #pragma ivdep
                    #pragma vector temporal
                    #pragma vector aligned
                    #pragma prefetch O_data:_MM_HINT_T1
                    for ( ; i < nRows_unrolled; i += 8) {
                        O_data[ dcsc_rows[(i+0) + dcsc_col_ptr[j]] * feature + kk ] += I_data[ col * feature + kk ] * dcsc_vals[(i+0) + dcsc_col_ptr[j] ];
                        O_data[ dcsc_rows[(i+1) + dcsc_col_ptr[j]] * feature + kk ] += I_data[ col * feature + kk ] * dcsc_vals[(i+1) + dcsc_col_ptr[j] ];
                        O_data[ dcsc_rows[(i+2) + dcsc_col_ptr[j]] * feature + kk ] += I_data[ col * feature + kk ] * dcsc_vals[(i+2) + dcsc_col_ptr[j] ];
                        O_data[ dcsc_rows[(i+3) + dcsc_col_ptr[j]] * feature + kk ] += I_data[ col * feature + kk ] * dcsc_vals[(i+3) + dcsc_col_ptr[j] ];
                        O_data[ dcsc_rows[(i+4) + dcsc_col_ptr[j]] * feature + kk ] += I_data[ col * feature + kk ] * dcsc_vals[(i+4) + dcsc_col_ptr[j] ];
                        O_data[ dcsc_rows[(i+5) + dcsc_col_ptr[j]] * feature + kk ] += I_data[ col * feature + kk ] * dcsc_vals[(i+5) + dcsc_col_ptr[j] ];
                        O_data[ dcsc_rows[(i+6) + dcsc_col_ptr[j]] * feature + kk ] += I_data[ col * feature + kk ] * dcsc_vals[(i+6) + dcsc_col_ptr[j] ];
                        O_data[ dcsc_rows[(i+7) + dcsc_col_ptr[j]] * feature + kk ] += I_data[ col * feature + kk ] * dcsc_vals[(i+7) + dcsc_col_ptr[j] ];
                    }

                    #pragma ivdep
                    #pragma vector temporal
                    #pragma vector aligned
                    #pragma prefetch O_data:_MM_HINT_T1
                    for ( ; i < nRows; i++) {
                        O_data[ dcsc_rows[i + dcsc_col_ptr[j]] * feature + kk ] += I_data[ col * feature + kk ] * dcsc_vals[ i + dcsc_col_ptr[j] ];
                    }
                }
            }
        }
    }
}


template<typename T, typename ITYPE>
void spmm_dcsc_jstream_compiler_vectorized( DCSC<T, ITYPE> &S, T *O, T *I, ITYPE feature, ITYPE Ti, ITYPE Tk, ITYPE chunk_size = 1 )
{
    T* dcsc_vals = S.vals;
    ITYPE *dcsc_rows = S.rows;
    ITYPE *dcsc_col_ptr = S.col_ptr;
    ITYPE *dcsc_cols = S.cols;
    ITYPE *panel_ptr = S.aux;
    T *I_data = I;
    T *O_data = O;

    __assume_aligned(dcsc_vals, ALLOC_ALIGNMENT);
    __assume_aligned(dcsc_rows, ALLOC_ALIGNMENT);
    __assume_aligned(dcsc_col_ptr, ALLOC_ALIGNMENT);
    __assume_aligned(dcsc_cols, ALLOC_ALIGNMENT);
    __assume_aligned(I_data, ALLOC_ALIGNMENT);
    __assume_aligned(O_data, ALLOC_ALIGNMENT);

    #pragma ivdep
    #pragma vector aligned
    #pragma vector temporal
    #pragma omp parallel for schedule (static, chunk_size)
    for (ITYPE row_panel = 0; row_panel < S.num_segments; row_panel++) { // I loop
        if (panel_ptr[row_panel] == panel_ptr[row_panel + 1]) {
            continue;
        }

        for (ITYPE k = 0; k < feature; k += Tk) {  // K loop

            for (ITYPE j = panel_ptr[row_panel]; j < panel_ptr[row_panel + 1]; j++) { // J loop
                ITYPE col = dcsc_cols[j]; // column number of the sparse matrix
                ITYPE nRows = dcsc_col_ptr[j + 1] - dcsc_col_ptr[j];
                ITYPE nRows_unrolled = ((nRows >> 3) << 3);

                for (ITYPE kk = k; kk < MIN(feature, k + Tk); kk ++) {

                    int i = 0;
                    #pragma ivdep
                    #pragma vector temporal
                    #pragma vector aligned
                    #pragma prefetch O_data:_MM_HINT_T1
                    for ( ; i < nRows_unrolled; i += 8) {
                        O_data[ dcsc_rows[(i+0) + dcsc_col_ptr[j]] * feature + kk ] += I_data[ col * feature + kk ] * dcsc_vals[(i+0) + dcsc_col_ptr[j] ];
                        O_data[ dcsc_rows[(i+1) + dcsc_col_ptr[j]] * feature + kk ] += I_data[ col * feature + kk ] * dcsc_vals[(i+1) + dcsc_col_ptr[j] ];
                        O_data[ dcsc_rows[(i+2) + dcsc_col_ptr[j]] * feature + kk ] += I_data[ col * feature + kk ] * dcsc_vals[(i+2) + dcsc_col_ptr[j] ];
                        O_data[ dcsc_rows[(i+3) + dcsc_col_ptr[j]] * feature + kk ] += I_data[ col * feature + kk ] * dcsc_vals[(i+3) + dcsc_col_ptr[j] ];
                        O_data[ dcsc_rows[(i+4) + dcsc_col_ptr[j]] * feature + kk ] += I_data[ col * feature + kk ] * dcsc_vals[(i+4) + dcsc_col_ptr[j] ];
                        O_data[ dcsc_rows[(i+5) + dcsc_col_ptr[j]] * feature + kk ] += I_data[ col * feature + kk ] * dcsc_vals[(i+5) + dcsc_col_ptr[j] ];
                        O_data[ dcsc_rows[(i+6) + dcsc_col_ptr[j]] * feature + kk ] += I_data[ col * feature + kk ] * dcsc_vals[(i+6) + dcsc_col_ptr[j] ];
                        O_data[ dcsc_rows[(i+7) + dcsc_col_ptr[j]] * feature + kk ] += I_data[ col * feature + kk ] * dcsc_vals[(i+7) + dcsc_col_ptr[j] ];
                    }

                    #pragma ivdep
                    #pragma vector temporal
                    #pragma vector aligned
                    #pragma prefetch O_data:_MM_HINT_T1
                    for ( ; i < nRows; i++) {
                        O_data[ dcsc_rows[i + dcsc_col_ptr[j]] * feature + kk ] += I_data[ col * feature + kk ] * dcsc_vals[ i + dcsc_col_ptr[j] ];
                    }
                }
            }
        }
    }
}

// SPLIT CSR JSTREAM
template <typename T, typename ITYPE>
long long spmm_split_csr_jstream( SPLIT_CSR<T, ITYPE> &S, T *I, T *O, ITYPE feature, ITYPE Ti, ITYPE Tk, ITYPE chunk_size = 1 )
{
    ITYPE num_row_panels = CEIL( S.nrows, Ti );
    ITYPE num_partitions = S.num_partitions;

    T* csr_v = S.vals;
    ITYPE *csr_c = S.cols;
    ITYPE *csr_ptr = S.row_ptr;
    T *I_data = I;
    T *O_data = O;

    #ifdef INTEL_COMPILER
        __assume_aligned(csr_v, 64);
        __assume_aligned(csr_c, 64);
        __assume_aligned(csr_ptr, 64);
        __assume_aligned(I_data, 64);
        __assume_aligned(O_data, 64);
    #endif

    ITYPE num_rows = S.nrows;

    long long start_cycle = readTSC();

    #pragma omp parallel for schedule(static, chunk_size)
    for ( ITYPE row_panel = 0; row_panel < num_row_panels; row_panel++ ) {

        for ( ITYPE partition = 0; partition < num_partitions; partition++ ) {

        // TIle / Panel start
            for ( ITYPE k = 0; k < feature; k += Tk ) {

                ITYPE row_start = row_panel * Ti;
                ITYPE row_end = MIN(row_start + Ti, num_rows);

                // Tile start -- Iterate over the row panel
                for ( ITYPE row = row_start; row < row_end; row++ ) {
                    ITYPE row_ptr = row + ( partition * num_rows );
                    ITYPE row_begin = csr_ptr[ row_ptr ];
                    ITYPE row_end = csr_ptr[row_ptr + 1];

                    // Output base register depends on the sparse row index - invariant in below loops
                    T* O_base_addr = &O_data[row * feature];

                    #ifdef REUSE_OUTPUT_FROM_REGS

                    /*
                        // Some funky code to prefetch the entire I row before use
                        for ( ITYPE kk = k; kk < MIN(feature, k + Tk); kk += 32 ) {
                            for ( ITYPE r = row_begin; r < row_end; r++) {
                                ITYPE col = csr_c[r];

                                T *input_ba = &I_data[col * feature];

                                _mm_prefetch( (input_ba + kk + 0), 1 );
                                _mm_prefetch( (input_ba + kk + 8), 1 );
                                _mm_prefetch( (input_ba + kk + 16), 1 );
                                _mm_prefetch( (input_ba + kk + 24), 1 );
                            }
                        }
                    */

                    // slice is 32 wide in this case
                    ITYPE kk_end = MIN(k + Tk, feature);
                    for ( ITYPE kk = k; kk < kk_end; kk += 32 ) {

                        // Load the output slice into registers
                        rtype Creg0 = vload( O_base_addr + kk + 0 );
                        rtype Creg1 = vload( O_base_addr + kk + 4 );
                        rtype Creg2 = vload( O_base_addr + kk + 8 );
                        rtype Creg3 = vload( O_base_addr + kk + 12 );
                        rtype Creg4 = vload( O_base_addr + kk + 16 );
                        rtype Creg5 = vload( O_base_addr + kk + 20 );
                        rtype Creg6 = vload( O_base_addr + kk + 24 );
                        rtype Creg7 = vload( O_base_addr + kk + 28 );

                        // Iterate over all non-zeroes of the row
                        for ( ITYPE r = row_begin; r < row_end; r++ ) {
                            ITYPE col = csr_c[r];
                            T val = csr_v[r];
                            rtype Areg = vset( val );

                            // The column index of the non-zero dictates the I base address
                            T *I_base_addr = &I_data[col * feature];

                            /*
                                // We might want to prefetch the slice of the next row
                                T *prefetch_base_addr = &I_data[csr_c[r + 1] * feature] + kk;
                                _mm_prefetch( (prefetch_base_addr + 0), 1 );
                                _mm_prefetch( (prefetch_base_addr + 8), 1 );
                                _mm_prefetch( (prefetch_base_addr + 16), 1 );
                                _mm_prefetch( (prefetch_base_addr + 24), 1 );
                            */

                            // Load the first half of the I slice
                            rtype Breg0 = vload( I_base_addr + kk + 0 );
                            rtype Breg1 = vload( I_base_addr + kk + 4 );
                            rtype Breg2 = vload( I_base_addr + kk + 8 );
                            rtype Breg3 = vload( I_base_addr + kk + 12 );

                            // First set of fused multiply adds performed in place
                            Creg0 = vfma( Areg, Breg0, Creg0 );
                            Creg1 = vfma( Areg, Breg1, Creg1 );
                            Creg2 = vfma( Areg, Breg2, Creg2 );
                            Creg3 = vfma( Areg, Breg3, Creg3 );

                            // Load the second half of the I slice
                            Breg0 = vload( I_base_addr + kk + 16 );
                            Breg1 = vload( I_base_addr + kk + 20 );
                            Breg2 = vload( I_base_addr + kk + 24 );
                            Breg3 = vload( I_base_addr + kk + 28 );


                            // Second set of fused multiply adds performed in place
                            Creg4 = vfma( Areg, Breg0, Creg4 );
                            Creg5 = vfma( Areg, Breg1, Creg5 );
                            Creg6 = vfma( Areg, Breg2, Creg6 );
                            Creg7 = vfma( Areg, Breg3, Creg7 );
                        }

                        // Store slice back into memory before processing the next sparse row
                        vstore( O_base_addr + kk + 0, Creg0 );
                        vstore( O_base_addr + kk + 4, Creg1 );
                        vstore( O_base_addr + kk + 8, Creg2 );
                        vstore( O_base_addr + kk + 12, Creg3 );
                        vstore( O_base_addr + kk + 16, Creg4 );
                        vstore( O_base_addr + kk + 20, Creg5 );
                        vstore( O_base_addr + kk + 24, Creg6 );
                        vstore( O_base_addr + kk + 28, Creg7 );
                    }


                    #else

                    for ( ITYPE r = row_begin; r < row_end; r++ ) {

                        ITYPE col = csr_c[r];
                        T val = csr_v[r];
                        rtype Areg = vset( val );

                        T* I_base_addr = &I_data[col * feature];
                        // K loops vetorized with avx2 instructions
                        for ( ITYPE kk = k; kk < MIN(feature, k + Tk); kk += 32 ) {

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

                            // reuse C
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

                    #endif // REUSE_OUTPUT_FROM_REGS
                }   // End processinig row panel for Tk wide Input and Output

            }

        }
    }


    // Single thread code -- maybe we should avoid printing from here
    // long long min_time = per_core_runtime[0];
    // for (ITYPE i = 0; i < 20; i++) {
    //     if (per_core_runtime[i] < min_time) {
    //         min_time = per_core_runtime[i];
    //     }
    // }

    // for (ITYPE i = 0; i < 20; i++) {
    //     std::cout << "Core: " << i << " -- " << per_core_runtime[i] / ((double) min_time) << std::endl;
    // }

    long long end_cycle = readTSC();
    return (end_cycle - start_cycle);
}


#endif // JSTREAM_H
