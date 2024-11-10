#ifndef SDDMM_TILED_H
#define SDDMM_TILED_H

#include "matrices/CSR.h"
#include "matrices/DCSC.h"
#include "matrices/ATM.h"

template <typename T, typename ITYPE>
long long sddmm_parallel( CSR<T, ITYPE> &S, T *D1, T *D2, T *O, ITYPE K, ITYPE Ti_NA = -1, ITYPE Ti_NK = -1, ITYPE chunk_size = 1 )
{

    long long start_cycle = readTSC();

    #pragma omp parallel for schedule (OMP_SCHEDULE, chunk_size)
    for ( ITYPE row = 0; row < S.nrows; row++ ) {
        ITYPE row_start = S.row_ptr[row];
        ITYPE row_end = S.row_ptr[row + 1];
        for ( ITYPE ptr = row_start; ptr < row_end; ptr++ ) {
            T temp = 0;
            for ( ITYPE k = 0; k < K; k++ ) {
                temp += D1[ S.cols[ptr] * K + k ] * D2[ row * K + k];
            }
            O[ptr] = temp * S.vals[ptr];
        }
    }

    long long end_cycle = readTSC();

    return (end_cycle - start_cycle);
}

#ifdef INTEL_COMPILER
    #define HAND_VECTORIZED_SIMPLE_SDDMM
#endif

// fused sddmm where chunk_size is akin to row panel height
template <typename T, typename ITYPE>
long long sddmm_csr_row_panel_compiler_vectorized(CSR<T, ITYPE> &S, T *D1, T *D2, T *O, ITYPE K, ITYPE Ti = -1, ITYPE Tk_NA = -1, ITYPE chunk_size = 1)
{
    ITYPE *row_ptr = S.row_ptr;
    ITYPE *cols = S.cols;
    T* vals = S.vals;
    ITYPE num_rows = S.nrows;
    ITYPE num_panels = CEIL(num_rows, Ti);

    long long start_cycle = readTSC();

    #ifdef RUN_PINTOOL_MEMORY_TRACING
        PIN_ROI_BEGIN();
    #endif // RUN_PINTOOL_MEMORY_TRACING

    #ifdef HAND_VECTORIZED_SIMPLE_SDDMM

        // #pragma omp parallel for schedule(OMP_SCHEDULE, chunk_size)
        // for (ITYPE panel = 0; panel < num_panels; panel++ ) {
            // ITYPE panel_start = Ti * panel;
            // ITYPE panel_end = MIN( (panel_start + Ti), num_rows );
            // for (ITYPE i = panel_start; i < panel_end; i++) {
            #pragma omp parallel for schedule(OMP_SCHEDULE, chunk_size)
            for ( ITYPE i = 0; i < num_rows; i++ ) {
                T *D2_addr = &D2[i * (K + PADDING_C)];
                ITYPE row_start = row_ptr[i];
                ITYPE row_end = row_ptr[i + 1];

                for ( ITYPE j = row_start; j < row_end; j++ ) {
                    T *D1_addr = &D1[cols[j] * (K + PADDING_B)];

                    rtype sum1 = vsetzero();
                    rtype sum2 = vsetzero();
                    rtype sum3 = vsetzero();
                    rtype sum4 = vsetzero();
                    for ( ITYPE k = 0; k < K; k += 16) {
                        rtype bval1 = vload( D1_addr + k + 0 );
                        rtype bval2 = vload( D1_addr + k + 4 );
                        rtype bval3 = vload( D1_addr + k + 8 );
                        rtype bval4 = vload( D1_addr + k + 12 );

                        rtype cval1 = vload( D2_addr + k + 0 );
                        rtype cval2 = vload( D2_addr + k + 4 );
                        rtype cval3 = vload( D2_addr + k + 8 );
                        rtype cval4 = vload( D2_addr + k + 12 );


                        sum1 = vfma(bval1, cval1, sum1);
                        sum2 = vfma(bval2, cval2, sum2);
                        sum3 = vfma(bval3, cval3, sum3);
                        sum4 = vfma(bval4, cval4, sum4);
                    }

                    sum1 = vadd(sum1, sum2);
                    sum3 = vadd(sum3, sum4);
                    sum1 = vadd(sum1, sum3);
                    O[j] = hsum_double_avx(sum1);
                }
            }

            #pragma omp parallel for schedule(OMP_SCHEDULE, chunk_size)
            for (ITYPE i = 0; i < S.nnzs; i++) {
                O[i] *= vals[i];
            }
        // }
    #else

        #pragma omp parallel for schedule(OMP_SCHEDULE, chunk_size)
        for ( ITYPE row = 0; row < num_rows; row++ ) {
            ITYPE row_start = row_ptr[row];
            ITYPE row_end = row_ptr[row + 1];
            for ( ITYPE ptr = row_start; ptr < row_end; ptr++ ) {
                for ( ITYPE k = 0; k < K; k++ ) {
                    O[ptr] += D1[ cols[ptr] * (K + PADDING_B) + k ] * D2[ row * (K + PADDING_C) + k ];
                }
                O[ptr] *= vals[ptr];
            }
        }

    #endif // HAND_VECTORIZED_SIMPLE_SDDMM

    #ifdef RUN_PINTOOL_MEMORY_TRACING
        PIN_ROI_END();
    #endif // RUN_PINTOOL_MEMORY_TRACING

    long long end_cycle = readTSC();

    return (end_cycle - start_cycle);
}

template <typename T, typename ITYPE>
// long long sddmm_rstream(CSR<T, ITYPE> &S, T *D1, T *D2, CSR<T, ITYPE> &O, ITYPE K, ITYPE Ti, ITYPE Tk, ITYPE chunk_size)
long long sddmm_rstream(CSR<T, ITYPE> &S, T *D1, T *D2, T *O, ITYPE K, ITYPE Ti, ITYPE Tk, ITYPE chunk_size)
{

    ITYPE *row_ptr = S.row_ptr;
    ITYPE *cols = S.cols;
    T* vals = S.vals;
    ITYPE num_panels = CEIL(S.nrows, Ti);

    // __assume_aligned( D1, ALLOC_ALIGNMENT );
    // __assume_aligned( D2, ALLOC_ALIGNMENT );
    // __assume_aligned( O, ALLOC_ALIGNMENT );
    // __assume_aligned( cols, ALLOC_ALIGNMENT );
    // __assume_aligned( row_ptr, ALLOC_ALIGNMENT );

    auto start_cycle = readTSC();

    #pragma omp parallel for schedule(OMP_SCHEDULE, chunk_size)
    for ( ITYPE panel = 0; panel < num_panels; panel++ ) {
        ITYPE panel_start = panel * Ti;
        ITYPE panel_end = MIN( (panel_start + Ti), S.nrows );

        for ( ITYPE k = 0; k < K; k += Tk ) {
            for ( ITYPE row = panel_start; row < panel_end; row++ ) {
                ITYPE row_start = row_ptr[row];
                ITYPE row_end = row_ptr[row + 1];

                for ( ITYPE ptr = row_start; ptr < row_end; ptr++ ) {

                    for ( ITYPE kk = k; kk < MIN(K, k + Tk) ; kk++ ) {
                        O[ptr] += D2[ row * K + kk ] * D1[ cols[ptr] * K + kk ];
                    }
                    O[ptr] *= vals[ptr];
                }
            }
        }
    }

    auto end_cycle = readTSC();

    return (end_cycle - start_cycle);
}

#ifdef INTEL_COMPILER
// #define JSTREAM_SIMPLE_COMPILER
    #define JSTREAM_HAND_VECTORIZED
#endif

// jstream sddmm
template <typename T, typename ITYPE>
long long sddmm_jstream(DCSC<T, ITYPE> &S, T *D1, T *D2, T *O, ITYPE K, ITYPE Ti, ITYPE Tk, ITYPE chunk_size)
{

    ITYPE *panel_ptr = S.aux;
    ITYPE *col_ptr = S.col_ptr;
    ITYPE *col_ndx = S.cols;
    ITYPE *row_ndx = S.rows;
    T *vals = S.vals;

    __assume_aligned( D1, ALLOC_ALIGNMENT );
    __assume_aligned( D2, ALLOC_ALIGNMENT );
    __assume_aligned( col_ptr, ALLOC_ALIGNMENT );
    __assume_aligned( col_ndx, ALLOC_ALIGNMENT );
    __assume_aligned( row_ndx, ALLOC_ALIGNMENT );
    __assume_aligned( O, ALLOC_ALIGNMENT );
    __assume_aligned( vals, ALLOC_ALIGNMENT );

    long long start_cycle = readTSC();

    #ifndef JSTREAM_HAND_VECTORIZED

        #pragma ivdep
        #pragma vector aligned
        #pragma omp parallel for schedule(OMP_SCHEDULE, chunk_size)
        for ( ITYPE panel = 0; panel < S.num_segments; panel++ ) {
            ITYPE panel_start = panel_ptr[panel];
            ITYPE panel_end = panel_ptr[panel + 1];
            for ( ITYPE k = 0; k < K; k += Tk ) {
                for ( ITYPE panel_ptr = panel_start; panel_ptr < panel_end; panel_ptr++ ) {
                    ITYPE col = col_ndx[panel_ptr];
                    ITYPE col_start = col_ptr[panel_ptr];
                    ITYPE col_end = col_ptr[panel_ptr + 1];
                    T *D1_addr = &D1[ col * (K + PADDING_B) ];
                    for ( ITYPE col_ptr = col_start; col_ptr < col_end; col_ptr++ ) {
                        ITYPE row = row_ndx[col_ptr];
                        T *D2_addr = &D2[ row * (K + PADDING_C) ];
                        T temp = 0;
                        for ( ITYPE kk = k; kk < MIN( K, k + Tk); kk++ ) {
                            temp += D1_addr[ kk ] * D2_addr[ kk ];
                        }
                        O[ col_ptr ] += (temp * S.vals[ col_ptr ]);
                    }
                }
            }
        }

    #else // JSTREAM_HAND_VECTORIZED

        #pragma ivdep
		#pragma vector aligned
		#pragma omp parallel for schedule(dynamic, chunk_size)
        for ( ITYPE panel = 0; panel < S.num_segments; panel++ ) {
            ITYPE panel_start = panel_ptr[panel];
            ITYPE panel_end = panel_ptr[panel + 1];
            for ( ITYPE k = 0; k < K; k += Tk ) {
                for ( ITYPE panel_ptr = panel_start; panel_ptr < panel_end; panel_ptr++ ) {
                    ITYPE col = col_ndx[panel_ptr];
                    ITYPE col_start = col_ptr[panel_ptr];
                    ITYPE col_end = col_ptr[panel_ptr + 1];
                    T *D1_addr = &D1[ col * (K + PADDING_B) ];
                    for ( ITYPE col_ptr = col_start; col_ptr < col_end; col_ptr++ ) {
                        rtype sum1 = vsetzero();
                        rtype sum2 = vsetzero();
                        rtype sum3 = vsetzero();
                        rtype sum4 = vsetzero();

                        ITYPE row = row_ndx[col_ptr];
                        T *D2_addr = &D2[ row * (K + PADDING_C) ];

                        for (int kk = k; kk < MIN(k + Tk ,K); kk+=16) {
                            rtype bval1 = vload( D1_addr + kk + 0 );
                            rtype bval2 = vload( D1_addr + kk + 4 );
                            rtype bval3 = vload( D1_addr + kk + 8 );
                            rtype bval4 = vload( D1_addr + kk + 12 );

                            rtype cval1 = vload( D2_addr + kk + 0 );
                            rtype cval2 = vload( D2_addr + kk + 4 );
                            rtype cval3 = vload( D2_addr + kk + 8 );
                            rtype cval4 = vload( D2_addr + kk + 12 );

                            sum1 = vfma(bval1, cval1, sum1);
                            sum2 = vfma(bval2, cval2, sum2);
                            sum3 = vfma(bval3, cval3, sum3);
                            sum4 = vfma(bval4, cval4, sum4);
                        }

                        sum1 = vadd(sum1, sum2);
                        sum3 = vadd(sum3, sum4);
                        sum1 = vadd(sum1, sum3);

                        double tmp = hsum_double_avx(sum1);
                        O[col_ptr] += tmp;
                        // O[ col_ptr ] += (tmp * vals[ col_ptr ]);
                    }
                }
            }
        }

        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (ITYPE i = 0; i < S.nnzs; i++) {
            O[ i ] *= vals[i];
        }

    #endif // JSTREAM_HAND_VECTORIZED
    long long end_cycle = readTSC();

    return (end_cycle - start_cycle);
}

#ifdef INTEL_COMPILER
    #define HAND_VECTORIZED_SDDMM
#endif
#define RUN_UNROLLED

// D1 - B = col of non-zero
// D2 - C = row of non-zero
template<typename T, typename ITYPE>
long long sddmm_kstream(ATM<T, ITYPE> &S, T* D1, T *D2, T* O, ITYPE feature, ITYPE Ti, ITYPE Tj, ITYPE chunk_size = 1, long long *per_core_runtime = nullptr, long long *per_panel_timing=nullptr)
{
    T* csr_vals = S.vals;
    ITYPE *csr_cols = S.cols;
    ITYPE *tile_row_ptrs = S.tile_row_ptr;
    ITYPE *panel_ptr = S.panel_ptr;
    ITYPE *panel_offset = S.panel_offset;
    ITYPE *panel_start = S.panel_start;
    ITYPE *panel_Ti = S.panel_Ti;

    ITYPE num_rows = S.nrows;
    ITYPE num_panels = S.num_panels;

    #ifdef INTEL_COMPILER
    __assume_aligned( D1, ALLOC_ALIGNMENT );
    __assume_aligned( D2, ALLOC_ALIGNMENT );
    __assume_aligned( panel_start, ALLOC_ALIGNMENT );
    __assume_aligned( tile_row_ptrs, ALLOC_ALIGNMENT );
    __assume_aligned( panel_ptr, ALLOC_ALIGNMENT );
    __assume_aligned( csr_vals, ALLOC_ALIGNMENT );
    __assume_aligned( csr_cols, ALLOC_ALIGNMENT );
    __assume_aligned( O, ALLOC_ALIGNMENT );
    #endif

    #ifdef RUN_PINTOOL_MEMORY_TRACING
        PIN_ROI_BEGIN();
    #endif // RUN_PINTOOL_MEMORY_TRACING

    long long start_cycle = readTSC();

    #ifdef TRACK_PER_CORE_RUNTIME
    #pragma omp parallel shared(csr_vals, csr_cols, tile_row_ptrs, panel_ptr, D1, D2, O)
    {
        auto my_tid = omp_get_thread_num();
        per_core_runtime[my_tid] = readTSC();
    #endif

    #pragma ivdep
    #pragma vector aligned
    #pragma vector temporal
    #ifdef TRACK_PER_CORE_RUNTIME
        #pragma omp for schedule (OMP_SCHEDULE, chunk_size), nowait
    #else
        #pragma omp parallel for schedule(OMP_SCHEDULE, chunk_size)
    #endif
    for (ITYPE row_panel = 0; row_panel < num_panels; row_panel++) {

        #ifdef TRACK_PER_PANEL_RUNTIME
            per_panel_timing[row_panel] = readTSC();
        #endif // PER_PANEL_RUNTIME

        ITYPE num_tiles = panel_ptr[row_panel + 1] - panel_ptr[row_panel];
        ITYPE base_ptr = panel_offset[row_panel];
        ITYPE pTi = panel_Ti[row_panel];
        ITYPE panel_row_start = panel_start[row_panel];
        ITYPE panel_row_end = MIN( panel_start[row_panel] + pTi, num_rows );

        for (ITYPE tile = 0; tile < num_tiles; tile++) {

            for (ITYPE i = panel_row_start; i < panel_row_end ; i++) {
                ITYPE ptr = base_ptr + (i - panel_row_start) * num_tiles + tile;
                ITYPE tile_end = tile_row_ptrs[ptr + 1];

                ITYPE tile_start = tile_row_ptrs[ptr];
                #ifdef RUN_ASPT_SPECIAL
                    tile_start += FLOOR( (tile_end-tile_start), SPECIAL_THRESHOLD ) * SPECIAL_THRESHOLD;
                #endif

#ifndef HAND_VECTORIZED_SDDMM

                ITYPE tile_unroll_end = tile_start + (((tile_end - tile_start) >> 3) << 3);

                ITYPE j = tile_start;

                #ifdef RUN_UNROLLED
                for ( ; j < tile_unroll_end; j += 8 ) {
                    #pragma vector aligned
                    #pragma prefetch D1:_MM_HINT_T1 // Not sure if the prefetch hint actually works when using AVX2 on CPU
                    for (ITYPE k = 0; k < feature; k++) {
                        O[j + 0] += D1[csr_cols[j + 0] * (feature + PADDING_B) + k] * D2[i * (feature + PADDING_C) + k];
                        O[j + 1] += D1[csr_cols[j + 1] * (feature + PADDING_B) + k] * D2[i * (feature + PADDING_C) + k];
                        O[j + 2] += D1[csr_cols[j + 2] * (feature + PADDING_B) + k] * D2[i * (feature + PADDING_C) + k];
                        O[j + 3] += D1[csr_cols[j + 3] * (feature + PADDING_B) + k] * D2[i * (feature + PADDING_C) + k];
                        O[j + 4] += D1[csr_cols[j + 4] * (feature + PADDING_B) + k] * D2[i * (feature + PADDING_C) + k];
                        O[j + 5] += D1[csr_cols[j + 5] * (feature + PADDING_B) + k] * D2[i * (feature + PADDING_C) + k];
                        O[j + 6] += D1[csr_cols[j + 6] * (feature + PADDING_B) + k] * D2[i * (feature + PADDING_C) + k];
                        O[j + 7] += D1[csr_cols[j + 7] * (feature + PADDING_B) + k] * D2[i * (feature + PADDING_C) + k];
                    }

                    // scale by the csr value
                    #pragma ivdep
                    for (ITYPE k = 0; k < 8; k++) {
                        O[j + k] *= csr_vals[j + k];
                    }
                }
                #endif // RUN_UNROLLED

                for ( ; j < tile_end; j++ ) {
                    #pragma vector aligned
                    #pragma prefetch D1:_MM_HINT_T1
                    for (ITYPE k = 0; k < feature; k++) {
                        O[j + 0] += D1[csr_cols[j + 0] * (feature + PADDING_B) + k] * D2[i * (feature + PADDING_C) + k];
                    }
                    O[j] *= csr_vals[j]; // scale by csr value

                }

#else // HAND_VECTORIZED_SDDMM

                T *D2_addr = &D2[i * (feature + PADDING_C)];

                for ( ITYPE j = tile_start; j < tile_end; j++ ) {
                    T *D1_addr = &D1[csr_cols[j] * (feature + PADDING_B)];

                    rtype sum1 = vsetzero();
                    rtype sum2 = vsetzero();
                    rtype sum3 = vsetzero();
                    rtype sum4 = vsetzero();
                    for ( ITYPE k = 0; k < feature; k += 16) {
                        rtype bval1 = vload( D1_addr + k + 0 );
                        rtype bval2 = vload( D1_addr + k + 4 );
                        rtype bval3 = vload( D1_addr + k + 8 );
                        rtype bval4 = vload( D1_addr + k + 12 );

                        rtype cval1 = vload( D2_addr + k + 0 );
                        rtype cval2 = vload( D2_addr + k + 4 );
                        rtype cval3 = vload( D2_addr + k + 8 );
                        rtype cval4 = vload( D2_addr + k + 12 );


                        sum1 = vfma(bval1, cval1, sum1);
                        sum2 = vfma(bval2, cval2, sum2);
                        sum3 = vfma(bval3, cval3, sum3);
                        sum4 = vfma(bval4, cval4, sum4);
                    }

                    sum1 = vadd(sum1, sum2);
                    sum3 = vadd(sum3, sum4);
                    sum1 = vadd(sum1, sum3);
                    O[j] = hsum_double_avx(sum1) * csr_vals[j];
                }
#endif // HAND_VECTORIZED_SDDMM

            }
        }

        #ifdef TRACK_PER_PANEL_RUNTIME
            per_panel_timing[row_panel] = readTSC() - per_panel_timing[row_panel];
        #endif
    }

    #ifdef TRACK_PER_CORE_RUNTIME
        per_core_runtime[my_tid] = readTSC() - per_core_runtime[my_tid];
    }
    #endif

    #ifdef RUN_ASPT_SPECIAL
        if (S.run_special) {

            #pragma vector aligned
            #pragma omp parallel for schedule(OMP_SCHEDULE, chunk_size)
            for ( ITYPE special_ptr = 0; special_ptr < S.special_count; special_ptr++ ) {
                ITYPE i = S.special_row_ndx[special_ptr];
                ITYPE row_panel = S.special_row_panel[special_ptr];
                ITYPE panel_num_tiles = panel_ptr[row_panel + 1] - panel_ptr[row_panel];

                ITYPE ptr = panel_offset[ row_panel ] + ((i - panel_start[ row_panel ]) + 1) * panel_num_tiles;

                ITYPE row_start = tile_row_ptrs[ptr-1] + S.special_ptr[special_ptr];
                ITYPE row_end = row_start + SPECIAL_THRESHOLD;

                #ifndef HAND_VECTORIZED_SDDMM
                    ITYPE j;
                    #ifdef RUN_UNROLLED
                    for (j = row_start; j < row_end; j+=8) {
                        #pragma vector nontemporal (csr_vals)
                        #pragma prefetch D1:_MM_HINT_T1
                        for (ITYPE k = 0; k < feature; k++) {
                            O[j + 0] += D1[csr_cols[j + 0] * (feature + PADDING_B) + k] * D2[i * (feature + PADDING_C) + k];
                            O[j + 1] += D1[csr_cols[j + 1] * (feature + PADDING_B) + k] * D2[i * (feature + PADDING_C) + k];
                            O[j + 2] += D1[csr_cols[j + 2] * (feature + PADDING_B) + k] * D2[i * (feature + PADDING_C) + k];
                            O[j + 3] += D1[csr_cols[j + 3] * (feature + PADDING_B) + k] * D2[i * (feature + PADDING_C) + k];
                            O[j + 4] += D1[csr_cols[j + 4] * (feature + PADDING_B) + k] * D2[i * (feature + PADDING_C) + k];
                            O[j + 5] += D1[csr_cols[j + 5] * (feature + PADDING_B) + k] * D2[i * (feature + PADDING_C) + k];
                            O[j + 6] += D1[csr_cols[j + 6] * (feature + PADDING_B) + k] * D2[i * (feature + PADDING_C) + k];
                            O[j + 7] += D1[csr_cols[j + 7] * (feature + PADDING_B) + k] * D2[i * (feature + PADDING_C) + k];
                        }
                        #pragma ivdep
                        for (ITYPE k = 0; k < 8; k++) {
                            O[j + k] *= csr_vals[j + k];
                        }
                    }

                    #else // RUN_UNROLLED

                    for (j = row_start; j < row_end; j++) {
                        for (ITYPE k = 0; k < feature; k++) {
                            O[j + 0] += D1[csr_cols[j + 0] * (feature + PADDING_B) + k] * D2[i * (feature + PADDING_C) + k];
                        }
                        O[j] *= csr_vals[j];
                    }

                    #endif // RUN_UNROLLED

                #else // HAND_VECTORIZED_SDDMM

                T *D2_addr = &D2[i * (feature + PADDING_C)];

                for ( ITYPE j = row_start; j < row_end; j++ ) {
                    T *D1_addr = &D1[csr_cols[j] * (feature + PADDING_B)];

                    rtype sum1 = vsetzero();
                    rtype sum2 = vsetzero();
                    rtype sum3 = vsetzero();
                    rtype sum4 = vsetzero();
                    for ( ITYPE k = 0; k < feature; k += 16) {
                        rtype bval1 = vload( D1_addr + k + 0 );
                        rtype bval2 = vload( D1_addr + k + 4 );
                        rtype bval3 = vload( D1_addr + k + 8 );
                        rtype bval4 = vload( D1_addr + k + 12 );

                        rtype cval1 = vload( D2_addr + k + 0 );
                        rtype cval2 = vload( D2_addr + k + 4 );
                        rtype cval3 = vload( D2_addr + k + 8 );
                        rtype cval4 = vload( D2_addr + k + 12 );


                        sum1 = vfma(bval1, cval1, sum1);
                        sum2 = vfma(bval2, cval2, sum2);
                        sum3 = vfma(bval3, cval3, sum3);
                        sum4 = vfma(bval4, cval4, sum4);
                    }

                    sum1 = vadd(sum1, sum2);
                    sum3 = vadd(sum3, sum4);
                    sum1 = vadd(sum1, sum3);
                    O[j] = hsum_double_avx(sum1) * csr_vals[j];
                }

                #endif // HAND_VECTORIZED_SDDMM
            }
        }
    #endif

    long long end_cycle = readTSC();

    #ifdef RUN_PINTOOL_MEMORY_TRACING
        PIN_ROI_END();
    #endif // RUN_PINTOOL_MEMORY_TRACING

    return (end_cycle - start_cycle);
}

template <typename T, typename ITYPE>
long long sddmm_csf(CSF<T, ITYPE> &S, T* D1, T *D2, T* O, ITYPE feature, ITYPE Ti, ITYPE Tj, ITYPE chunk_size = 1, long long *per_core_runtime = nullptr, long long *per_panel_timing=nullptr)
{
    T *C_vals = (T *)S.vals;
    ITYPE *A1_pos = (ITYPE *)S.indices[0][0];
    ITYPE *A1_crd = (ITYPE *)S.indices[0][1];
    ITYPE *A2_pos = (ITYPE *)S.indices[1][0];
    ITYPE *A2_crd = (ITYPE *)S.indices[1][1];
    ITYPE *A3_pos = (ITYPE *)S.indices[2][0];
    ITYPE *A3_crd = (ITYPE *)S.indices[2][1];
    ITYPE *A4_pos = (ITYPE *)S.indices[3][0];
    ITYPE *A4_crd = (ITYPE *)S.indices[3][1];

    long long start_cycle = readTSC();

    #pragma omp parallel for schedule(OMP_SCHEDULE, chunk_size)
    for (ITYPE ioA = A1_pos[0]; ioA < A1_pos[1]; ioA++) {
        ITYPE io = A1_crd[ioA]; // tile row
        for (ITYPE joA = A2_pos[ioA]; joA < A2_pos[(ioA + 1)]; joA++) {
            ITYPE jo = A2_crd[joA]; // tile column
            for (ITYPE iiA = A3_pos[joA]; iiA < A3_pos[(joA + 1)]; iiA++) {
                ITYPE ii = A3_crd[iiA]; // row
                for (ITYPE jiA = A4_pos[iiA]; jiA < A4_pos[(iiA + 1)]; jiA++) {
                    ITYPE ji = A4_crd[jiA]; // column
                    for (ITYPE k = 0; k < feature; k++) {
                        O[jiA + 0] += D1[ji * (feature + PADDING_B) + k] * D2[ii * (feature + PADDING_C) + k];
                    }
                    O[jiA] *= C_vals[jiA];
                }
            }
        }
    }

    long long end_cycle = readTSC();

    return (end_cycle - start_cycle);
}

#endif // SDDMM_TILED_H

