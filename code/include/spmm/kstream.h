#ifndef KSTREAM_H
#define KSTREAM_H

#include "config.h"
#include "matrices/ATM.h"
#include "matrices/CSR.h"
#include "matrices/Matrix.h"
#include "utils/util.h"
#include <omp.h>

// #define TRIPLE_TILED_KSTREAM

// k-stream
template<typename T, typename ITYPE>
long long spmm_atm_kstream_compiler_vectorized(ATM<T, ITYPE> &S, T* I, T *O, ITYPE feature, ITYPE Ti, ITYPE Tj, ITYPE chunk_size = 1, long long *per_core_runtime = nullptr, long long *per_panel_timing=nullptr)
{

    T* csr_vals = S.vals;
    ITYPE *csr_cols = S.cols;
    // ITYPE *csr_row_ptr = S.row_ptr;
    ITYPE *tile_row_ptrs = S.tile_row_ptr;
    ITYPE *panel_ptr = S.panel_ptr;


    ITYPE *panel_offset = S.panel_offset;
    ITYPE *panel_start = S.panel_start;
    ITYPE *panel_Ti = S.panel_Ti;

    T *I_data = I;
    T *O_data = O;

#ifdef TRIPLE_TILED_KSTREAM
    ITYPE Tk = Tj;
#endif // TRIPLE_TILED_KSTREAM

    ITYPE num_rows = S.nrows;
    ITYPE num_panels = S.num_panels;

    #ifdef INTEL_COMPILER
    __assume_aligned(tile_row_ptrs, ALLOC_ALIGNMENT);
    __assume_aligned(panel_ptr, ALLOC_ALIGNMENT);
    __assume_aligned(csr_vals, ALLOC_ALIGNMENT);
    __assume_aligned(csr_cols, ALLOC_ALIGNMENT);
    // __assume_aligned(csr_row_ptr, ALLOC_ALIGNMENT);
    __assume_aligned(I_data, ALLOC_ALIGNMENT);
    __assume_aligned(O_data, ALLOC_ALIGNMENT);
    #endif

    #ifdef RUN_PINTOOL_MEMORY_TRACING
        PIN_ROI_BEGIN();
    #endif // RUN_PINTOOL_MEMORY_TRACING

    long long start_cycle = readTSC();

    #ifdef TRACK_PER_CORE_RUNTIME
    #pragma omp parallel shared(csr_vals, csr_cols, tile_row_ptrs, panel_ptr, I_data, O_data)
    {
        auto my_tid = omp_get_thread_num();
        per_core_runtime[my_tid] = readTSC();
    #endif

    #pragma ivdep
    // #pragma GCC ivdep
    #pragma vector aligned
    #pragma vector temporal
    #ifdef TRACK_PER_CORE_RUNTIME
    #pragma omp for schedule (OMP_SCHEDULE, chunk_size), nowait
    #else
    #pragma omp parallel for schedule(OMP_SCHEDULE, 1)
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


    #ifdef TRIPLE_TILED_KSTREAM
        for ( ITYPE kk = 0; kk < feature; kk += Tk ) {
    #endif

        for (ITYPE tile = 0; tile < num_tiles; tile++) {

            for (ITYPE i = panel_row_start; i < panel_row_end ; i++) {
                ITYPE ptr = base_ptr + (i - panel_row_start) * num_tiles + tile;

                ITYPE tile_start = tile_row_ptrs[ptr];
                ITYPE tile_end = tile_row_ptrs[ptr + 1];

                #if defined(RUN_ASPT_SPECIAL) || defined(ASPT_SPECIAL_SIMD_PARALLEL)
                    tile_start += FLOOR( (tile_end-tile_start), SPECIAL_THRESHOLD ) * SPECIAL_THRESHOLD;
                #endif

                ITYPE tile_unroll_end = tile_start + (((tile_end - tile_start) >> 3) << 3);

                ITYPE j = tile_start;
                for ( ; j < tile_unroll_end; j += 8 ) {
                    #pragma GCC ivdep
                    #pragma vector nontemporal (csr_vals)
                    // #pragma vector aligned
                    #pragma prefetch I_data:_MM_HINT_T1 // Not sure if the prefetch hint actually works when using AVX2 on CPU
                #ifdef TRIPLE_TILED_KSTREAM
                    for ( ITYPE k = kk; k < MIN(feature, kk + Tk); k++ ) {
                #else
                    for (ITYPE k = 0; k < feature; k++) {
                #endif
                        O_data[ i * (feature + PADDING_C) + k ] +=
                                        csr_vals[j+0] * I_data[ csr_cols[j+0] * (feature + PADDING_B) + k ]
                                    +   csr_vals[j+1] * I_data[ csr_cols[j+1] * (feature + PADDING_B) + k ]
                                    +   csr_vals[j+2] * I_data[ csr_cols[j+2] * (feature + PADDING_B) + k ]
                                    +   csr_vals[j+3] * I_data[ csr_cols[j+3] * (feature + PADDING_B) + k ]
                                    +   csr_vals[j+4] * I_data[ csr_cols[j+4] * (feature + PADDING_B) + k ]
                                    +   csr_vals[j+5] * I_data[ csr_cols[j+5] * (feature + PADDING_B) + k ]
                                    +   csr_vals[j+6] * I_data[ csr_cols[j+6] * (feature + PADDING_B) + k ]
                                    +   csr_vals[j+7] * I_data[ csr_cols[j+7] * (feature + PADDING_B) + k ];
                    }
                }
                for ( ; j < tile_end; j++ ) {
                    #pragma ivdep
                    #pragma vector nontemporal (csr_vals)
                    // #pragma vector aligned
                    #pragma prefetch I_data:_MM_HINT_T1
                #ifdef TRIPLE_TILED_KSTREAM
                    for ( ITYPE k = kk; k < MIN(feature, kk + Tk); k++ ) {
                #else
                    for (ITYPE k = 0; k < feature; k++) {
                #endif
                        O_data[ i * (feature + PADDING_C) + k ] += csr_vals[j] * I_data[ csr_cols[j] * (feature + PADDING_B) + k ];
                    }
                }
            }
        }

    #ifdef TRIPLE_TILED_KSTREAM
        }
    #endif // TRIPLE_TILED_KSTREAM

        #ifdef TRACK_PER_PANEL_RUNTIME
            per_panel_timing[row_panel] = readTSC() - per_panel_timing[row_panel];
        #endif
    }

    #ifdef TRACK_PER_CORE_RUNTIME
        per_core_runtime[my_tid] = readTSC() - per_core_runtime[my_tid];
    }
    #endif

    #if defined(RUN_ASPT_SPECIAL)
        if (S.run_special) {

            #pragma ivdep
            #pragma vector aligned
            #pragma omp parallel for schedule(OMP_SCHEDULE, chunk_size)
            for ( ITYPE special_ptr = 0; special_ptr < S.special_count; special_ptr++ ) {
                ITYPE i = S.special_row_ndx[special_ptr];
                ITYPE row_panel = S.special_row_panel[special_ptr];
                ITYPE panel_num_tiles = panel_ptr[row_panel + 1] - panel_ptr[row_panel];

                ITYPE ptr = panel_offset[ row_panel ] + ((i - panel_start[ row_panel ])) * panel_num_tiles;
                ITYPE row_start = tile_row_ptrs[ptr] + S.special_ptr[special_ptr];

                // ITYPE ptr = panel_offset[ row_panel ] + ((i - panel_start[ row_panel ]) + 1) * panel_num_tiles;
                // ITYPE row_start = tile_row_ptrs[ptr-1] + S.special_ptr[special_ptr];
                ITYPE row_end = row_start + SPECIAL_THRESHOLD;

                T temp_sum[feature] = {0,};
                for (ITYPE j = row_start; j < row_end; j+=8) {
                    #pragma ivdep
				    #pragma vector nontemporal (csr_vals)
				    #pragma prefetch I_data:_MM_HINT_T1
                    for (ITYPE k = 0; k < feature; k++) {
                        temp_sum[ k ] +=
                                        csr_vals[j+0] * I_data[ csr_cols[j+0] * (feature + PADDING_B) + k ]
                                    +   csr_vals[j+1] * I_data[ csr_cols[j+1] * (feature + PADDING_B) + k ]
                                    +   csr_vals[j+2] * I_data[ csr_cols[j+2] * (feature + PADDING_B) + k ]
                                    +   csr_vals[j+3] * I_data[ csr_cols[j+3] * (feature + PADDING_B) + k ]
                                    +   csr_vals[j+4] * I_data[ csr_cols[j+4] * (feature + PADDING_B) + k ]
                                    +   csr_vals[j+5] * I_data[ csr_cols[j+5] * (feature + PADDING_B) + k ]
                                    +   csr_vals[j+6] * I_data[ csr_cols[j+6] * (feature + PADDING_B) + k ]
                                    +   csr_vals[j+7] * I_data[ csr_cols[j+7] * (feature + PADDING_B) + k ];
                    }
                }
                #pragma ivdep
                for (ITYPE k = 0; k < feature; k++) {
                    #pragma omp atomic
                    O_data[ i * (feature + PADDING_C) + k ] += temp_sum[ k ];
                }

            }
        }

    #elif defined(ASPT_SPECIAL_SIMD_PARALLEL)
        if (S.run_special) {
            for ( ITYPE special_ptr = 0; special_ptr < S.special_count; special_ptr++ ) {
                ITYPE i = S.special_row_ndx[special_ptr];
                ITYPE row_panel = S.special_row_panel[special_ptr];
                ITYPE panel_num_tiles = panel_ptr[row_panel + 1] - panel_ptr[row_panel];
                ITYPE ptr = panel_offset[ row_panel ] + ((i - panel_start[ row_panel ]) + 1) * panel_num_tiles;

                ITYPE row_start = tile_row_ptrs[ptr-1] + S.special_ptr[special_ptr];
                ITYPE row_end = row_start + SPECIAL_THRESHOLD;

                #pragma ivdep
                #pragma vector aligned
                #pragma omp parallel for schedule(OMP_SCHEDULE, chunk_size)
                for (ITYPE k = 0; k < feature; k += 8) {
                    for (ITYPE j = row_start; j < row_end; j++) {
                        O_data[ i * (feature + PADDING_C) + k + 0 ] += csr_vals[j] * I_data[ csr_cols[j] * (feature + PADDING_B) + k + 0 ];
                        O_data[ i * (feature + PADDING_C) + k + 1 ] += csr_vals[j] * I_data[ csr_cols[j] * (feature + PADDING_B) + k + 1 ];
                        O_data[ i * (feature + PADDING_C) + k + 2 ] += csr_vals[j] * I_data[ csr_cols[j] * (feature + PADDING_B) + k + 2 ];
                        O_data[ i * (feature + PADDING_C) + k + 3 ] += csr_vals[j] * I_data[ csr_cols[j] * (feature + PADDING_B) + k + 3 ];
                        O_data[ i * (feature + PADDING_C) + k + 4 ] += csr_vals[j] * I_data[ csr_cols[j] * (feature + PADDING_B) + k + 4 ];
                        O_data[ i * (feature + PADDING_C) + k + 5 ] += csr_vals[j] * I_data[ csr_cols[j] * (feature + PADDING_B) + k + 5 ];
                        O_data[ i * (feature + PADDING_C) + k + 6 ] += csr_vals[j] * I_data[ csr_cols[j] * (feature + PADDING_B) + k + 6 ];
                        O_data[ i * (feature + PADDING_C) + k + 7 ] += csr_vals[j] * I_data[ csr_cols[j] * (feature + PADDING_B) + k + 7 ];
                    }
                }
            }
        }
    #endif // RUN_ASPT_SPECIAL || ASPT_SPECIAL_SIMD_PARALLEL

    long long end_cycle = readTSC();

    #ifdef RUN_PINTOOL_MEMORY_TRACING
        PIN_ROI_END();
    #endif // RUN_PINTOOL_MEMORY_TRACING

    return (end_cycle - start_cycle);
}


// template <typename T, typename ITYPE>
// void spmm_kstream_memstream()


#endif // KSTREAM_H
