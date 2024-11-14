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


/********************** J-Stream implementation based on column-wise traversals **********************/

// #define PRINT_THREAD_ID

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


    long long start_cycle = readTSC();
    // using the loop body from SC'20

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


    long long end_cycle = readTSC();

    return (end_cycle - start_cycle);
}

#endif // JSTREAM_H
