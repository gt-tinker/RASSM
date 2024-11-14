#ifndef TACO_H
#define TACO_H

#include "matrices/CSF.h"

// Hand unrolling added as a test point, not used in the evaluation and have limited impact on performance
// #define UNROLL_CSF

template <typename T, typename ITYPE>
long long spmm_csf_compiler_vectorized(CSF<T, ITYPE> &S, T *I, T *O, ITYPE feature, ITYPE Ti, ITYPE Tj, ITYPE chunk_size, long long *per_core_runtime = nullptr, long long *per_panel_timing = nullptr)
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

    T *I_data = I;
    T *O_data = O;

    long long start_cycle = readTSC();

    #pragma omp parallel for schedule(OMP_SCHEDULE, chunk_size)
    for (ITYPE ioA = A1_pos[0]; ioA < A1_pos[1]; ioA++) {
        ITYPE io = A1_crd[ioA]; // tile row
        for (ITYPE joA = A2_pos[ioA]; joA < A2_pos[(ioA + 1)]; joA++) {
            ITYPE jo = A2_crd[joA]; // tile column
            for (ITYPE iiA = A3_pos[joA]; iiA < A3_pos[(joA + 1)]; iiA++) {
                ITYPE ii = A3_crd[iiA]; // row

                ITYPE jiA = A4_pos[iiA]; // column pointer
                #ifdef UNROLL_CSF
                    ITYPE j_unroll_end = A4_pos[iiA] + (((A4_pos[(iiA + 1)] - A4_pos[iiA]) >> 3) << 3);
                    for ( ; jiA < j_unroll_end; jiA += 8) {
                        #pragma ivdep
                        #pragma vector nontemporal(C_vals)
                        for (ITYPE k = 0; k < feature; k++) {
                            O_data[ ii * (feature + PADDING_C) + k ] +=
                                                                        C_vals[jiA] * I_data[ A4_crd[jiA] * (feature + PADDING_B) + k ]
                                                                    +   C_vals[jiA + 1] * I_data[ A4_crd[jiA + 1] * (feature + PADDING_B) + k ]
                                                                    +   C_vals[jiA + 2] * I_data[ A4_crd[jiA + 2] * (feature + PADDING_B) + k ]
                                                                    +   C_vals[jiA + 3] * I_data[ A4_crd[jiA + 3] * (feature + PADDING_B) + k ]
                                                                    +   C_vals[jiA + 4] * I_data[ A4_crd[jiA + 4] * (feature + PADDING_B) + k ]
                                                                    +   C_vals[jiA + 5] * I_data[ A4_crd[jiA + 5] * (feature + PADDING_B) + k ]
                                                                    +   C_vals[jiA + 6] * I_data[ A4_crd[jiA + 6] * (feature + PADDING_B) + k ]
                                                                    +   C_vals[jiA + 7] * I_data[ A4_crd[jiA + 7] * (feature + PADDING_B) + k ];

                        }
                    }
                #endif

                // cleanup
                for ( ; jiA < A4_pos[(iiA + 1)]; jiA++) {
                    ITYPE ji = A4_crd[jiA]; // column
                    // ITYPE jiB = jo * B2_dimension + ji;
                    #pragma GCC ivdep
                    for (ITYPE k = 0; k < feature; k++) {
                        // ITYPE kC = iiC * feature + k;
                        // ITYPE kB = jiB * feature + k;
                        // C_vals[kC] = C_vals[kC] + A_vals[jiA] * B_vals[kB];
                        O_data[ ii * (feature + PADDING_C) + k ] += C_vals[jiA] * I_data[ ji * (feature + PADDING_B) + k ];
                    }
                }
            }
        }
    }

    long long end_cycle = readTSC();

    return (end_cycle - start_cycle);
}

#endif // TACO_H

