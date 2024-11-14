#ifndef SIMPLE_H
#define SIMPLE_H

#include "matrices/CSR.h"
#include "matrices/Matrix.h"

#include "utils/util.h"

#include <omp.h>

// Simple single core sparse-dense matrix multiplication implementation
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

#endif // SIMPLE_H
