
#include "utils/util.h"
#include <omp.h>

TYPE *A = nullptr;
TYPE *B = nullptr;
TYPE *Amkl = nullptr, *Bmkl = nullptr, *Cmkl = nullptr;
bool cache_flush_init = false;
size_t cache_flush_size = 0;

ITYPE m = 2000, k = 200, n = 1000;

// For sorting v_struct entries into row major format
int compare1(const void *a, const void *b)
{
	if (((struct v_struct *)a)->grp - ((struct v_struct *)b)->grp > 0) return 1;
	if (((struct v_struct *)a)->grp - ((struct v_struct *)b)->grp < 0) return -1;
	if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row > 0) return 1;
	if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row < 0) return -1;
	return (int) ((struct v_struct *)a)->col - ((struct v_struct *)b)->col;
}

// For sorting in column major format
int compare2(const void *a, const void *b)
{
	if (((struct v_struct *)a)->grp - ((struct v_struct *)b)->grp > 0) return 1;
	if (((struct v_struct *)a)->grp - ((struct v_struct *)b)->grp < 0) return -1;
	if (((struct v_struct *)a)->col - ((struct v_struct *)b)->col > 0) return 1;
	if (((struct v_struct *)a)->col - ((struct v_struct *)b)->col < 0) return -1;
	return (int) ((struct v_struct *)a)->row - ((struct v_struct *)b)->row;
}

void init_cache_flush(ITYPE size)
{
    // ITYPE num_entries = size / (sizeof(TYPE));

    if (!cache_flush_init) {

        print_status("Allocating cache flush of size: %ld MB\n", (size * sizeof(TYPE)) / (1024 * 1024) );

        cache_flush_size = size;

        A = new TYPE[size];
        B = new TYPE[size];

        assert( A && B && "Cache Flush init failed" );

        #pragma omp parallel for schedule(static)
        for(size_t i=0; i<cache_flush_size; i++) {
            A[i] = ((double) (i % 100)) / ((double) 100.0);
            B[i] = ((double) (i % 100)) / ((double) 100.0);
        }

        cache_flush_init = true;
    }
}

void free_cache_flush()
{
    delete[] A;
    delete[] B;
}

// Default size is 30 M
#pragma intel optimization_level 1
void cache_flush(ITYPE num_threads)
{
    if (cache_flush_init == false) {
        return;
    }

    TYPE total_sum = 0.0;

    #pragma omp parallel num_threads(num_threads)
    {
        #pragma omp for schedule(static) reduction(+ : total_sum)
        for(int i=0; i<cache_flush_size; i++)
        {
            // partial_sums[my_tid] += A[i] + B[i];
            total_sum += A[i] + B[i];
        }

    } // end omp parallel

    fprintf(stderr, "flush %lf\n", total_sum);
}


// PIN tool handling flag. Making PIN_ROI volatile to ensure that the function does
// not disappear
volatile int PIN_ROI = 0;
void __attribute__((noinline)) PIN_ROI_BEGIN()
{
    PIN_ROI = 1;
}

void __attribute__((noinline)) PIN_ROI_END()
{
    PIN_ROI = 0;
}

