#ifndef EXPERIMENTS_H
#define EXPERIMENTS_H

#include "Reader.h"
#include "utils/Statistics.h"
#include "utils/util.h"
#include "config.h"
#include "global.h"


#include "matrices/ATM.h"
#include "matrices/CSR.h"
#include "matrices/CSF.h"
#include "matrices/DCSC.h"
#include "matrices/Matrix.h"

#ifdef INTEL_COMPILER
    #include "spmm/simple.h"
    #include "spmm/jstream.h"
    #include "spmm/kstream.h"
    #include "spmm/hybrid.h"
    #include "spmm/taco.h"

    #include "sddmm/tiled.h"
    #include "sddmm/simple.h"

    #include "spmv/simple.h"
    #include "spmv/tiled.h"
#else   // Might be using a compiler that does not recoganize Intel intrinsics
    #include "spmm/kstream.h"
    #include "spmm/simple.h"
    #include "spmm/taco.h"

    #include "sddmm/simple.h"
    #include "sddmm/tiled.h"
#endif

#include <sched.h>

// #define STRIDE 32
#define START_POINT 0

// #define VEC_EVENTS

#define CACHE_FLUSH_EXPR


#ifdef CACHE_FLUSH_EXPR
    #define CACHE_FLUSH     {                                                                       \
                                cache_flush(num_threads);                                           \
                                lfence();                                                       \
                            }
#else
    #define CACHE_FLUSH // Nothing
#endif


// #ifndef INTEL_MKL
// #define INTEL_MKL
// #endif

#ifdef INTEL_MKL
    #include "mkl_spblas.h"
#endif

// #define RUN_ROW_PANEL_COMPILER

// #define RUN_CSR_EXPR
// #define RUN_SIMPLE_CSR

// #define RUN_CSR_WORKLIST_EXPR

// #define RUN_SIMPLE_COMPILER_CSR

// #define RUN_CSR_JSTREAM_EXPR

// #define RUN_CSR_JSTREAM_COMPILER_EXPR

// #define RUN_CSR_KSTREAM_EXPR

// #define RUN_DCSC_JSTREAM_EXPR

// #define RUN_DCSC_JSTREAM_COMPILER_EXPR
// #define RUN_SPLIT_JSTREAM_EXPR
// #define RUN_CSR_WORKLIST
// #define RUN_DCSC_WORKLIST

// #define RUN_HYB_EXPR

// TODO: Fix me
// #define RUN_CSR_ATM_KSTREAM

// #define RUN_DYN_HYB_EXPR

#define RUN_CSF_EXPR


// #define RUN_DYN_HYB_COMP_VEC_EXPR

// #define RUN_DYN_HYB_DM_EXPR // Data movement only

// Run n warmup iterations
#define RUN_WARMUP
#define RUN_CORRECTNESS_CHECK


// Helper to reset matrices in parallel
template <typename T, typename ITYPE>
static void reset_matrix_helper(T *arr, size_t size)
{
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size; i++) {
        arr[i] = ((T) 0);
    }
}

template<typename T, typename ITYPE>
static void reset_matrix_helper(T **arr, size_t size, ITYPE layers)
{
    for (ITYPE l = 0; l < layers; l++) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < size; i++) {
            arr[l][i] = ((T) 0);
        }
    }
}


#ifdef LARGE_ARRAY
    #define MEM_RESET(arr, size, layers) reset_matrix_helper(arr, size)
#else
    #define MEM_RESET(arr, size, layers) reset_matrix_helper<T, ITYPE>(arr, size, layers)
#endif

#pragma intel optimization_level 0
void reset_matrix_helper_slow(double *arr, size_t size, int num_threads = 20)
{
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (size_t i = 0; i < size; i++) {
        arr[i] = (double) 0;
    }

}


// Macros to call the function if tracking stats is enabled otherwise nothing
#ifdef TRACK_PARALLEL_STATS
    #define PERF_START(p) parallel_perf_start_helper(p)
    #define PERF_STOP(p, s) parallel_perf_stop_helper(p, s)
    #define DRAM_START(p) parallel_dram_start_helper(p)
    #define DRAM_STOP(p, s) parallel_dram_stop_helper(p, s)
    #define PERF_PRINT(type, stats, num_threads, num_events) parallel_perf_print_helper<long long, ITYPE>(type, stats, num_threads, num_events)
#else
    #define PERF_START(p)
    #define PERF_STOP(p, s)
    #define DRAM_START(p)
    #define DRAM_STOP(p, s)
    #define PERF_PRINT(type, stats, num_threads, num_events)
#endif



// f - spmm function to call
// p - performance counter array
// s - stats array
// e - performance counter type
// A - The sparse matrix pointer
// Ti - Row panel height
// Tk - Tk
#ifdef TRACK_PARALLEL_STATS

#ifdef LARGE_ARRAY

#define RUN_PERF_HELPER(f,p,s,e,A,Ti,Tk){                                                                               \
                                            stats_t<long long, ITYPE> cycle_counts;                                     \
                                            cycle_counts.name = "Cycle Counts";                                         \
                                                                                                                        \
                                            for ( ITYPE i = 0; i < n; i++ ) {                                           \
                                                                                                                        \
                                                T *input = &B[ (i % layers) * (ncols * feature) ];                      \
                                                T *output = &C[ (i % layers) * (nrows * feature) ];                     \
                                                                                                                        \
                                                CACHE_FLUSH;                                                            \
                                                                                                                        \
                                                PERF_START(p);                                                          \
                                                long long duration = f(*A, input, output, feature, Ti, Tk, chunk_size); \
                                                PERF_STOP(p, s);                                                        \
                                                                                                                        \
                                                cycle_counts.insert(duration);                                          \
                                                                                                                        \
                                            }                                                                           \
                                            ITYPE num_events_to_print = perf::NUM_EVENTS_PER_SET;                       \
                                            if (e == perf_event_set::CACHE_STALL_EVENTS) {                              \
                                                num_events_to_print = NUM_CACHE_STALL_EVENTS;                           \
                                            } else if (e == perf_event_set::NAMED_CACHE_EVENTS) {                       \
                                                num_events_to_print = NUM_NAMED_CACHE_EVENTS;                           \
                                            } else if (e == perf_event_set::L3_CACHE_EVENTS) {                          \
                                                num_events_to_print = NUM_L3_CACHE_EVENTS;                              \
                                            } else if (e == perf_event_set::NAMED_STALL_EVENTS) {                       \
                                                num_events_to_print = NUM_NAMED_STALL_EVENTS;                           \
                                            }                                                                           \
                                            cycle_counts.process();                                                     \
                                            cycle_counts.print();                                                       \
                                            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                            std::cout << "Median Time: " << median_time << std::endl;                   \
                                            PERF_PRINT(e, s, num_threads, num_events_to_print);                         \
                                            size_t flop_count = ((size_t) 2) * ((size_t) A->nnzs) * ((size_t) feature); \
                                            double gflops = ((double) flop_count) / median_time / 1E9;                  \
                                            std::cout << "GFLOPS: " << gflops << std::endl;                             \
                                        }

#else // LARGE_ARRAY

#define RUN_PERF_HELPER(f,p,s,e,A,Ti,Tk){                                                                               \
                                            stats_t<long long, ITYPE> cycle_counts;                                     \
                                            cycle_counts.name = "Cycle Counts";                                         \
                                                                                                                        \
                                            for ( ITYPE i = 0; i < n; i++ ) {                                           \
                                                                                                                        \
                                                T *input = &(B[ (i % layers) ][0]);                                     \
                                                T *output = &(C[ (i % layers) ][0]);                                    \
                                                                                                                        \
                                                CACHE_FLUSH;                                                            \
                                                                                                                        \
                                                PERF_START(p);                                                          \
                                                long long duration = f(*A, input, output, feature, Ti, Tk, chunk_size); \
                                                PERF_STOP(p, s);                                                        \
                                                                                                                        \
                                                cycle_counts.insert(duration);                                          \
                                                                                                                        \
                                            }                                                                           \
                                            cycle_counts.process();                                                     \
                                            cycle_counts.print();                                                       \
                                            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                            std::cout << "Median Time: " << median_time << std::endl;                   \
                                            ITYPE num_events_to_print = perf::NUM_EVENTS_PER_SET;                       \
                                            if (e == perf_event_set::CACHE_STALL_EVENTS) {                              \
                                                num_events_to_print = NUM_CACHE_STALL_EVENTS;                           \
                                            } else if (e == perf_event_set::NAMED_CACHE_EVENTS) {                       \
                                                num_events_to_print = NUM_NAMED_CACHE_EVENTS;                           \
                                            } else if (e == perf_event_set::L3_CACHE_EVENTS) {                          \
                                                num_events_to_print = NUM_L3_CACHE_EVENTS;                              \
                                            } else if (e == perf_event_set::NAMED_STALL_EVENTS) {                       \
                                                num_events_to_print = NUM_NAMED_STALL_EVENTS;                           \
                                            } else if (e == perf_event_set::FILL_EVENTS) {                              \
                                                num_events_to_print = NUM_FILL_EVENTS;                                  \
                                            }                                                                           \
                                            PERF_PRINT(e, s, num_threads, num_events_to_print);                         \
                                            size_t flop_count = ((size_t) 2) * ((size_t) A->nnzs) * ((size_t) feature); \
                                            double gflops = ((double) flop_count) / median_time / 1E9;                  \
                                            std::cout << "flop count: " << flop_count << std::endl;                       \
                                            std::cout << "GFLOPS: " << gflops << std::endl;                             \
                                            std::cout << "Mean Time: " << ((double) cycle_counts.mean) / CLOCK_FREQUENCY << std::endl; \
                                            std::cout << "Std Dev: " << ((double) cycle_counts.std_dev) / CLOCK_FREQUENCY << std::endl; \
                                            std::cout << "RAW: ";                                                       \
                                            for ( ITYPE i = 0; i < n; i++ ) {                                           \
                                                std::cout << ((double) cycle_counts.records[i]) / CLOCK_FREQUENCY << ", "; \
                                            }                                                                           \
                                            std::cout << std::endl;                                                     \
                                        }

#endif // LARGE_ARRAY



#ifdef LARGE_ARRAY

#define RUN_HYBRID_PERF_HELPER(f,p,s,e,A,wl,ppt){                                                                       \
                                            stats_t<long long, ITYPE> cycle_counts;                                     \
                                            cycle_counts.name = "Cycle Counts";                                         \
                                                                                                                        \
                                            for ( ITYPE i = 0; i < n; i++ ) {                                           \
                                                                                                                        \
                                                T *input = &B[ (i % layers) * (ncols * feature) ];                      \
                                                T *output = &C[ (i % layers) * (nrows * feature) ];                     \
                                                                                                                        \
                                                CACHE_FLUSH;                                                            \
                                                                                                                        \
                                                PERF_START(p);                                                          \
                                                long long duration = f(*A, input, output, feature, wl, ppt, num_threads, chunk_size); \
                                                PERF_STOP(p, s);                                                        \
                                                                                                                        \
                                                cycle_counts.insert(duration);                                          \
                                                                                                                        \
                                            }                                                                           \
                                            cycle_counts.process();                                                     \
                                            cycle_counts.print();                                                       \
                                            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                            std::cout << "Median Time: " << median_time << std::endl;                   \
                                            PERF_PRINT(e, s, num_threads, (e == perf_event_set::CACHE_STALL_EVENTS || e == perf_event_set::NAMED_STALL_EVENTS)  ? NUM_CACHE_STALL_EVENTS :  perf::NUM_EVENTS_PER_SET);                    \
                                            size_t flop_count = ((size_t) 2) * ((size_t) A->nnzs) * ((size_t) feature); \
                                            double gflops = ((double) flop_count) / median_time / 1E9;                  \
                                        }

#else // LARGE_ARRAY

#define RUN_HYBRID_PERF_HELPER(f,p,s,e,A,wl,ppt){                                                                       \
                                            stats_t<long long, ITYPE> cycle_counts;                                     \
                                            cycle_counts.name = "Cycle Counts";                                         \
                                                                                                                        \
                                            for ( ITYPE i = 0; i < n; i++ ) {                                           \
                                                                                                                        \
                                                T *input = &(B[ (i % layers) ][0]);                                     \
                                                T *output = &(C[ (i % layers) ][0]);                                    \
                                                                                                                        \
                                                CACHE_FLUSH;                                                            \
                                                                                                                        \
                                                PERF_START(p);                                                          \
                                                long long duration = f(*A, input, output, feature, wl, ppt, num_threads, chunk_size); \
                                                PERF_STOP(p, s);                                                        \
                                                                                                                        \
                                                cycle_counts.insert(duration);                                          \
                                                                                                                        \
                                            }                                                                           \
                                            cycle_counts.process();                                                     \
                                            cycle_counts.print();                                                       \
                                            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                            std::cout << "Median Time: " << median_time << std::endl;                   \
                                            ITYPE num_events_to_print = perf::NUM_EVENTS_PER_SET;                       \
                                            if (e == perf_event_set::CACHE_STALL_EVENTS) {                              \
                                                num_events_to_print = NUM_CACHE_STALL_EVENTS;                           \
                                            } else if (e == perf_event_set::NAMED_CACHE_EVENTS) {                       \
                                                num_events_to_print = NUM_NAMED_CACHE_EVENTS;                           \
                                            } else if (e == perf_event_set::L3_CACHE_EVENTS) {                          \
                                                num_events_to_print = NUM_L3_CACHE_EVENTS;                              \
                                            } else if (e == perf_event_set::NAMED_STALL_EVENTS) {                       \
                                                num_events_to_print = NUM_NAMED_STALL_EVENTS;                           \
                                            } else if (e == perf_event_set::FILL_EVENTS) {                              \
                                                num_events_to_print = NUM_FILL_EVENTS;                                  \
                                            }                                                                           \
                                            PERF_PRINT(e, s, num_threads, num_events_to_print);                         \
                                            size_t flop_count = ((size_t) 2) * ((size_t) A->nnzs) * ((size_t) feature); \
                                            double gflops = ((double) flop_count) / median_time / 1E9;                  \
                                            std::cout << "flop count: " << flop_count << std::endl;                       \
                                            std::cout << "GFLOPS: " << gflops << std::endl;                             \
                                            std::cout << "Mean Time: " << ((double) cycle_counts.mean) / CLOCK_FREQUENCY << std::endl; \
                                            std::cout << "Std Dev: " << ((double) cycle_counts.std_dev) / CLOCK_FREQUENCY << std::endl; \
                                            std::cout << "RAW: ";                                                       \
                                            for ( ITYPE i = 0; i < n; i++ ) {                                           \
                                                std::cout << ((double) cycle_counts.records[i]) / CLOCK_FREQUENCY << ", "; \
                                            }                                                                           \
                                            std::cout << std::endl;                                                     \
                                        }
#endif // LARGE_ARRAY


#ifdef LARGE_ARRAY

#define RUN_MKL_PERF_HELPER(p,s,e){                                                                                     \
                                            stats_t<long long, ITYPE> cycle_counts;                                     \
                                            cycle_counts.name = "Cycle Counts";                                         \
                                                                                                                        \
                                            for ( ITYPE i = 0; i < n; i++ ) {                                           \
                                                                                                                        \
                                                T *input = &B[ (i % layers) * (ncols * feature) ];                      \
                                                T *output = &C[ (i % layers) * (nrows * feature) ];                     \
                                                                                                                        \
                                                CACHE_FLUSH;                                                            \
                                                                                                                        \
                                                PERF_START(p);                                                          \
                                                auto start_cycle = readTSC();                                           \
                                                mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, mkl_alpha, mkl_S, mkl_S_desc, SPARSE_LAYOUT_ROW_MAJOR, input, mkl_feature, mkl_feature, mkl_beta, output, mkl_feature ); \
                                                auto end_cycle = readTSC();                                             \
                                                PERF_STOP(p, s);                                                        \
                                                                                                                        \
                                                cycle_counts.insert( ( end_cycle - start_cycle) );                      \
                                                                                                                        \
                                            }                                                                           \
                                            cycle_counts.process();                                                     \
                                            cycle_counts.print();                                                       \
                                            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                            std::cout << "Median Time: " << median_time << std::endl;                   \
                                            PERF_PRINT(e, s, num_threads, (e == perf_event_set::CACHE_STALL_EVENTS || e == perf_event_set::NAMED_STALL_EVENTS)  ? NUM_CACHE_STALL_EVENTS :  perf::NUM_EVENTS_PER_SET);                    \
                                        }

#else // LARGE_ARRAY

#define RUN_MKL_PERF_HELPER(p,s,e){                                                                                     \
                                            stats_t<long long, ITYPE> cycle_counts;                                     \
                                            cycle_counts.name = "Cycle Counts";                                         \
                                                                                                                        \
                                            for ( ITYPE i = 0; i < n; i++ ) {                                           \
                                                                                                                        \
                                                T *input = &(B[ (i % layers) ][0]);                                     \
                                                T *output = &(C[ (i % layers) ][0]);                                    \
                                                                                                                        \
                                                CACHE_FLUSH;                                                            \
                                                                                                                        \
                                                PERF_START(p);                                                          \
                                                auto start_cycle = readTSC();                                           \
                                                mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, mkl_alpha, mkl_S, mkl_S_desc, SPARSE_LAYOUT_ROW_MAJOR, input, mkl_feature, mkl_feature, mkl_beta, output, mkl_feature ); \
                                                auto end_cycle = readTSC();                                             \
                                                PERF_STOP(p, s);                                                        \
                                                                                                                        \
                                                cycle_counts.insert( (end_cycle - start_cycle) );                       \
                                                                                                                        \
                                            }                                                                           \
                                            ITYPE num_events_to_print = perf::NUM_EVENTS_PER_SET;                       \
                                            if (e == perf_event_set::CACHE_STALL_EVENTS) {                              \
                                                num_events_to_print = NUM_CACHE_STALL_EVENTS;                           \
                                            } else if (e == perf_event_set::NAMED_CACHE_EVENTS) {                       \
                                                num_events_to_print = NUM_NAMED_CACHE_EVENTS;                           \
                                            } else if (e == perf_event_set::L3_CACHE_EVENTS) {                          \
                                                num_events_to_print = NUM_L3_CACHE_EVENTS;                              \
                                            } else if (e == perf_event_set::NAMED_STALL_EVENTS) {                       \
                                                num_events_to_print = NUM_NAMED_STALL_EVENTS;                           \
                                            } else if (e == perf_event_set::FILL_EVENTS) {                              \
                                                num_events_to_print = NUM_FILL_EVENTS;                                  \
                                            }                                                                           \
                                            PERF_PRINT(e, s, num_threads, num_events_to_print);                         \
                                            cycle_counts.process();                                                     \
                                            cycle_counts.print();                                                       \
                                            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                            std::cout << "Median Time: " << median_time << std::endl;                   \
                                            size_t flop_count = ((size_t)2) * ((size_t) mkl_nnzs) * ((size_t) mkl_feature);    \
                                            double gflops = ((double)flop_count) / median_time / 1E9;                   \
                                            std::cout << "GFLOPS: " << gflops << std::endl;                             \
                                        }
#endif // LARGE_ARRAY


#else // TRACK_PARALLEL_STATS
#define RUN_PERF_HELPER(f,p,s,e,A,Ti,Tk)
#define RUN_HYBRID_PERF_HELPER(f,p,s,e,A,num_panel_per_thread,worklist)
#define RUN_MKL_PERF_HELPER(p,s,e)
#endif // TRACK_PARALLEL_STATS





// f - spmm function to call
// p - performance counter array
// s - stats array
// e - performance counter type
// A - The sparse matrix pointer
// Ti - Row panel height
// Tk - Tk
#ifdef TRACK_PARALLEL_STATS

#ifdef LARGE_ARRAY

#define RUN_DRAM_HELPER(f,p,s,e,A,Ti,Tk){                                                                               \
                                            stats_t<long long, ITYPE> cycle_counts;                                     \
                                            cycle_counts.name = "Cycle Counts";                                         \
                                                                                                                        \
                                            for ( ITYPE i = 0; i < n; i++ ) {                                           \
                                                                                                                        \
                                                T *input = &B[ (i % layers) * (ncols * feature) ];                      \
                                                T *output = &C[ (i % layers) * (nrows * feature) ];                     \
                                                                                                                        \
                                                CACHE_FLUSH;                                                            \
                                                                                                                        \
                                                DRAM_START(p);                                                          \
                                                long long duration = f(*A, input, output, feature, Ti, Tk, chunk_size); \
                                                DRAM_STOP(p, s);                                                        \
                                                                                                                        \
                                                cycle_counts.insert(duration);                                          \
                                                                                                                        \
                                            }                                                                           \
                                            cycle_counts.process();                                                     \
                                            cycle_counts.print();                                                       \
                                            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                            std::cout << "Median Time: " << median_time << std::endl;                   \
                                            PERF_PRINT(e, s, num_threads, e == DRAM_EVENTS ? NUM_MEMORY_CHANNELS * 2 : NUM_MEMORY_CHANNELS * 2);                     \
                                        }

#else

#define RUN_DRAM_HELPER(f,p,s,e,A,Ti,Tk){                                                                               \
                                            stats_t<long long, ITYPE> cycle_counts;                                     \
                                            cycle_counts.name = "Cycle Counts";                                         \
                                                                                                                        \
                                            for ( ITYPE i = 0; i < n; i++ ) {                                           \
                                                                                                                        \
                                                T *input = &(B[ (i % layers) ][0]);                                     \
                                                T *output = &(C[ (i % layers) ][0]);                                    \
                                                                                                                        \
                                                CACHE_FLUSH;                                                            \
                                                                                                                        \
                                                DRAM_START(p);                                                          \
                                                long long duration = f(*A, input, output, feature, Ti, Tk, chunk_size); \
                                                DRAM_STOP(p, s);                                                        \
                                                                                                                        \
                                                cycle_counts.insert(duration);                                          \
                                                                                                                        \
                                            }                                                                           \
                                            cycle_counts.process();                                                     \
                                            cycle_counts.print();                                                       \
                                            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                            std::cout << "Median Time: " << median_time << std::endl;                   \
                                            PERF_PRINT(e, s, num_threads, e == DRAM_EVENTS ? NUM_MEMORY_CHANNELS * 2 : NUM_MEMORY_CHANNELS * 2);                     \
                                        }

#endif // LARGE_ARRAY


#ifdef LARGE_ARRAY

#define RUN_HYBRID_DRAM_HELPER(f,p,s,e,A,wl,ppt){                                                                       \
                                            stats_t<long long, ITYPE> cycle_counts;                                     \
                                            cycle_counts.name = "Cycle Counts";                                         \
                                                                                                                        \
                                            for ( ITYPE i = 0; i < n; i++ ) {                                           \
                                                                                                                        \
                                                T *input = &B[ (i % layers) * (ncols * feature) ];                      \
                                                T *output = &C[ (i % layers) * (nrows * feature) ];                     \
                                                                                                                        \
                                                CACHE_FLUSH;                                                            \
                                                                                                                        \
                                                DRAM_START(p);                                                          \
                                                long long duration = f(*A, input, output, feature, wl, ppt, num_threads, chunk_size); \
                                                DRAM_STOP(p, s);                                                        \
                                                                                                                        \
                                                cycle_counts.insert(duration);                                          \
                                                                                                                        \
                                            }                                                                           \
                                            cycle_counts.process();                                                     \
                                            cycle_counts.print();                                                       \
                                            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                            std::cout << "Median Time: " << median_time << std::endl;                   \
                                            PERF_PRINT(e, s, num_threads, e == DRAM_EVENTS ? NUM_MEMORY_CHANNELS * 2 : NUM_MEMORY_CHANNELS * 2);                     \
                                        }

#else

#define RUN_HYBRID_DRAM_HELPER(f,p,s,e,A,wl,ppt){                                                                       \
                                            stats_t<long long, ITYPE> cycle_counts;                                     \
                                            cycle_counts.name = "Cycle Counts";                                         \
                                                                                                                        \
                                            for ( ITYPE i = 0; i < n; i++ ) {                                           \
                                                                                                                        \
                                                T *input = &(B[ (i % layers) ][0]);                                     \
                                                T *output = &(C[ (i % layers) ][0]);                                    \
                                                                                                                        \
                                                CACHE_FLUSH;                                                            \
                                                                                                                        \
                                                DRAM_START(p);                                                          \
                                                long long duration = f(*A, input, output, feature, wl, ppt, num_threads, chunk_size); \
                                                DRAM_STOP(p, s);                                                        \
                                                                                                                        \
                                                cycle_counts.insert(duration);                                          \
                                                                                                                        \
                                            }                                                                           \
                                            cycle_counts.process();                                                     \
                                            cycle_counts.print();                                                       \
                                            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                            std::cout << "Median Time: " << median_time << std::endl;                   \
                                            PERF_PRINT(e, s, num_threads, e == DRAM_EVENTS ? NUM_MEMORY_CHANNELS * 2 : NUM_MEMORY_CHANNELS * 2);                     \
                                        }

#endif // LARGE_ARRAY

#else // TRACK_PARALLEL_STATS
#define RUN_DRAM_HELPER(f,p,s,e,A,Ti,Tk)
#define RUN_HYBRID_DRAM_HELPER(f,p,s,e,A,wl,ppt)
#endif // TRACK_PARALLEL_STATS

#ifdef LARGE_ARRAY

// f - spmm function to call
// A - The sparse matrix pointer
// Ti - Row panel height
// Tk - Tk
#define RUN_TIMING(f, A, Ti, Tk)        {                                                                               \
                                            stats_t<long long, ITYPE> cycle_counts;                                     \
                                            cycle_counts.name = "Cycle Counts";                                         \
                                                                                                                        \
                                            for ( ITYPE i = 0; i < n; i++ ) {                                           \
                                                                                                                        \
                                                T *input = &B[ (i % layers) * countB ];                                 \
                                                T *output = &C[ (i % layers) * countB ];                                \
                                                                                                                        \
                                                CACHE_FLUSH;                                                            \
                                                                                                                        \
                                                long long duration = f(*A, input, output, feature, Ti, Tk, chunk_size); \
                                                                                                                        \
                                                cycle_counts.insert(duration);                                          \
                                                                                                                        \
                                            }                                                                           \
                                            cycle_counts.process();                                                     \
                                            cycle_counts.print();                                                       \
                                            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                            std::cout << "Median Time: " << median_time << std::endl;                   \
                                            double flop_count = 2 * A->nnzs * feature;                                  \
                                            double gflops = flop_count / median_time / 1E9;                             \
                                            std::cout << "GFLOPS: " << gflops << std::endl;                             \
                                        }

#else

#define RUN_TIMING(f, A, Ti, Tk)        {                                                                               \
                                            stats_t<long long, ITYPE> cycle_counts;                                     \
                                            cycle_counts.name = "Cycle Counts";                                         \
                                                                                                                        \
                                            for ( ITYPE i = 0; i < n; i++ ) {                                           \
                                                                                                                        \
                                                T *input = &(B[ (i % layers) ][0]);                                     \
                                                T *output = &(C[ (i % layers) ][0]);                                    \
                                                                                                                        \
                                                CACHE_FLUSH;                                                            \
                                                                                                                        \
                                                long long duration = f(*A, input, output, feature, Ti, Tk, chunk_size); \
                                                                                                                        \
                                                cycle_counts.insert(duration);                                          \
                                                                                                                        \
                                            }                                                                           \
                                            cycle_counts.process();                                                     \
                                            cycle_counts.print();                                                       \
                                            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                            std::cout << "Median Time: " << median_time << std::endl;                   \
                                            double flop_count = 2 * ((size_t) A->nnzs) * ((size_t) feature);            \
                                            double gflops = flop_count / median_time / 1E9;                             \
                                            std::cout << "GFLOPS: " << gflops << std::endl;                             \
                                        }




#endif // LARGE_ARRAY


#ifdef LARGE_ARRAY
#define RUN_HYBRID_TIMING(f,A,wl,ppt){                                                                                  \
                                            stats_t<long long, ITYPE> cycle_counts;                                     \
                                            cycle_counts.name = "Cycle Counts";                                         \
                                                                                                                        \
                                            for ( ITYPE i = 0; i < n; i++ ) {                                           \
                                                                                                                        \
                                                T *input = &B[ (i % layers) * (ncols * feature) ];                      \
                                                T *output = &C[ (i % layers) * (nrows * feature) ];                     \
                                                                                                                        \
                                                CACHE_FLUSH;                                                            \
                                                                                                                        \
                                                long long duration = f(*A, input, output, feature, wl, ppt, num_threads, chunk_size); \
                                                                                                                        \
                                                cycle_counts.insert(duration);                                          \
                                                                                                                        \
                                            }                                                                           \
                                            cycle_counts.process();                                                     \
                                            cycle_counts.print();                                                       \
                                            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                            std::cout << "Median Time: " << median_time << std::endl;                   \
                                            size_t flop_count = ((size_t) 2) * ((size_t) A->nnzs) * ((size_t) feature); \
                                            double gflops = ((double) flop_count) / median_time / 1E9;                  \
                                            std::cout << "GFLOPS: " << gflops << std::endl;                             \
                                        }

#else

#define RUN_HYBRID_TIMING(f,A,wl,ppt){                                                                                  \
                                            stats_t<long long, ITYPE> cycle_counts;                                     \
                                            cycle_counts.name = "Cycle Counts";                                         \
                                                                                                                        \
                                            for ( ITYPE i = 0; i < n; i++ ) {                                           \
                                                                                                                        \
                                                T *input = &(B[ (i % layers) ][0]);                                     \
                                                T *output = &(C[ (i % layers) ][0]);                                    \
                                                                                                                        \
                                                CACHE_FLUSH;                                                            \
                                                                                                                        \
                                                long long duration = f(*A, input, output, feature, wl, ppt, num_threads, chunk_size); \
                                                                                                                        \
                                                cycle_counts.insert(duration);                                          \
                                                                                                                        \
                                            }                                                                           \
                                            cycle_counts.process();                                                     \
                                            cycle_counts.print();                                                       \
                                            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                            std::cout << "Median Time: " << median_time << std::endl;                   \
                                            size_t flop_count = ((size_t) 2) * ((size_t) A->nnzs) * ((size_t) feature); \
                                            double gflops = ((double) flop_count) / median_time / 1E9;                  \
                                            std::cout << "GFLOPS: " << gflops << std::endl;                             \
                                        }

#define RUN_HYBRID_PER_CORE_TIMING(f,A,wl,ppt){                                                                         \
                                            stats_t<long long, ITYPE> cycle_counts;                                     \
                                            cycle_counts.name = "Cycle Counts";                                         \
                                            long long core_cycle_counts[num_threads];                                   \
                                            stats_t<long long, ITYPE> per_core_cycle_counts[num_threads];               \
                                            stats_t<double, ITYPE> per_core_execution_times[num_threads];               \
                                            for (ITYPE i = 0; i < num_threads; i++) {                                   \
                                                per_core_cycle_counts[i].name = "core:" + std::to_string(i);            \
                                                per_core_execution_times[i].name = "core:" + std::to_string(i);         \
                                            }                                                                           \
                                                                                                                        \
                                            for ( ITYPE i = 0; i < n; i++ ) {                                           \
                                                                                                                        \
                                                T *input = &(B[ (i % layers) ][0]);                                     \
                                                T *output = &(C[ (i % layers) ][0]);                                    \
                                                                                                                        \
                                                CACHE_FLUSH;                                                            \
                                                                                                                        \
                                                long long duration = f(*A, input, output, feature, wl, ppt, num_threads, chunk_size, core_cycle_counts); \
                                                                                                                        \
                                                for (ITYPE i = 0; i < num_threads; i++) {                               \
                                                    per_core_cycle_counts[i].insert(core_cycle_counts[i]);              \
                                                }                                                                       \
                                                cycle_counts.insert(duration);                                          \
                                                                                                                        \
                                                                                                                        \
                                            }                                                                           \
                                            cycle_counts.process();                                                     \
                                            cycle_counts.print();                                                       \
                                            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                            std::cout << "Median Time: " << median_time << std::endl;                   \
                                            double flop_count = 2 * A->nnzs * feature;                                  \
                                            double gflops = flop_count / median_time / 1E9;                             \
                                            std::cout << "GFLOPS: " << gflops << std::endl;                             \
                                            double min_core_time = std::numeric_limits<double>::max();                  \
                                            double max_core_time = std::numeric_limits<double>::min();                  \
                                            for ( ITYPE tid = 0; tid < num_threads; tid++ ) {                           \
                                                per_core_cycle_counts[tid].process();                                   \
                                                for ( ITYPE rid = 0; rid < n; rid++) {                                  \
                                                    double core_execution_time = ((double) per_core_cycle_counts[tid].records[rid]) / CLOCK_FREQUENCY; \
                                                    per_core_execution_times[tid].insert( core_execution_time );        \
                                                }                                                                       \
                                                per_core_execution_times[tid].process();                                \
                                                if (max_core_time < per_core_execution_times[tid].median) {             \
                                                    max_core_time = per_core_execution_times[tid].median;               \
                                                }                                                                       \
                                                if (min_core_time > per_core_execution_times[tid].median) {             \
                                                    min_core_time = per_core_execution_times[tid].median;               \
                                                }                                                                       \
                                            }                                                                           \
                                            std::cout << "Min Core Time: " << min_core_time << std::endl;               \
                                            std::cout << "Max Core Time: " << max_core_time << std::endl;               \
                                            for ( ITYPE tid = 0; tid < num_threads; tid++ ) {                           \
                                                double core_idle_perc = (median_time - per_core_execution_times[tid].median) / median_time * 100; \
                                                std::cout << "core: " << tid << ", " << per_core_execution_times[tid].median << ", " << core_idle_perc << std::endl; \
                                            }                                                                           \
                                        }

#define RUN_HYBRID_PER_PANEL_TIMING(f,A,wl,ppt){                                                                        \
                                            stats_t<long long, ITYPE> cycle_counts;                                     \
                                            cycle_counts.name = "Cycle Counts";                                         \
                                            long long core_cycle_counts[num_threads];                                   \
                                            long long *per_panel_cycle_counts = new long long[A->num_panels * n];       \
                                            stats_t<long long, ITYPE> per_core_cycle_counts[num_threads];               \
                                            stats_t<double, ITYPE> per_core_execution_times[num_threads];               \
                                            for (ITYPE i = 0; i < num_threads; i++) {                                   \
                                                per_core_cycle_counts[i].name = "core:" + std::to_string(i);            \
                                                per_core_execution_times[i].name = "core:" + std::to_string(i);         \
                                            }                                                                           \
                                                                                                                        \
                                            for ( ITYPE i = 0; i < n; i++ ) {                                           \
                                                                                                                        \
                                                T *input = &(B[ (i % layers) ][0]);                                     \
                                                T *output = &(C[ (i % layers) ][0]);                                    \
                                                long long *iter_panel_cycle_counts = &per_panel_cycle_counts[A->num_panels * i]; \
                                                CACHE_FLUSH;                                                            \
                                                                                                                        \
                                                long long duration = f(*A, input, output, feature, wl, ppt, num_threads, chunk_size, core_cycle_counts, iter_panel_cycle_counts); \
                                                                                                                        \
                                                for (ITYPE i = 0; i < num_threads; i++) {                               \
                                                    per_core_cycle_counts[i].insert(core_cycle_counts[i]);              \
                                                }                                                                       \
                                                cycle_counts.insert(duration);                                          \
                                                                                                                        \
                                                                                                                        \
                                            }                                                                           \
                                            cycle_counts.process();                                                     \
                                            cycle_counts.print();                                                       \
                                            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                            std::cout << "Median Time: " << median_time << std::endl;                   \
                                            double flop_count = ((double) (2 * A->nnzs)) * ((double) feature);          \
                                            double gflops = flop_count / median_time / 1E9;                             \
                                            std::cout << "GFLOPS: " << gflops << std::endl;                             \
                                            double min_core_time = std::numeric_limits<double>::max();                  \
                                            double max_core_time = std::numeric_limits<double>::min();                  \
                                            for ( ITYPE tid = 0; tid < num_threads; tid++ ) {                           \
                                                per_core_cycle_counts[tid].process();                                   \
                                                for ( ITYPE rid = 0; rid < n; rid++) {                                  \
                                                    double core_execution_time = ((double) per_core_cycle_counts[tid].records[rid]) / CLOCK_FREQUENCY; \
                                                    per_core_execution_times[tid].insert( core_execution_time );        \
                                                }                                                                       \
                                                per_core_execution_times[tid].process();                                \
                                                if (max_core_time < per_core_execution_times[tid].median) {             \
                                                    max_core_time = per_core_execution_times[tid].median;               \
                                                }                                                                       \
                                                if (min_core_time > per_core_execution_times[tid].median) {             \
                                                    min_core_time = per_core_execution_times[tid].median;               \
                                                }                                                                       \
                                            }                                                                           \
                                            std::cout << "Min Core Time: " << min_core_time << std::endl;               \
                                            std::cout << "Max Core Time: " << max_core_time << std::endl;               \
                                            for ( ITYPE tid = 0; tid < num_threads; tid++ ) {                           \
                                                double core_idle_perc = (median_time - per_core_execution_times[tid].median) / median_time * 100; \
                                                std::cout << "core: " << tid << ", " << per_core_execution_times[tid].median << ", " << core_idle_perc << std::endl; \
                                            }                                                                           \
                                            stats_t<double, ITYPE> *per_panel_execution_times = new stats_t<double, ITYPE>[A->num_panels]; \
                                            std::cout << "###START PER PANEL TIME PARSING###" << std::endl;             \
                                            std::cout << "panel, type, tk, OI, NNZS, NACS, NARS, RNACS, RNARS, median time, time median" << std::endl; \
                                            for ( ITYPE panel = 0; panel < A->num_panels; panel++ ) {                   \
                                                                                                                        \
                                                per_panel_execution_times[panel].name = "panel " + std::to_string(panel);   \
                                                for ( ITYPE i = 0; i < n; i++ ) {                                       \
                                                    double panel_execution_time = ((double) per_panel_cycle_counts[ (i * A->num_panels) + panel ]) / CLOCK_FREQUENCY;   \
                                                    per_panel_execution_times[panel].insert( panel_execution_time );           \
                                                }                                                                       \
                                                per_panel_execution_times[panel].process();                             \
                                                long long panel_cycle_count = (per_panel_cycle_counts[cycle_counts.median_index_A * A->num_panels + panel] + per_panel_cycle_counts[cycle_counts.median_index_B * A->num_panels + panel]) / 2;    \
                                                double median_runtime_panel_time = ((double) panel_cycle_count ) / CLOCK_FREQUENCY; \
                                                ITYPE panel_dense_bytes_moved = (work_list[panel].nacs + work_list[panel].nars) * feature * sizeof(T);  \
                                                ITYPE panel_sparse_bytes_moved = work_list[panel].nnzs * (sizeof(ITYPE) + sizeof(T)) * CEIL(feature, work_list[panel].Tk); \
                                                ITYPE panel_flop_count = work_list[panel].nnzs * 2 * feature; \
                                                double panel_OI = ((double) panel_flop_count) / ((double) (panel_dense_bytes_moved + panel_sparse_bytes_moved)); \
                                                ITYPE panel_Tk = A->Tk[panel] & TK_MASK;                                \
                                                std::cout << panel << ", " << A->panel_type[panel] << ", " << panel_Tk << ", " << panel_OI << ", " << work_list[panel].nnzs << ", " << work_list[panel].nacs << ", " << work_list[panel].nars << ", " << work_list[panel].rnacs << ", " << work_list[panel].rnars << ", " << per_panel_execution_times[panel].median << ", " << median_runtime_panel_time << std::endl; \
                                            }                                                                           \
                                            std::cout << "###STOP PER PANEL TIME PARSING###" << std::endl;              \
                                            delete[] per_panel_execution_times;                                         \
                                            delete[] per_panel_cycle_counts;                                            \
                                        }

#endif  // LARGE_ARRAY

#ifdef LARGE_ARRAY
#define RUN_MKL_TIMING              {   std::cout << "No implementation of MKL large array" << std::cout;   }
#else // LARGE_ARRAY
#define RUN_MKL_TIMING          {                                                                                   \
                                        stats_t<long long, ITYPE> cycle_counts;                                     \
                                        cycle_counts.name = "Cycle Counts";                                         \
                                                                                                                    \
                                        for ( ITYPE i = 0; i < n; i++ ) {                                           \
                                                                                                                    \
                                            T *input = &(B[ (i % layers) ][0]);                                     \
                                            T *output = &(C[ (i % layers) ][0]);                                    \
                                                                                                                    \
                                            CACHE_FLUSH;                                                            \
                                                                                                                    \
                                            auto start_cycle = readTSC();                                           \
                                            mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, mkl_alpha, mkl_S, mkl_S_desc, SPARSE_LAYOUT_ROW_MAJOR, input, mkl_feature, mkl_feature, mkl_beta, output, mkl_feature ); \
                                            auto end_cycle = readTSC();                                             \
                                                                                                                    \
                                            cycle_counts.insert( (end_cycle - start_cycle) );                       \
                                                                                                                    \
                                        }                                                                           \
                                        cycle_counts.process();                                                     \
                                        cycle_counts.print();                                                       \
                                        double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                        std::cout << "Median Time: " << median_time << std::endl;                   \
                                        double flop_count = ((size_t) 2) * (((size_t) mkl_nnzs) * ((size_t)mkl_feature));                             \
                                        double gflops = flop_count / median_time / 1E9;                             \
                                        std::cout << "GFLOPS: " << gflops << std::endl;                             \
                                }

#endif // LARGE_ARRAY


#define RUN_PER_CORE_TIMING(f, A, Ti, Tk)   {                                                                           \
                                            std::cout << "Running per core timing with " << num_threads << " cores" << std::endl;    \
                                            long long core_cycle_counts[num_threads];                                   \
                                            long long duration;                                                         \
                                            stats_t<long long, ITYPE> per_core_cycle_counts[num_threads];               \
                                            stats_t<double, ITYPE> per_core_execution_times[num_threads];               \
                                            stats_t<long long, ITYPE> cycle_counts;                                     \
                                            for (ITYPE i = 0; i < num_threads; i++) {                                   \
                                                per_core_cycle_counts[i].name = "core:" + std::to_string(i);            \
                                                per_core_execution_times[i].name = "core:" + std::to_string(i);         \
                                            }                                                                           \
                                            cycle_counts.name = "Cycle Counts";                                         \
                                                                                                                        \
                                            for ( ITYPE i = 0; i < n; i++ ) {                                           \
                                                                                                                        \
                                                T *input = &B[ (i % layers) * (ncols * feature) ];                      \
                                                T *output = &C[ (i % layers) * (nrows * feature) ];                     \
                                                                                                                        \
                                                CACHE_FLUSH;                                                            \
                                                                                                                        \
                                                duration = f(*A, input, output, feature, Ti, Tk, chunk_size, core_cycle_counts);    \
                                                                                                                        \
                                                for (ITYPE i = 0; i < num_threads; i++) {                               \
                                                    per_core_cycle_counts[i].insert(core_cycle_counts[i]);              \
                                                }                                                                       \
                                                cycle_counts.insert(duration);                                          \
                                                                                                                        \
                                            }                                                                           \
                                            cycle_counts.process();                                                     \
                                            cycle_counts.print();                                                       \
                                            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                            std::cout << "Median Time: " << median_time << std::endl;                   \
                                            double min_core_time = std::numeric_limits<double>::max();                  \
                                            double max_core_time = std::numeric_limits<double>::min();                  \
                                            for ( ITYPE tid = 0; tid < num_threads; tid++ ) {                           \
                                                per_core_cycle_counts[tid].process();                                   \
                                                for ( ITYPE rid = 0; rid < n; rid++) {                                  \
                                                    double core_execution_time = ((double) per_core_cycle_counts[tid].records[rid]) / CLOCK_FREQUENCY; \
                                                    per_core_execution_times[tid].insert( core_execution_time );        \
                                                }                                                                       \
                                                per_core_execution_times[tid].process();                                \
                                                if (max_core_time < per_core_execution_times[tid].median) {             \
                                                    max_core_time = per_core_execution_times[tid].median;               \
                                                }                                                                       \
                                                if (min_core_time > per_core_execution_times[tid].median) {             \
                                                    min_core_time = per_core_execution_times[tid].median;               \
                                                }                                                                       \
                                            }                                                                           \
                                            std::cout << "Min Core Time: " << min_core_time << std::endl;               \
                                            std::cout << "Max Core Time: " << max_core_time << std::endl;               \
                                            for ( ITYPE tid = 0; tid < num_threads; tid++ ) {                           \
                                                double core_idle_perc = (median_time - per_core_execution_times[tid].median) / median_time * 100; \
                                                std::cout << "core: " << tid << ", " << per_core_execution_times[tid].median << ", " << core_idle_perc << std::endl; \
                                            }                                                                           \
                                        }


template<typename T, typename ITYPE>
T *calculate_panel_median_times( ITYPE num_threads, ITYPE num_iters, ITYPE *num_panels_per_thread, stats_t<T, ITYPE> *raw_stats )
{
    ITYPE max_panels_per_thread = num_panels_per_thread[0];
    for (ITYPE i = 0; i < num_threads; i++) {
        if (num_panels_per_thread[i] > max_panels_per_thread) {
            max_panels_per_thread = num_panels_per_thread[i];
        }
    }

    T *panel_median_times = new T[ num_threads * max_panels_per_thread ];
    std::memset( panel_median_times, 0, (sizeof(T) * (num_threads * max_panels_per_thread)) );

    stats_t<T, ITYPE> temp;
    temp.resize(num_iters);

    for ( ITYPE tid = 0; tid < num_threads; tid++ ) {
        ITYPE tid_num_panels = num_panels_per_thread[tid];

        std::cout << "core: " << tid << ", ";
        for ( ITYPE pid = 0; pid < tid_num_panels; pid++ ) {
            temp.reset();
            for ( ITYPE rid = 0; rid < num_iters; rid++ ) {
                ITYPE record_offset = pid + (rid * tid_num_panels);
                temp.insert( raw_stats[tid].records[record_offset] );
            }
            temp.process();
            panel_median_times[ (pid * num_threads) + tid ] = temp.median;
            std::cout << temp.median << ",";
        }
        std::cout << std::endl;
    }

    return panel_median_times;
}

#ifdef LARGE_ARRAY

#define RUN_PER_PANEL_TIMING(f, A, Ti, Tk)   {                                                                           \
                                            std::cout << "Running per panel timing with " << num_threads << " cores" << std::endl;    \
                                            ITYPE num_panels_per_core = CEIL( CEIL(A->nrows, Ti), num_threads );        \
                                            long long core_cycle_counts[num_threads];                                    \
                                            long long duration;                                                         \
                                            stats_t<long long, ITYPE> per_core_cycle_counts_stats[num_threads];         \
                                            stats_t<long long, ITYPE> cycle_counts;                                     \
                                            stats_t<long long, ITYPE> per_panel_cycle_counts_stats[num_threads];        \
                                            ITYPE per_core_panel_count[num_threads];                                    \
                                            long long *per_panel_cycle_counts = new long long[num_panels_per_core * num_threads];   \
                                            std::memset(per_panel_cycle_counts, 0, sizeof(long long) * num_panels_per_core * num_threads); \
                                            for ( ITYPE tid = 0; tid < num_threads; tid++ ) {                           \
                                                per_core_cycle_counts_stats[tid].name = "Cycle Count Core: " + std::to_string(tid);   \
                                                per_panel_cycle_counts_stats[tid].resize( n * num_panels_per_core );    \
                                                per_core_panel_count[tid] = 0;                                          \
                                            }                                                                           \
                                                                                                                        \
                                            for ( ITYPE panel = 0; panel < CEIL(A->nrows, Ti); panel++ ) {              \
                                                per_core_panel_count[ panel % num_threads ]++;                          \
                                            }                                                                           \
                                                                                                                        \
                                            for ( ITYPE i = 0; i < n; i++ ) {                                           \
                                                                                                                        \
                                                T *input = &B[ (i % layers) * (ncols * feature) ];                      \
                                                T *output = &C[ (i % layers) * (nrows * feature) ];                     \
                                                                                                                        \
                                                CACHE_FLUSH;                                                            \
                                                                                                                        \
                                                duration = f(*A, input, output, feature, Ti, Tk, chunk_size, core_cycle_counts, per_panel_cycle_counts);    \
                                                                                                                        \
                                                cycle_counts.insert(duration);                                          \
                                                                                                                        \
                                                for (ITYPE tid = 0; tid < num_threads; tid++) {                         \
                                                    per_core_cycle_counts_stats[tid].insert( core_cycle_counts[tid] );  \
                                                    for (ITYPE rid = 0; rid < per_core_panel_count[tid]; rid++) {       \
                                                        per_panel_cycle_counts_stats[tid].insert(per_panel_cycle_counts[tid + (rid * num_threads)]);  \
                                                    }                                                                   \
                                                }                                                                       \
                                                                                                                        \
                                            }                                                                           \
                                            cycle_counts.process();                                                     \
                                            cycle_counts.print();                                                       \
                                            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                            std::cout << "Median Time: " << median_time << std::endl;                   \
                                            for ( ITYPE tid = 0; tid < num_threads; tid++ ) {                           \
                                                per_core_cycle_counts_stats[tid].process();                             \
                                                double core_execution_time = ((double) per_core_cycle_counts_stats[tid].median) / CLOCK_FREQUENCY;  \
                                                double core_idle_perc = (median_time - core_execution_time) / median_time * 100;                    \
                                                std::cout << "core: " << tid << ", " << core_execution_time << ", " << core_idle_perc << std::endl; \
                                            }                                                                                                       \
                                            auto processed_execution_times = calculate_panel_median_times(num_threads, n, per_core_panel_count, per_panel_cycle_counts_stats); \
                                            for (ITYPE tid = 0; tid < num_threads; tid++) {                             \
                                                std::cout << "panel: " << tid << ", ";                                   \
                                                for (ITYPE pid = 0; pid < per_core_panel_count[tid]; pid++) {         \
                                                    std::cout << processed_execution_times[ (pid * num_threads) + tid ] << ", ";    \
                                                }                                                                       \
                                                std::cout << std::endl;                                                 \
                                            }                                                                           \
                                            delete[] per_panel_cycle_counts;                                            \
                                        }

#else // LARGE_ARRAY

#define RUN_PER_PANEL_TIMING(f, A, Ti, Tk)   {                                                                          \
                                            stats_t<long long, ITYPE> cycle_counts;                                     \
                                            cycle_counts.name = "Cycle Counts";                                         \
                                            long long core_cycle_counts[num_threads];                                   \
                                            long long *per_panel_cycle_counts = new long long[A->num_panels * n];       \
                                            stats_t<long long, ITYPE> per_core_cycle_counts[num_threads];               \
                                            stats_t<double, ITYPE> per_core_execution_times[num_threads];               \
                                            for (ITYPE i = 0; i < num_threads; i++) {                                   \
                                                per_core_cycle_counts[i].name = "core:" + std::to_string(i);            \
                                                per_core_execution_times[i].name = "core:" + std::to_string(i);         \
                                            }                                                                           \
                                                                                                                        \
                                            for ( ITYPE i = 0; i < n; i++ ) {                                           \
                                                T *input = &(B[ (i % layers) ][0]);                                     \
                                                T *output = &(C[ (i % layers) ][0]);                                    \
                                                long long *iter_panel_cycle_counts = &per_panel_cycle_counts[A->num_panels * i]; \
                                                                                                                        \
                                                CACHE_FLUSH;                                                            \
                                                                                                                        \
                                                long long duration = f(*A, input, output, feature, Ti, Tk, chunk_size, core_cycle_counts, iter_panel_cycle_counts);    \
                                                                                                                        \
                                                cycle_counts.insert(duration);                                          \
                                                                                                                        \
                                                for (ITYPE tid = 0; tid < num_threads; tid++) {                         \
                                                    per_core_cycle_counts[tid].insert( core_cycle_counts[tid] );        \
                                                }                                                                       \
                                                                                                                        \
                                            }                                                                           \
                                            cycle_counts.process();                                                     \
                                            cycle_counts.print();                                                       \
                                            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                            std::cout << "Median Time: " << median_time << std::endl;                   \
                                            double flop_count = 2 * A->nnzs * feature;                                  \
                                            double gflops = flop_count / median_time / 1E9;                             \
                                            std::cout << "GFLOPS: " << gflops << std::endl;                             \
                                            double min_core_time = std::numeric_limits<double>::max();                  \
                                            double max_core_time = std::numeric_limits<double>::min();                  \
                                            for ( ITYPE tid = 0; tid < num_threads; tid++ ) {                           \
                                                per_core_cycle_counts[tid].process();                                   \
                                                for ( ITYPE rid = 0; rid < n; rid++) {                                  \
                                                    double core_execution_time = ((double) per_core_cycle_counts[tid].records[rid]) / CLOCK_FREQUENCY; \
                                                    per_core_execution_times[tid].insert( core_execution_time );        \
                                                }                                                                       \
                                                per_core_execution_times[tid].process();                                \
                                                if (max_core_time < per_core_execution_times[tid].median) {             \
                                                    max_core_time = per_core_execution_times[tid].median;               \
                                                }                                                                       \
                                                if (min_core_time > per_core_execution_times[tid].median) {             \
                                                    min_core_time = per_core_execution_times[tid].median;               \
                                                }                                                                       \
                                            }                                                                           \
                                            std::cout << "Min Core Time: " << min_core_time << std::endl;               \
                                            std::cout << "Max Core Time: " << max_core_time << std::endl;               \
                                            for ( ITYPE tid = 0; tid < num_threads; tid++ ) {                           \
                                                double core_idle_perc = (median_time - per_core_execution_times[tid].median) / median_time * 100; \
                                                std::cout << "core: " << tid << ", " << per_core_execution_times[tid].median << ", " << core_idle_perc << std::endl; \
                                            }                                                                           \
                                            stats_t<double, ITYPE> *per_panel_execution_times = new stats_t<double, ITYPE>[A->num_panels]; \
                                            std::cout << "###START PER PANEL TIME PARSING###" << std::endl;             \
                                            std::cout << "panel, type, tk, OI, NNZS, NACS, NARS, median time, time median" << std::endl; \
                                            for ( ITYPE panel = 0; panel < A->num_panels; panel++ ) {                   \
                                                                                                                        \
                                                per_panel_execution_times[panel].name = "panel " + std::to_string(panel);   \
                                                for ( ITYPE i = 0; i < n; i++ ) {                                       \
                                                    double panel_execution_time = ((double) per_panel_cycle_counts[ (i * A->num_panels) + panel ]) / CLOCK_FREQUENCY;   \
                                                    per_panel_execution_times[panel].insert( panel_execution_time );           \
                                                }                                                                       \
                                                per_panel_execution_times[panel].process();                             \
                                                long long panel_cycle_count = (per_panel_cycle_counts[cycle_counts.median_index_A * A->num_panels + panel] + per_panel_cycle_counts[cycle_counts.median_index_B * A->num_panels + panel]) / 2;    \
                                                double median_runtime_panel_time = ((double) panel_cycle_count ) / CLOCK_FREQUENCY; \
                                                std::cout << panel << "," << per_panel_execution_times[panel].median << "," << median_runtime_panel_time << std::endl; \
                                            }                                                                           \
                                            std::cout << "###STOP PER PANEL TIME PARSING###" << std::endl;              \
                                            delete[] per_panel_execution_times;                                         \
                                            delete[] per_panel_cycle_counts;                                            \
                                        }

/*
                                                ITYPE panel_dense_bytes_moved = (work_list[panel].nacs + work_list[panel].nars) * feature * sizeof(T);  \
                                                ITYPE panel_sparse_bytes_moved = work_list[panel].nnzs * (sizeof(ITYPE) + sizeof(T)) * CEIL(feature, work_list[panel].Tk); \
                                                ITYPE panel_flop_count = work_list[panel].nnzs * 2 * feature; \
                                                double panel_OI = ((double) panel_flop_count) / ((double) (panel_dense_bytes_moved + panel_sparse_bytes_moved)); \
                                                std::cout << panel << ", " << A->panel_type[panel] << ", " << A->Tk[panel] << ", " << panel_OI << ", " << work_list[panel].nnzs << ", " << work_list[panel].nacs << ", " << work_list[panel].nars << ", " << per_panel_execution_times[panel].median << ", " << median_runtime_panel_time << std::endl; \
 */

#endif // LARGE_ARRAY


// f - spmm function to call
// A - The sparse matrix pointer
// Ti - Row panel height
// Tk - Tk
#ifdef RUN_WARMUP

#ifdef LARGE_ARRAY

#define RUN_WARMUP_EXPR(f, A, Ti, Tk)   {                                                                               \
                                            stats_t<long long, ITYPE> cycle_counts;                                     \
                                            cycle_counts.name = "Cycle Counts";                                         \
                                                                                                                        \
                                            for ( ITYPE i = 0; i < (n / WARMUP_DIVIDER); i++ ) {                        \
                                                                                                                        \
                                                T *input = &B[ (i % layers) * (ncols * (feature + PADDING_B)) ];        \
                                                T *output = &C[ (i % layers) * (nrows * (feature + PADDING_C)) ];       \
                                                                                                                        \
                                                CACHE_FLUSH;                                                            \
                                                                                                                        \
                                                long long duration = f(*A, input, output, feature, Ti, Tk, chunk_size); \
                                                                                                                        \
                                                cycle_counts.insert(duration);                                          \
                                                                                                                        \
                                            }                                                                           \
                                            cycle_counts.process();                                                     \
                                            cycle_counts.print();                                                       \
                                            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                            std::cout << "Warmup Median Time: " << median_time << std::endl;            \
                                                                                                                        \
                                        }

#else // LARGE_ARRAY

#define RUN_WARMUP_EXPR(f, A, Ti, Tk)   {                                                                               \
                                            stats_t<long long, ITYPE> cycle_counts("Cycle Counts");                     \
                                                                                                                        \
                                            for ( ITYPE i = 0; i < (n/WARMUP_DIVIDER); i++ ) {                          \
                                                                                                                        \
                                                T *input = &(B[ (i % layers) ][0]);                                     \
                                                T *output = &(C[ (i % layers) ][0]);                                    \
                                                                                                                        \
                                                CACHE_FLUSH;                                                            \
                                                                                                                        \
                                                long long duration = f(*A, input, output, feature, Ti, Tk, chunk_size); \
                                                                                                                        \
                                                cycle_counts.insert(duration);                                          \
                                                                                                                        \
                                            }                                                                           \
                                            cycle_counts.process();                                                     \
                                            cycle_counts.print();                                                       \
                                            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                            std::cout << "Warmup Median Time: " << median_time << std::endl;            \
                                                                                                                        \
                                        }




#endif // LARGE_ARRAY


#ifdef LARGE_ARRAY
#define RUN_HYBRID_WARMUP_EXPR(f,A,wl,ppt){                                                                             \
                                            stats_t<long long, ITYPE> cycle_counts("Cycle_count");                      \
                                                                                                                        \
                                            for ( ITYPE i = 0; i < n; i++ ) {                                           \
                                                                                                                        \
                                                T *input = &B[ (i % layers) * (ncols * feature) ];                      \
                                                T *output = &C[ (i % layers) * (nrows * feature) ];                     \
                                                                                                                        \
                                                CACHE_FLUSH;                                                            \
                                                                                                                        \
                                                long long duration = f(*A, input, output, feature, wl, ppt, num_threads, chunk_size); \
                                                                                                                        \
                                                cycle_counts.insert(duration);                                          \
                                                                                                                        \
                                            }                                                                           \
                                            cycle_counts.process();                                                     \
                                            cycle_counts.print();                                                       \
                                            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                            std::cout << "Warmup Median Time: " << median_time << std::endl;            \
                                        }

#else // LARGE_ARRAY
#define RUN_HYBRID_WARMUP_EXPR(f,A,wl,ppt){                                                                             \
                                            stats_t<long long, ITYPE> cycle_counts("Cycle_count");                      \
                                                                                                                        \
                                            for ( ITYPE i = 0; i < (n/WARMUP_DIVIDER); i++ ) {                          \
                                                                                                                        \
                                                T *input = &(B[ (i % layers) ][0]);                                     \
                                                T *output = &(C[ (i % layers) ][0]);                                    \
                                                                                                                        \
                                                CACHE_FLUSH;                                                            \
                                                                                                                        \
                                                long long duration = f(*A, input, output, feature, wl, ppt, num_threads, chunk_size); \
                                                                                                                        \
                                                cycle_counts.insert(duration);                                          \
                                                                                                                        \
                                            }                                                                           \
                                            cycle_counts.process();                                                     \
                                            cycle_counts.print();                                                       \
                                            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                            std::cout << "Warmup Median Time: " << median_time << std::endl;            \
                                        }


#define RUN_HYBRID_PER_CORE_WARMUP_EXPR(f,A,wl,ppt){                                                                    \
                                            stats_t<long long, ITYPE> cycle_counts("Cycle_count");                      \
                                            long long per_core_runtime[num_threads];                                    \
                                                                                                                        \
                                            for ( ITYPE i = 0; i < n; i++ ) {                                           \
                                                                                                                        \
                                                T *input = &(B[ (i % layers) ][0]);                                     \
                                                T *output = &(C[ (i % layers) ][0]);                                    \
                                                                                                                        \
                                                CACHE_FLUSH;                                                            \
                                                                                                                        \
                                                long long duration = f(*A, input, output, feature, wl, ppt, num_threads, chunk_size, per_core_runtime); \
                                                                                                                        \
                                                cycle_counts.insert(duration);                                          \
                                                                                                                        \
                                            }                                                                           \
                                            cycle_counts.process();                                                     \
                                            cycle_counts.print();                                                       \
                                            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                            std::cout << "Warmup Median Time: " << median_time << std::endl;            \
                                        }


#define RUN_HYBRID_PER_PANEL_WARMUP_EXPR(f,A,wl,ppt){                                                                   \
                                            stats_t<long long, ITYPE> cycle_counts("Cycle_count");                      \
                                            long long per_core_runtime[num_threads];                                    \
                                            long long *per_panel_cycle_counts = new long long[A->num_panels];           \
                                                                                                                        \
                                            for ( ITYPE i = 0; i < (n/WARMUP_DIVIDER); i++ ) {                                           \
                                                                                                                        \
                                                T *input = &(B[ (i % layers) ][0]);                                     \
                                                T *output = &(C[ (i % layers) ][0]);                                    \
                                                                                                                        \
                                                CACHE_FLUSH;                                                            \
                                                                                                                        \
                                                long long duration = f(*A, input, output, feature, wl, ppt, num_threads, chunk_size, per_core_runtime, per_panel_cycle_counts); \
                                                                                                                        \
                                                cycle_counts.insert(duration);                                          \
                                                                                                                        \
                                            }                                                                           \
                                            cycle_counts.process();                                                     \
                                            cycle_counts.print();                                                       \
                                            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                            std::cout << "Warmup Median Time: " << median_time << std::endl;            \
                                            delete[] per_panel_cycle_counts;                                            \
                                        }




#endif // LARGE_ARRAY

#ifdef LARGE_ARRAY

#define RUN_PER_CORE_WARMUP_EXPR(f, A, Ti, Tk)   {                                                                      \
                                            stats_t<long long, ITYPE> cycle_counts;                                     \
                                            cycle_counts.name = "Cycle Counts";                                         \
                                            long long per_core_cycle_counts[num_threads];                               \
                                                                                                                        \
                                            for ( ITYPE i = 0; i < n; i++ ) {                                           \
                                                                                                                        \
                                                T *input = &B[ (i % layers) * (ncols * feature) ];                      \
                                                T *output = &C[ (i % layers) * (nrows * feature) ];                     \
                                                                                                                        \
                                                CACHE_FLUSH;                                                            \
                                                                                                                        \
                                                long long duration = f(*A, input, output, feature, Ti, Tk, chunk_size, per_core_cycle_counts); \
                                                                                                                        \
                                                cycle_counts.insert(duration);                                          \
                                                                                                                        \
                                            }                                                                           \
                                            cycle_counts.process();                                                     \
                                            cycle_counts.print();                                                       \
                                            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                            std::cout << "Warmup Median Time: " << median_time << std::endl;            \
                                                                                                                        \
                                        }


#define RUN_PER_PANEL_WARMUP_EXPR(f, A, Ti, Tk)   {                                                                           \
                                            std::cout << "Running per core timing with " << num_threads << " cores" << std::endl;    \
                                            long long core_cycle_counts[num_threads];                                   \
                                            long long duration;                                                         \
                                            stats_t<long long, ITYPE> per_core_cycle_counts[num_threads];               \
                                            stats_t<double, ITYPE> per_core_execution_times[num_threads];               \
                                            stats_t<long long, ITYPE> cycle_counts;                                     \
                                            ITYPE num_panels = CEIL(A->nrows, Ti);                                      \
                                            ITYPE num_panels_per_thread = CEIL(num_panels, num_threads);                \
                                            ITYPE panel_count_per_thread[num_threads];                                  \
                                            long long *per_panel_cycle_counts = new long long[num_panels_per_thread * num_threads]; \
                                            stats_t<long long, ITYPE> per_panel_cycle_counts_stats[num_threads];        \
                                            for (ITYPE i = 0; i < num_threads; i++) {                                   \
                                                per_core_cycle_counts[i].name = "core:" + std::to_string(i);            \
                                                per_core_execution_times[i].name = "core:" + std::to_string(i);         \
                                                per_panel_cycle_counts_stats[i].resize( n * num_panels_per_thread );    \
                                                per_panel_cycle_counts_stats[i].name = "core:" + std::to_string(i);     \
                                                panel_count_per_thread[i] = 0;                                          \
                                            }                                                                           \
                                            std::cout << "Num row panels: " << num_panels << std::endl;                 \
                                            for ( ITYPE i = 0; i < num_panels; i++ ) {                                  \
                                                panel_count_per_thread[ i % num_threads ]++;                            \
                                            }                                                                           \
                                            for ( ITYPE tid = 0; tid < num_threads; tid++ ) {                           \
                                                std::cout << "core: " << tid << " -- Num Panels: " << panel_count_per_thread[tid] << std::endl; \
                                            }                                                                           \
                                            cycle_counts.name = "Cycle Counts";                                         \
                                                                                                                        \
                                            for ( ITYPE i = 0; i < n; i++ ) {                                           \
                                                                                                                        \
                                                T *input = &B[ (i % layers) * (ncols * feature) ];                      \
                                                T *output = &C[ (i % layers) * (nrows * feature) ];                     \
                                                                                                                        \
                                                CACHE_FLUSH;                                                            \
                                                                                                                        \
                                                duration = f(*A, input, output, feature, Ti, Tk, chunk_size, core_cycle_counts, per_panel_cycle_counts);    \
                                                                                                                        \
                                                for (ITYPE i = 0; i < num_threads; i++) {                               \
                                                    per_core_cycle_counts[i].insert(core_cycle_counts[i]);              \
                                                }                                                                       \
                                                cycle_counts.insert(duration);                                          \
                                                                                                                        \
                                                for (ITYPE tid = 0; tid < num_threads; tid++) {                         \
                                                    for (ITYPE rid = 0; rid < panel_count_per_thread[tid]; rid++) {     \
                                                        per_panel_cycle_counts_stats[tid].insert(per_panel_cycle_counts[tid + (rid * num_threads)]);  \
                                                    }                                                                   \
                                                }                                                                       \
                                                                                                                        \
                                            }                                                                           \
                                            cycle_counts.process();                                                     \
                                            cycle_counts.print();                                                       \
                                            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                            std::cout << "Warmup Median Time: " << median_time << std::endl;                   \
                                        }

#else // LARGE_ARRAY

#define RUN_PER_CORE_WARMUP_EXPR(f, A, Ti, Tk)   {                                                                      \
                                            stats_t<long long, ITYPE> cycle_counts;                                     \
                                            cycle_counts.name = "Cycle Counts";                                         \
                                            long long per_core_cycle_counts[num_threads];                               \
                                                                                                                        \
                                            for ( ITYPE i = 0; i < n; i++ ) {                                           \
                                                                                                                        \
                                                T *input = &(B[ (i % layers) ][0]);                                     \
                                                T *output = &(C[ (i % layers) ][0]);                                    \
                                                                                                                        \
                                                CACHE_FLUSH;                                                            \
                                                                                                                        \
                                                long long duration = f(*A, input, output, feature, Ti, Tk, chunk_size, per_core_cycle_counts); \
                                                                                                                        \
                                                cycle_counts.insert(duration);                                          \
                                                                                                                        \
                                            }                                                                           \
                                            cycle_counts.process();                                                     \
                                            cycle_counts.print();                                                       \
                                            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                            std::cout << "Warmup Median Time: " << median_time << std::endl;            \
                                                                                                                        \
                                        }


#define RUN_PER_PANEL_WARMUP_EXPR(f, A, Ti, Tk)   {                                                                     \
                                            stats_t<long long, ITYPE> cycle_counts;                                     \
                                            cycle_counts.name = "Cycle Counts";                                         \
                                            long long core_cycle_counts[num_threads];                                   \
                                            long long *per_panel_cycle_counts = new long long[A->num_panels * n];       \
                                            stats_t<long long, ITYPE> per_core_cycle_counts[num_threads];               \
                                            stats_t<double, ITYPE> per_core_execution_times[num_threads];               \
                                            for (ITYPE i = 0; i < num_threads; i++) {                                   \
                                                per_core_cycle_counts[i].name = "core:" + std::to_string(i);            \
                                                per_core_execution_times[i].name = "core:" + std::to_string(i);         \
                                            }                                                                           \
                                                                                                                        \
                                            for ( ITYPE i = 0; i < (WARMUP_DIVIDER); i++ ) {                            \
                                                T *input = &(B[ (i % layers) ][0]);                                     \
                                                T *output = &(C[ (i % layers) ][0]);                                    \
                                                long long *iter_panel_cycle_counts = &per_panel_cycle_counts[A->num_panels * i]; \
                                                                                                                        \
                                                CACHE_FLUSH;                                                            \
                                                                                                                        \
                                                long long duration = f(*A, input, output, feature, Ti, Tk, chunk_size, core_cycle_counts, iter_panel_cycle_counts);    \
                                                                                                                        \
                                                cycle_counts.insert(duration);                                          \
                                                                                                                        \
                                                for (ITYPE tid = 0; tid < num_threads; tid++) {                         \
                                                    per_core_cycle_counts[tid].insert( core_cycle_counts[tid] );        \
                                                }                                                                       \
                                                                                                                        \
                                            }                                                                           \
                                            cycle_counts.process();                                                     \
                                            cycle_counts.print();                                                       \
                                            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;        \
                                            std::cout << "Warmup Median Time: " << median_time << std::endl;            \
                                            delete[] per_panel_cycle_counts;                                            \
                                        }



#endif // LARGE_ARRAY

#else
#define RUN_WARMUP_EXPR(f, A, Ti, Tk)
#define RUN_PER_CORE_WARMUP_EXPR(f, A, Ti, Tk)
#define RUN_PER_PANEL_WARMUP_EXPR(f, A, Ti, Tk)
#define RUN_HYBRID_WARMUP_EXPR(f,A,wl,ppt)
#define RUN_HYBRID_PER_CORE_WARMUP_EXPR(f,A,wl,ppt)
#endif


#ifdef RUN_CORRECTNESS_CHECK
#define RUN_CHECK(f, A, Ti, Tk) {                                                                                                           \
                                    T *check_O = (T *) std::aligned_alloc( ALLOC_ALIGNMENT,  nrows * (feature + PADDING_C) * sizeof(T) );   \
                                    T *O = (T *) std::aligned_alloc( ALLOC_ALIGNMENT, nrows * (feature + PADDING_C) * sizeof(T) );          \
                                                                                                                                            \
                                    reset_matrix_helper<T, ITYPE>( check_O, (size_t) (nrows * (feature + PADDING_C)) );                     \
                                    reset_matrix_helper<T, ITYPE>( O, (size_t) (nrows * (feature + PADDING_C)) );                           \
                                                                                                                                            \
                                    f<T, ITYPE>( *A, B[0], O, feature, Ti, Tk, chunk_size );                                                \
                                    auto check_result = check_simple<T, ITYPE>( *S_csr, B[0], check_O, feature, O );                        \
                                    std::cout << "Is correct? " << check_result << std::endl;                                               \
                                                                                                                                            \
                                    std::free(check_O);                                                                                     \
                                    std::free(O);                                                                                           \
                                }
#else
#define RUN_CHECK(f, A, Ti, Tk)
#endif

#ifdef TRACK_PARALLEL_STATS
    #define RUN_PERF_RESET(p, n) reset_perf_stats_helper<long long, ITYPE>(p, n)
    #define RUN_DRAM_RESET(p) reset_dram_stats_helper<long long, ITYPE>(p, NUM_MEMORY_CHANNELS * 2)
#else
    #define RUN_PERF_RESET(p, n)
    #define RUN_DRAM_RESET(p)
#endif

#define PRINT_TYPE(T) std::cout << "T: " << T << std::endl;


#define LARGE_ARRAY_SIZE ( 1024 * 1024 * 1024 / sizeof(double) )
#define ERROR_THRESHOLD 0.03    // We can tolerate a 3% error


template <typename T, typename ITYPE>
void data_movement_experiment( std::string mtx_filename, ITYPE feature, ITYPE Ti, ITYPE Tj, ITYPE Tk, ITYPE n, ITYPE num_threads = 1, ITYPE chunk_size = 1, ITYPE layers = 1, ITYPE num_panels = 0, ITYPE *num_panels_per_thread = nullptr, struct workitem *work_list = nullptr, ITYPE *panel_worklist = nullptr, ITYPE num_partitions = 2, std::vector<struct panel_t> *adaptive_panels = nullptr, std::string stm_filename = "", ITYPE fixed_nnzs = 0 )
{
    ITYPE nrows, ncols, nnzs;
    std::pair<ITYPE, ITYPE> *locs = nullptr;
    T* vals = nullptr;
    bool using_global = true;

    CSR<T, ITYPE> *S_csr = nullptr;
    DCSC<T, ITYPE> *S_dcsc = nullptr;
    CSF<T, ITYPE> *S_csf = nullptr;
    ATM<T, ITYPE> *S_atm = nullptr;

    if (global_locs && global_vals) {
        locs = global_locs;
        vals = global_vals;
        nrows = global_nrows;
        ncols = global_ncols;
        nnzs = global_nnzs;
    } else {
        using_global = false;
        read_mtx_matrix_into_arrays(mtx_filename.c_str(), &locs, &vals, &nrows, &ncols, &nnzs);
        assert( locs && vals && (nnzs > 0) && (ncols > 0) && (nnzs > 0) && "Could not read mtx file" );
    }

    S_csr = new CSR<T, ITYPE>( nrows, ncols, nnzs, locs, vals );

    #if defined (RUN_CSR_WORKLIST_EXPR)
        S_csr->augment_panel_ptrs( num_panels, work_list );
        std::pair<ITYPE, ITYPE> *pairs_worklist;
    #endif

    std::cout << "M: " << S_csr->nrows << std::endl;
    std::cout << "N: " << S_csr->ncols << std::endl;
    std::cout << "NNZ: " << S_csr->nnzs << std::endl;
    // #if defined(RUN_CSR_EXPR) || defined(RUN_CSR_JSTREAM_EXPR) || defined(RUN_SIMPLE_CSR)
    //     S_csr = new CSR<T, ITYPE>( nrows, ncols, nnzs, locs, vals );
    // #endif

    if (chunk_size == -1) {
        ITYPE num_panels = CEIL(S_csr->nrows, Ti);
        chunk_size = CEIL(num_panels, num_threads);
    }
    print_status("Chunk size for openmp: %d\n", chunk_size);
    // std::cout << "Chunk Size: " << chunk_size << std::endl;

    #if defined(RUN_DCSC_JSTREAM_EXPR) || defined(RUN_DCSC_WORKLIST) || defined(RUN_DCSC_JSTREAM_COMPILER_EXPR)
        if (work_list && num_panels_per_thread) {
            std::cerr << "STATUS: Building work item based DCSC matrix" << std::endl;
            // ITYPE num_panels = 0;
            // for (ITYPE tid = 0; tid < num_threads; tid++) {
            //     num_panels += num_panels_per_thread[tid];
            // }
            S_dcsc = new DCSC<T, ITYPE>( nrows, ncols, nnzs, locs, vals, num_panels, work_list );

            S_stm = new STM<T, ITYPE>( nrows, ncols, nnzs, locs, vals, Ti, Tj );
            std::cout << "Matrices are correct?: " << verify_matrices(S_csr, S_dcsc, S_stm) << std::endl;
        } else if (work_list) {
            std::cerr << "STATUS: Building work item based DCSC matrix" << std::endl;
            S_dcsc = new DCSC<T, ITYPE>( nrows, ncols, nnzs, locs, vals, num_panels, work_list );

            S_stm = new STM<T, ITYPE>( nrows, ncols, nnzs, locs, vals, Ti, Tj );
            std::cout << "Matrices are correct?: " << verify_matrices(S_csr, S_dcsc, S_stm) << std::endl;
        } else {
            std::cerr << "STATUS: Building statically split DCSC matrix" << std::endl;
            S_dcsc = new DCSC<T, ITYPE>( nrows, ncols, nnzs, locs, vals, Ti );
            // S_stm = new STM<T, ITYPE>( nrows, ncols, nnzs, locs, vals, Ti, Tj );


            bool dcsc_matrix_correct = verify_matrices( *S_csr, *S_dcsc );

            std::cout << "DCSC Matrix is correct? : " << dcsc_matrix_correct << std::endl;


            if (!dcsc_matrix_correct) {
                std::cout << "MATRIX ERROR" << std::endl;
                std::exit(EXIT_FAILURE);
            }

            // std::exit(0);

            // std::cout << "Matrices are correct?: " << verify_matrices(S_csr, S_dcsc, S_stm) << std::endl;
        }
    #endif

    #if defined(RUN_CSR_KSTREAM_EXPR)
        if ( stm_filename.size() > 0 ) {
            S_stm = read_serialized_stm( stm_filename.c_str(), S_csr );
            Ti = S_stm->Ti;
        } else if (adaptive_panels != nullptr) {
            S_stm = new STM<TYPE, ITYPE>(nrows, ncols, nnzs, locs, vals, Ti, *adaptive_panels);
        } else {
            S_stm = new STM<T, ITYPE>( nrows, ncols, nnzs, locs, vals, Ti, Tj );
        }
        bool stm_correct = verify_matrix_structure( S_csr, S_stm );
        std::cout << "STM Matrix correct? " << stm_correct << std::endl;
    #endif

    #if defined(RUN_SPLIT_JSTREAM_EXPR)
        S_split = new SPLIT_CSR<T, ITYPE>( nrows, ncols, nnzs, locs, vals, num_partitions );
    #endif

    #if defined(RUN_CSR_ATM_KSTREAM)
        // S_atm = new ATM<T, ITYPE>( nrows, ncols, nnzs, locs, vals, adaptive_panels->size(), *adaptive_panels );
        S_atm = new ATM<T, ITYPE>( nrows, ncols, nnzs, locs, vals, *adaptive_panels );
        auto atm_correct = verify_matrix_structure( *S_csr, *S_atm );
        std::cout << "ATM Matrix correct? " << atm_correct << std::endl;
    #endif

    #if defined(RUN_CSF_EXPR)
        // S_csf = new CSF<T, ITYPE>( nrows, ncols, nnzs, locs, vals, Ti, Tj, Tk );
        // First generate the tiled CSF coordinate ordering, then call the constructor
        ITYPE *C1, *C2, *C3, *C4;
        // ITYPE num_csf_panels = generate_coo_representation(Ti, Tj, nnzs, locs, vals, &C1, &C2, &C3, &C4);

        ITYPE num_csf_panels = 0;
        if ( (*adaptive_panels).size() > 0 ) {
            print_status("[CSF] Building rassm csf matrix\n");
            num_csf_panels = generate_coo_representation_rassm(nnzs, locs, vals, *adaptive_panels, &C1, &C2, &C3, &C4);
        } else {
            if (fixed_nnzs > 0) {
                print_status("[CSF] Building fixed nnzs csf matrix\n");
                num_csf_panels = generate_fixed_nnzs_coo_representation(nnzs, locs, vals, fixed_nnzs, &C1, &C2, &C3, &C4);
            } else {
                print_status("[CSF] Building fixed size csf matrix\n");
                num_csf_panels = generate_coo_representation(Ti, Tj, nnzs, locs, vals, &C1, &C2, &C3, &C4);
            }
        }

        std::cout << "Unique Tile Rows: " << num_csf_panels << std::endl;

        // call the constructor
        S_csf = new CSF<T, ITYPE>( nrows, ncols, nnzs, num_csf_panels, C1, C2, C3, C4, vals );

        // verify the constructed matrix via a function call

        bool csf_correct = verify_matrix_structure( *S_csr, *S_csf );
        std::cout << "CSF Matrix correct? " << csf_correct << std::endl;
    #endif

/*
    #if defined(RUN_DCSC_JSTREAM_EXPR) && defined(RUN_CSR_KSTREAM_EXPR)
        std::cout << "Matrices are correct?: " << verify_matrices(S_csr, S_dcsc, S_stm) << std::endl;
    #endif
*/

    #if defined(RUN_HYB_EXPR) || defined(RUN_DYN_HYB_EXPR) || defined(RUN_DYN_HYB_DM_EXPR) || defined(RUN_DYN_HYB_COMP_VEC_EXPR)
        std::cerr << "Building DCSH matrix for dynamic runs" << std::endl;

        std::pair<ITYPE, ITYPE> *pairs_worklist;
        S_dcsh = new DCSH<T, ITYPE>( nrows, ncols, nnzs, locs, vals, num_panels, work_list );

        bool S_dcsh_correct = verify_dcsh_matrix<T, ITYPE>( S_csr, S_dcsh );
        std::cout << "DCSH Matrix construction correct? : " << S_dcsh_correct << std::endl;

        /*
        ITYPE temp_panel_Tk[num_panels];
        for ( ITYPE i = 0; i < num_panels; i++ ) {
            temp_panel_Tk[i] = work_list[i].Tk;
        }
        S_dcsh->set_per_panel_Tk( temp_panel_Tk );

        std::sort( work_list, work_list + num_panels, []( struct workitem &a, struct workitem &b) {
                return a.nnzs < b.nnzs;
            });


        //  new DCSH<T, ITYPE>(nrows, ncols, nnz, locs, vals, num_panels, panel_type, panel_offset);
        bool S_dcsh_correct = verify_dcsh_matrix<T, ITYPE>( S_csr, S_dcsh );
        if (!S_dcsh_correct) {
            std::exit(-1);
        }
        std::cout << "DCSH Matrix correct? : " << S_dcsh_correct << std::endl;

        ITYPE *panel_order = new ITYPE[ num_panels ];
        for ( ITYPE i = 0; i < num_panels; i++ ) {
            panel_order[i] = work_list[i].panel_id;
            std::cerr << "panel: " << panel_order[i] << " -- " << work_list[i].nnzs << std::endl;
        }
        */


        ITYPE max_panels_per_thread = 0;
        for (ITYPE i = 0; i < num_threads; i++) {
            if (num_panels_per_thread[i] > max_panels_per_thread) {
                max_panels_per_thread = num_panels_per_thread[i];
            }
        }

        ITYPE temp_panel_Tk[num_panels];

        pairs_worklist = (std::pair<ITYPE, ITYPE> *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(std::pair<ITYPE, ITYPE>) * max_panels_per_thread * num_threads );
        for (ITYPE tid = 0; tid < num_threads; tid++) {
            for (ITYPE p = 0; p < num_panels_per_thread[tid]; p++) {
                ITYPE panel_id = panel_worklist[ p * num_threads + tid ];
                pairs_worklist[ p * num_threads + tid ] = { panel_id, work_list[panel_id].Tk };
                temp_panel_Tk[ panel_id ] = work_list[panel_id].Tk;
            }
        }

        S_dcsh->set_per_panel_Tk( temp_panel_Tk );


    #endif


    // set number of threads
    // omp_set_num_threads( (int) num_threads );


    size_t sizeB = ncols * (feature + PADDING_B) * sizeof(T);
    size_t sizeB_rounded = CEIL(sizeB, PAGE_SIZE) * PAGE_SIZE;
    size_t countB = CEIL(sizeB_rounded, sizeof(T));

    size_t sizeC = nrows * (feature + PADDING_C) * sizeof(T);
    size_t sizeC_rounded = CEIL(sizeC, PAGE_SIZE) * PAGE_SIZE;
    size_t countC = CEIL(sizeC_rounded, sizeof(T));

    #ifdef LARGE_ARRAY
        // Generate the Input and Output Matrices
        // T *B = generate_dense<T, ITYPE>(ncols, (feature + PADDING_B), layers);
        // T *C = generate_zeroes<T, ITYPE>(nrows, (feature + PADDING_C), layers);
        // size_t sizeC = ((size_t) nrows) * ((size_t) feature) * ((size_t) layers);

        size_t countC_alloc = countC * (size_t) layers;
        size_t countB_alloc = countB * (size_t) layers;

        T *B = generate_dense<T, ITYPE>( countB_alloc );
        T *C = generate_zeroes<T, ITYPE>( countC_alloc );

    #else
        T *B[layers];
        T *C[layers];

        for ( ITYPE l = 0; l < layers; l++ ) {
            B[l] = generate_dense<T, ITYPE>( countB );
            C[l] = generate_zeroes<T, ITYPE>( countC );
        }
    #endif

    // launch_experiment<T, ITYPE, CSR<T, ITYPE>, spmm_csr_jstream<T, ITYPE>>( spmm_csr_jstream<T, ITYPE>, S_csr, B, C, feature, Ti, Tk, num_threads, n, layers );

    // data volume and flop count of the input matrix
    // size_t data_volume =  ((size_t) (nrows + ncols) * feature * sizeof(T)) + ((size_t) (nrows+1) * sizeof(ITYPE)) + ((size_t) S_csr->nnzs * (sizeof(T) + sizeof(ITYPE)));
    // size_t flop_count = 2 * ((size_t) S_csr->nnzs) * ((size_t) feature);

    #ifdef TRACK_PARALLEL_STATS
        perf::init_papi(); // Initialize PAPI
        ITYPE temp;
        #ifndef TRACK_CACHE_ONLY
            perf *dram_perf;
            perf *reg_perf;
            perf *cache_perf;
            perf *stall_perf;
            perf *tlb_perf;
            perf *cache_stall_perf;
            perf *l3_cache_perf;
            perf *fill_perf;

            perf *dram_precharge_rd_perf;
            perf *dram_precharge_wr_perf;
            perf *dram_activate_rd_perf;
            perf *dram_activate_wr_perf;

            stats_t<long long, ITYPE> **dram_counters;
            stats_t<long long, ITYPE> **reg_counters;
            stats_t<long long, ITYPE> **cache_counters;
            stats_t<long long, ITYPE> **stall_counters;
            stats_t<long long, ITYPE> **tlb_counters;
            stats_t<long long, ITYPE> **cache_stall_counters;
            stats_t<long long, ITYPE> **l3_cache_counters;
            stats_t<long long, ITYPE> **fill_counters;

            stats_t<long long, ITYPE> **dram_precharge_rd_counters;
            stats_t<long long, ITYPE> **dram_precharge_wr_counters;

            stats_t<long long, ITYPE> **dram_activate_rd_counters;
            stats_t<long long, ITYPE> **dram_activate_wr_counters;

            // Setup all the performance counters
            // setup_perf_counters_helper<T, ITYPE>(perf_event_set::REG_EVENTS, num_threads, &reg_perf, &reg_counters, &temp);
            // setup_perf_counters_helper<T, ITYPE>(perf_event_set::CACHE_EVENTS, num_threads, &cache_perf, &cache_counters, &temp);
            // setup_perf_counters_helper<T, ITYPE>(perf_event_set::STALL_EVENTS, num_threads, &stall_perf, &stall_counters, &temp);
            // setup_dram_counters_helper<T, ITYPE>(num_threads, &dram_perf, &dram_counters, &temp);

            // setup_native_perf_counters_helper<T, ITYPE>(perf_event_set::CACHE_STALL_EVENTS, num_threads, &cache_stall_perf, &cache_stall_counters, &temp);

            // setup_native_perf_counters_helper<T, ITYPE>(perf_event_set::DRAM_PRECHARGE_RD_EVENTS, num_threads, &dram_precharge_rd_perf, &dram_precharge_rd_counters, &temp);
            // setup_native_perf_counters_helper<T, ITYPE>(perf_event_set::DRAM_PRECHARGE_WR_EVENTS, num_threads, &dram_precharge_wr_perf, &dram_precharge_wr_counters, &temp);

            // setup_native_perf_counters_helper<T, ITYPE>(perf_event_set::DRAM_ACTIVATE_RD_EVENTS, num_threads, &dram_activate_rd_perf, &dram_activate_rd_counters, &temp);
            // setup_native_perf_counters_helper<T, ITYPE>(perf_event_set::DRAM_ACTIVATE_WR_EVENTS, num_threads, &dram_activate_wr_perf, &dram_activate_wr_counters, &temp);

            #if defined(REG_STATS)
                setup_perf_counters_helper<T, ITYPE>(perf_event_set::REG_EVENTS, num_threads, &reg_perf, &reg_counters, &temp);
                RUN_PERF_RESET(reg_counters, perf::NUM_EVENTS_PER_SET);
            #elif defined(CACHE_STATS)
                setup_perf_counters_helper<T, ITYPE>(perf_event_set::CACHE_EVENTS, num_threads, &cache_perf, &cache_counters, &temp);
                // setup_native_perf_counters_helper<T, ITYPE>(perf_event_set::FILL_EVENTS, num_threads, &cache_perf, &cache_counters, &temp);
                // setup_native_perf_counters_helper<T, ITYPE>(perf_event_set::NAMED_CACHE_EVENTS, num_threads, &cache_perf, &cache_counters, &temp);
                RUN_PERF_RESET(cache_counters, perf::NUM_EVENTS_PER_SET);
            #elif defined(FILL_STATS)
                // setup_native_perf_counters_helper<T, ITYPE>( perf_event_set::FILL_EVENTS, num_threads, &fill_perf, &fill_counters, &temp );
                setup_native_perf_counters_helper<T, ITYPE>( perf_event_set::FILL_EVENTS, num_threads, &fill_perf, &fill_counters, &temp );
            #elif defined(STALL_STATS)
                // setup_perf_counters_helper<T, ITYPE>(perf_event_set::STALL_EVENTS, num_threads, &stall_perf, &stall_counters, &temp);
                setup_native_perf_counters_helper<T, ITYPE>(perf_event_set::NAMED_STALL_EVENTS, num_threads, &stall_perf, &stall_counters, &temp);
                RUN_PERF_RESET(stall_counters, NUM_NAMED_STALL_EVENTS);
            #elif defined(DRAM_STATS)
                setup_dram_counters_helper<T, ITYPE>(num_threads, &dram_perf, &dram_counters, &temp);
                RUN_DRAM_RESET(dram_counters);
            #elif defined(CACHE_STALL_STATS)
                setup_native_perf_counters_helper<T, ITYPE>(perf_event_set::CACHE_STALL_EVENTS, num_threads, &cache_stall_perf, &cache_stall_counters, &temp);
                RUN_PERF_RESET(cache_stall_counters, NUM_CACHE_STALL_EVENTS);
            #elif defined(L3_CACHE_STATS)
                setup_native_perf_counters_helper<T, ITYPE>(perf_event_set::L3_CACHE_EVENTS, num_threads, &l3_cache_perf, &l3_cache_counters, &temp);
            #elif defined(TLB_STATS)
                setup_perf_counters_helper<T, ITYPE>(perf_event_set::TLB_EVENTS, num_threads, &tlb_perf, &tlb_counters, &temp);
                RUN_PERF_RESET(tlb_counters, perf::NUM_EVENTS_PER_SET);
            #elif defined(DRAM_PRECHARGE_RD_STATS)
                setup_native_perf_counters_helper<T, ITYPE>(perf_event_set::DRAM_PRECHARGE_RD_EVENTS, num_threads, &dram_precharge_rd_perf, &dram_precharge_rd_counters, &temp);
                RUN_DRAM_RESET(dram_precharge_rd_counters);
            #elif defined(DRAM_PRECHARGE_WR_STATS)
                setup_native_perf_counters_helper<T, ITYPE>(perf_event_set::DRAM_PRECHARGE_WR_EVENTS, num_threads, &dram_precharge_wr_perf, &dram_precharge_wr_counters, &temp);
                RUN_DRAM_RESET(dram_precharge_wr_counters);
            #elif defined(DRAM_ACTIVATE_RD_STATS)
                setup_native_perf_counters_helper<T, ITYPE>(perf_event_set::DRAM_ACTIVATE_RD_EVENTS, num_threads, &dram_activate_rd_perf, &dram_activate_rd_counters, &temp);
                RUN_DRAM_RESET(dram_activate_rd_counters);
            #elif defined(DRAM_ACTIVATE_WR_STATS)
                setup_native_perf_counters_helper<T, ITYPE>(perf_event_set::DRAM_ACTIVATE_WR_EVENTS, num_threads, &dram_activate_wr_perf, &dram_activate_wr_counters, &temp);
                RUN_DRAM_RESET(dram_activate_wr_counters);
            #endif

            // {
            //     RUN_PERF_RESET(reg_counters, perf::NUM_EVENTS_PER_SET);
            //     RUN_PERF_RESET(cache_counters, perf::NUM_EVENTS_PER_SET);
            //     RUN_PERF_RESET(stall_counters, perf::NUM_EVENTS_PER_SET);
            //     RUN_PERF_RESET(cache_stall_counters, NUM_CACHE_STALL_EVENTS);

            //     RUN_DRAM_RESET(dram_counters);
            //     RUN_DRAM_RESET(dram_precharge_rd_counters);
            //     RUN_DRAM_RESET(dram_precharge_wr_counters);
            //     RUN_DRAM_RESET(dram_activate_rd_counters);
            //     RUN_DRAM_RESET(dram_activate_wr_counters);
            // }

            {
                #ifdef VALIDATE_PERF_COUNTERS
                    setup_dram_counters_helper<T, ITYPE>(num_threads, &dram_perf, &dram_counters, &temp);
                    RUN_DRAM_RESET(dram_counters);

                    // Perform a validation test on the DRAM counters to check amount of memory read
                    #ifdef TRACK_PARALLEL_STATS

        validate_dram_counters<T, ITYPE>(dram_perf, dram_counters, temp, num_threads);
                    #else
                    validate_dram_counters<T, ITYPE>(nullptr, nullptr, 0, num_threads);
                    #endif

                    // Lets also validate the timing performance of the timestamp counter
                    validate_timing<T, ITYPE>();

                    {
                        RUN_PERF_RESET(reg_counters, perf::NUM_EVENTS_PER_SET);
                        RUN_PERF_RESET(cache_counters, perf::NUM_EVENTS_PER_SET);
                        RUN_PERF_RESET(stall_counters, perf::NUM_EVENTS_PER_SET);
                        RUN_DRAM_RESET(dram_counters);
                    }
                #endif
            }

        #else // TRACK_CACHE_ONLY

            // perf *cache_only_perf;
            // stats_t<long long, ITYPE> **cache_only_counters;

            // setup_perf_counters_helper<T, ITYPE>(perf_event_set::CACHE_ONLY_EVENTS, num_threads, &cache_only_perf, &cache_only_counters, &temp);
            // RUN_PERF_RESET(cache_only_counters, perf::NUM_EVENTS_PER_SET);
        #endif // TRACK_CACHE_ONLY
    #endif



    #if defined(RUN_ROW_PANEL_COMPILER)
        PRINT_TYPE("CSR_ROW_PANEL_COMPILER");

        // RUN_CHECK(csr_row_panel_compiler_vectorized, S_csr, Ti, Tk);


        #ifdef TRACK_PER_CORE_RUNTIME
            #ifdef TRACK_PER_PANEL_RUNTIME
                RUN_PER_PANEL_WARMUP_EXPR(csr_row_panel_compiler_vectorized, S_csr, Ti, Tk);
            #else
                RUN_PER_CORE_WARMUP_EXPR(csr_row_panel_compiler_vectorized, S_csr, Ti, Tk);
            #endif
        #else
            RUN_WARMUP_EXPR(csr_row_panel_compiler_vectorized, S_csr, Ti, Tk);
        #endif

            // RUN_CHECK(csr_row_panel_compiler_vectorized, S_csr, Ti, Tk);
        MEM_RESET(C, countC, layers);
        #if defined(TRACK_STATS_NEW)
            #if defined(REG_STATS)
                RUN_PERF_HELPER(csr_row_panel_compiler_vectorized, reg_perf, reg_counters, perf_event_set::REG_EVENTS, S_csr, Ti, Tk);
            #elif defined(CACHE_STATS)
                RUN_PERF_HELPER(csr_row_panel_compiler_vectorized, cache_perf, cache_counters, perf_event_set::CACHE_EVENTS, S_csr, Ti, Tk);
            #elif defined(STALL_STATS)
                RUN_PERF_HELPER(csr_row_panel_compiler_vectorized, stall_perf, stall_counters, perf_event_set::STALL_EVENTS, S_csr, Ti, Tk);
            #elif defined(L3_CACHE_STATS)
                RUN_PERF_HELPER(csr_row_panel_compiler_vectorized, l3_cache_perf, l3_cache_counters, perf_event_set::L3_CACHE_EVENTS, S_csr, Ti, Tk);
            #elif defined(DRAM_STATS)
                RUN_DRAM_HELPER(csr_row_panel_compiler_vectorized, dram_perf, dram_counters, perf_event_set::DRAM_EVENTS, S_csr, Ti, Tk);
            #elif defined(CACHE_STALL_STATS)
                RUN_PERF_HELPER(csr_row_panel_compiler_vectorized, cache_stall_perf, cache_stall_counters, perf_event_set::CACHE_STALL_EVENTS, S_csr, Ti, Tk);
            #elif defined(DRAM_PRECHARGE_RD_STATS)
                RUN_DRAM_HELPER(csr_row_panel_compiler_vectorized, dram_precharge_rd_perf, dram_precharge_rd_counters, perf_event_set::DRAM_PRECHARGE_RD_EVENTS, S_csr, Ti, Tk);
            #elif defined(DRAM_PRECHARGE_WR_STATS)
                RUN_DRAM_HELPER(csr_row_panel_compiler_vectorized, dram_precharge_wr_perf, dram_precharge_wr_counters, perf_event_set::DRAM_PRECHARGE_WR_EVENTS, S_csr, Ti, Tk);
            #elif defined(DRAM_ACTIVATE_RD_STATS)
                RUN_DRAM_HELPER(csr_row_panel_compiler_vectorized, dram_activate_rd_perf, dram_activate_rd_counters, perf_event_set::DRAM_ACTIVATE_RD_EVENTS, S_csr, Ti, Tk);
            #elif defined(DRAM_ACTIVATE_WR_STATS)
                RUN_DRAM_HELPER(csr_row_panel_compiler_vectorized, dram_activate_wr_perf, dram_activate_wr_counters, perf_event_set::DRAM_ACTIVATE_WR_EVENTS, S_csr, Ti, Tk);
            #endif
        #else


            #ifdef TRACK_PER_CORE_RUNTIME
                #ifdef TRACK_PER_PANEL_RUNTIME
                    RUN_PER_PANEL_TIMING(csr_row_panel_compiler_vectorized, S_csr, Ti, Tk);
                #else
                    RUN_PER_CORE_TIMING(csr_row_panel_compiler_vectorized, S_csr, Ti, Tk);
                #endif
            #else
                RUN_TIMING(csr_row_panel_compiler_vectorized, S_csr, Ti, Tk);
            #endif
        #endif
    #endif





    #if defined(RUN_SIMPLE_COMPILER_CSR)
        PRINT_TYPE("CSR_SIMPLE_COMPILER_VECTORIZED");

        RUN_CHECK(spmm_simple_parallel_compiler_vectorized, S_csr, Ti, Tk);

        MEM_RESET(C, countC, layers);
        RUN_WARMUP_EXPR(spmm_simple_parallel_compiler_vectorized, S_csr, Ti, Tk);
        MEM_RESET(C, countC, layers);
        #if defined(TRACK_STATS_NEW)
            #if defined(REG_STATS)
                RUN_PERF_HELPER(spmm_simple_parallel_compiler_vectorized, reg_perf, reg_counters, perf_event_set::REG_EVENTS, S_csr, Ti, Tk);
            #elif defined(CACHE_STATS)
                RUN_PERF_HELPER(spmm_simple_parallel_compiler_vectorized, cache_perf, cache_counters, perf_event_set::CACHE_EVENTS, S_csr, Ti, Tk);
            #elif defined(STALL_STATS)
                RUN_PERF_HELPER(spmm_simple_parallel_compiler_vectorized, stall_perf, stall_counters, perf_event_set::STALL_EVENTS, S_csr, Ti, Tk);
            #elif defined(DRAM_STATS)
                RUN_DRAM_HELPER(spmm_simple_parallel_compiler_vectorized, dram_perf, dram_counters, perf_event_set::DRAM_EVENTS, S_csr, Ti, Tk);
            #elif defined(CACHE_STALL_STATS)
                RUN_PERF_HELPER(spmm_simple_parallel_compiler_vectorized, cache_stall_perf, cache_stall_counters, perf_event_set::CACHE_STALL_EVENTS, S_csr, Ti, Tk);
            #elif defined(L3_CACHE_STATS)
                RUN_PERF_HELPER(spmm_simple_parallel_compiler_vectorized, l3_cache_perf, l3_cache_counters, perf_event_set::L3_CACHE_EVENTS, S_csr, Ti, Tk);
            #elif defined(DRAM_PRECHARGE_RD_STATS)
                RUN_DRAM_HELPER(spmm_simple_parallel_compiler_vectorized, dram_precharge_rd_perf, dram_precharge_rd_counters, perf_event_set::DRAM_PRECHARGE_RD_EVENTS, S_csr, Ti, Tk);
            #elif defined(DRAM_PRECHARGE_WR_STATS)
                RUN_DRAM_HELPER(spmm_simple_parallel_compiler_vectorized, dram_precharge_wr_perf, dram_precharge_wr_counters, perf_event_set::DRAM_PRECHARGE_WR_EVENTS, S_csr, Ti, Tk);
            #elif defined(DRAM_ACTIVATE_RD_STATS)
                RUN_DRAM_HELPER(spmm_simple_parallel_compiler_vectorized, dram_activate_rd_perf, dram_activate_rd_counters, perf_event_set::DRAM_ACTIVATE_RD_EVENTS, S_csr, Ti, Tk);
            #elif defined(DRAM_ACTIVATE_WR_STATS)
                RUN_DRAM_HELPER(spmm_simple_parallel_compiler_vectorized, dram_activate_wr_perf, dram_activate_wr_counters, perf_event_set::DRAM_ACTIVATE_WR_EVENTS, S_csr, Ti, Tk);
            #endif
        #else
                RUN_TIMING(spmm_simple_parallel_compiler_vectorized, S_csr, Ti, Tk);
        #endif
    #endif

    #if defined(RUN_SIMPLE_CSR)
        PRINT_TYPE("CSR_SIMPLE");

        RUN_CHECK(spmm_simple_vectorized_parallel, S_csr, Ti, Tk);

        MEM_RESET(C, countC, layers);
        RUN_WARMUP_EXPR(spmm_simple_vectorized_parallel, S_csr, Ti, Tk);
        MEM_RESET(C, countC, layers);
        #if defined(TRACK_STATS_NEW)
            #if defined(REG_STATS)
                RUN_PERF_HELPER(spmm_simple_vectorized_parallel, reg_perf, reg_counters, perf_event_set::REG_EVENTS, S_csr, Ti, Tk);
            #elif defined(CACHE_STATS)
                RUN_PERF_HELPER(spmm_simple_vectorized_parallel, cache_perf, cache_counters, perf_event_set::CACHE_EVENTS, S_csr, Ti, Tk);
            #elif defined(STALL_STATS)
                RUN_PERF_HELPER(spmm_simple_vectorized_parallel, stall_perf, stall_counters, perf_event_set::STALL_EVENTS, S_csr, Ti, Tk);
            #elif defined(DRAM_STATS)
                RUN_DRAM_HELPER(spmm_simple_vectorized_parallel, dram_perf, dram_counters, perf_event_set::DRAM_EVENTS, S_csr, Ti, Tk);
            #elif defined(CACHE_STALL_STATS)
                RUN_PERF_HELPER(spmm_simple_vectorized_parallel, cache_stall_perf, cache_stall_counters, perf_event_set::CACHE_STALL_EVENTS, S_csr, Ti, Tk);
            #elif defined(DRAM_PRECHARGE_RD_STATS)
                RUN_DRAM_HELPER(spmm_simple_vectorized_parallel, dram_precharge_rd_perf, dram_precharge_rd_counters, perf_event_set::DRAM_PRECHARGE_RD_EVENTS, S_csr, Ti, Tk);
            #elif defined(DRAM_PRECHARGE_WR_STATS)
                RUN_DRAM_HELPER(spmm_simple_vectorized_parallel, dram_precharge_wr_perf, dram_precharge_wr_counters, perf_event_set::DRAM_PRECHARGE_WR_EVENTS, S_csr, Ti, Tk);
            #elif defined(DRAM_ACTIVATE_RD_STATS)
                RUN_DRAM_HELPER(spmm_simple_vectorized_parallel, dram_activate_rd_perf, dram_activate_rd_counters, perf_event_set::DRAM_ACTIVATE_RD_EVENTS, S_csr, Ti, Tk);
            #elif defined(DRAM_ACTIVATE_WR_STATS)
                RUN_DRAM_HELPER(spmm_simple_vectorized_parallel, dram_activate_wr_perf, dram_activate_wr_counters, perf_event_set::DRAM_ACTIVATE_WR_EVENTS, S_csr, Ti, Tk);
            #endif
        #else
            {
                RUN_TIMING(spmm_simple_vectorized_parallel, S_csr, Ti, Tk);
                MEM_RESET(C, countC, layers);
            }
            #ifndef TRACK_CACHE_ONLY
                {
                    RUN_PERF_HELPER(spmm_simple_vectorized_parallel, reg_perf, reg_counters, perf_event_set::REG_EVENTS, S_csr, Ti, Tk);
                }
                MEM_RESET(C, countC, layers);
                {
                    RUN_PERF_HELPER(spmm_simple_vectorized_parallel, cache_perf, cache_counters, perf_event_set::CACHE_EVENTS, S_csr, Ti, Tk);
                }
                MEM_RESET(C, countC, layers);
                {
                    RUN_PERF_HELPER(spmm_simple_vectorized_parallel, stall_perf, stall_counters, perf_event_set::STALL_EVENTS, S_csr, Ti, Tk);
                }
                MEM_RESET(C, countC, layers);
                {
                    RUN_DRAM_HELPER(spmm_simple_vectorized_parallel, dram_perf, dram_counters, perf_event_set::DRAM_EVENTS, S_csr, Ti, Tk);
                }
            #else // TRACK_CACHE_ONLY
                {
                    RUN_PERF_HELPER(spmm_simple_vectorized_parallel, cache_only_perf, cache_only_counters, perf_event_set::CACHE_ONLY_EVENTS, S_csr, Ti, Tk);
                }
            #endif
        #endif // TRACK_STATS_NEW
    #endif


    #if defined(RUN_CSR_JSTREAM_COMPILER_EXPR)
        RUN_WARMUP_EXPR(spmm_csr_jstream_compiler_vectorized, S_csr, Ti, Tk);
        MEM_RESET(C, countC, layers);

        #if defined(TRACK_STATS_NEW)
            #if defined(REG_STATS)
                RUN_PERF_HELPER(spmm_csr_jstream_compiler_vectorized, reg_perf, reg_counters, perf_event_set::REG_EVENTS, S_csr, Ti, Tk);
            #elif defined(CACHE_STATS)
                RUN_PERF_HELPER(spmm_csr_jstream_compiler_vectorized, cache_perf, cache_counters, perf_event_set::CACHE_EVENTS, S_csr, Ti, Tk);
            #elif defined(STALL_STATS)
                RUN_PERF_HELPER(spmm_csr_jstream_compiler_vectorized, stall_perf, stall_counters, perf_event_set::STALL_EVENTS, S_csr, Ti, Tk);
            #endif
        #else
            RUN_TIMING(spmm_csr_jstream_compiler_vectorized, S_csr, Ti, Tk);
        #endif
    #endif // RUN_CSR_JSTREAM_COMPILER_EXPR

    #if defined(RUN_DCSC_JSTREAM_COMPILER_EXPR)
        RUN_WARMUP_EXPR(spmm_dcsc_compiler_vectorized, S_dcsc, Ti, Tk);
        MEM_RESET(C, countC, layers);
        RUN_TIMING(spmm_dcsc_compiler_vectorized, S_dcsc, Ti, Tk);
    #endif // RUN_DCSC_JSTREAM_COMPILER_EXPR



    // Reset all the stat objects
    // {
    //     #ifndef TRACK_CACHE_ONLY
    //         RUN_PERF_RESET(reg_counters, perf::NUM_EVENTS_PER_SET);
    //         RUN_PERF_RESET(cache_counters, perf::NUM_EVENTS_PER_SET);
    //         RUN_PERF_RESET(stall_counters, perf::NUM_EVENTS_PER_SET);
    //         RUN_DRAM_RESET(dram_counters);
    //     #else
    //         RUN_PERF_RESET(cache_only_counters, perf::NUM_EVENTS_PER_SET);
    //     #endif
    // }

    #if defined(RUN_DCSC_JSTREAM_EXPR)
        PRINT_TYPE("DCSC_JSTREAM");

    #ifdef RUN_CORRECTNESS_CHECK
        {
            T *check_O = (T *) std::aligned_alloc( ALLOC_ALIGNMENT,  nrows * (feature + PADDING_C) * sizeof(T) );
            T *O_check = (T *) std::aligned_alloc( ALLOC_ALIGNMENT,  nrows * (feature + PADDING_C) * sizeof(T) );

            reset_matrix_helper<T, ITYPE>( check_O, (nrows * (feature + PADDING_C)) );
            reset_matrix_helper<T, ITYPE>( O_check, (nrows * (feature + PADDING_C)) );
            // reset_matrix_helper<T>( O, (nrows * feature) );
            // MEM_RESET(C, countC, layers);

            // f<T, ITYPE>( *A, B, O, feature, Ti, Tk, chunk_size );

            spmm_dcsc_jstream(*S_dcsc, B[0], O_check, feature, Ti, Tk );
            std::cout << "Is correct? " << check_simple( *S_csr, B[0], check_O, feature, O_check ) << std::endl;

            std::free(check_O);
            std::free(O_check);
        }
    #endif


        // RUN_CHECK(spmm_dcsc_jstream, S_dcsc, Ti, Tk);

        MEM_RESET(C, countC, layers);

        #ifdef TRACK_PER_CORE_RUNTIME
            #ifdef TRACK_PER_PANEL_RUNTIME
                RUN_PER_PANEL_WARMUP_EXPR(spmm_dcsc_jstream, S_dcsc, Ti, Tk);
            #else
                RUN_PER_CORE_WARMUP_EXPR(spmm_dcsc_jstream, S_dcsc, Ti, Tk);
            #endif
        #else
            RUN_WARMUP_EXPR(spmm_dcsc_jstream, S_dcsc, Ti, Tk);
        #endif

        MEM_RESET(C, countC, layers);

        #if defined(TRACK_STATS_NEW)
            #if defined(REG_STATS)
                RUN_PERF_HELPER(spmm_dcsc_jstream, reg_perf, reg_counters, perf_event_set::REG_EVENTS, S_dcsc, Ti, Tk);
            #elif defined(CACHE_STATS)
                RUN_PERF_HELPER(spmm_dcsc_jstream, cache_perf, cache_counters, perf_event_set::CACHE_EVENTS, S_dcsc, Ti, Tk);
            #elif defined(STALL_STATS)
                RUN_PERF_HELPER(spmm_dcsc_jstream, stall_perf, stall_counters, perf_event_set::STALL_EVENTS, S_dcsc, Ti, Tk);
            #elif defined(DRAM_STATS)
                RUN_DRAM_HELPER(spmm_dcsc_jstream, dram_perf, dram_counters, perf_event_set::DRAM_EVENTS, S_dcsc, Ti, Tk);
            #elif defined(CACHE_STALL_STATS)
                RUN_PERF_HELPER(spmm_dcsc_jstream, cache_stall_perf, cache_stall_counters, perf_event_set::CACHE_STALL_EVENTS, S_dcsc, Ti, Tk);
            #elif defined(L3_CACHE_STATS)
                RUN_PERF_HELPER(spmm_dcsc_jstream, l3_cache_perf, l3_cache_counters, perf_event_set::L3_CACHE_EVENTS, S_dcsc, Ti, Tk);
            #elif defined(DRAM_PRECHARGE_RD_STATS)
                RUN_DRAM_HELPER(spmm_dcsc_jstream, dram_precharge_rd_perf, dram_precharge_rd_counters, perf_event_set::DRAM_PRECHARGE_RD_EVENTS, S_dcsc, Ti, Tk);
            #elif defined(DRAM_PRECHARGE_WR_STATS)
                RUN_DRAM_HELPER(spmm_dcsc_jstream, dram_precharge_wr_perf, dram_precharge_wr_counters, perf_event_set::DRAM_PRECHARGE_WR_EVENTS, S_dcsc, Ti, Tk);
            #elif defined(DRAM_ACTIVATE_RD_STATS)
                RUN_DRAM_HELPER(spmm_dcsc_jstream, dram_activate_rd_perf, dram_activate_rd_counters, perf_event_set::DRAM_ACTIVATE_RD_EVENTS, S_dcsc, Ti, Tk);
            #elif defined(DRAM_ACTIVATE_WR_STATS)
                RUN_DRAM_HELPER(spmm_dcsc_jstream, dram_activate_wr_perf, dram_activate_wr_counters, perf_event_set::DRAM_ACTIVATE_WR_EVENTS, S_dcsc, Ti, Tk);
            #endif
        #else
            {

                #ifdef TRACK_PER_CORE_RUNTIME
                    #ifdef TRACK_PER_PANEL_RUNTIME
                        RUN_PER_PANEL_TIMING(spmm_dcsc_jstream, S_dcsc, Ti, Tk);
                    #else
                        RUN_PER_CORE_TIMING(spmm_dcsc_jstream, S_dcsc, Ti, Tk);
                    #endif
                #else
                    RUN_TIMING(spmm_dcsc_jstream, S_dcsc, Ti, Tk);
                #endif
                MEM_RESET(C, countC, layers);
            }
            #ifndef TRACK_CACHE_ONLY
                {
                    RUN_PERF_HELPER(spmm_dcsc_jstream, reg_perf, reg_counters, perf_event_set::REG_EVENTS, S_dcsc, Ti, Tk);
                }
                MEM_RESET(C, countC, layers);
                {
                    RUN_PERF_HELPER(spmm_dcsc_jstream, cache_perf, cache_counters, perf_event_set::CACHE_EVENTS, S_dcsc, Ti, Tk);
                }
                MEM_RESET(C, countC, layers);
                {
                    RUN_PERF_HELPER(spmm_dcsc_jstream, stall_perf, stall_counters, perf_event_set::STALL_EVENTS, S_dcsc, Ti, Tk);
                }
                MEM_RESET(C, countC, layers);
                {
                    RUN_DRAM_HELPER(spmm_dcsc_jstream, dram_perf, dram_counters, perf_event_set::DRAM_EVENTS, S_dcsc, Ti, Tk);
                }
            #else
                {
                    RUN_PERF_HELPER(spmm_dcsc_jstream, cache_only_perf, cache_only_counters, perf_event_set::CACHE_ONLY_EVENTS, S_dcsc, Ti, Tk);
                }
            #endif
        #endif // TRACK_STATS_NEW
    #endif // RUN_DCSC_JSTREAM_EXPR


    // Reset all the stat objects
    // {
    //     #ifndef TRACK_CACHE_ONLY
    //         RUN_PERF_RESET(reg_counters, perf::NUM_EVENTS_PER_SET);
    //         RUN_PERF_RESET(cache_counters, perf::NUM_EVENTS_PER_SET);
    //         RUN_PERF_RESET(stall_counters, perf::NUM_EVENTS_PER_SET);
    //         RUN_DRAM_RESET(dram_counters);
    //     #else
    //         RUN_PERF_RESET(cache_only_counters, perf::NUM_EVENTS_PER_SET);
    //     #endif
    // }


    #if defined(RUN_CSR_JSTREAM_EXPR)
        PRINT_TYPE("CSR_JSTREAM");

        RUN_CHECK(spmm_csr_jstream, S_csr, Ti, Tk);

        MEM_RESET(C, countC, layers);
        #ifdef TRACK_PER_CORE_RUNTIME
            RUN_PER_CORE_WARMUP_EXPR(spmm_csr_jstream, S_csr, Ti, Tk);
        #else
            RUN_WARMUP_EXPR(spmm_csr_jstream_hand_optimized, S_csr, Ti, Tk);
        #endif
        MEM_RESET(C, countC, layers);
        #if defined(TRACK_STATS_NEW)
            #if defined(REG_STATS)
                RUN_PERF_HELPER(spmm_csr_jstream, reg_perf, reg_counters, perf_event_set::REG_EVENTS, S_csr, Ti, Tk);
            #elif defined(CACHE_STATS)
                RUN_PERF_HELPER(spmm_csr_jstream_hand_optimized, cache_perf, cache_counters, perf_event_set::CACHE_EVENTS, S_csr, Ti, Tk);
            #elif defined(STALL_STATS)
                RUN_PERF_HELPER(spmm_csr_jstream, stall_perf, stall_counters, perf_event_set::STALL_EVENTS, S_csr, Ti, Tk);
            #elif defined(DRAM_STATS)
                RUN_DRAM_HELPER(spmm_csr_jstream, dram_perf, dram_counters, perf_event_set::DRAM_EVENTS, S_csr, Ti, Tk);
            #elif defined(L3_CACHE_STATS)
                RUN_PERF_HELPER(spmm_csr_jstream, l3_cache_perf, l3_cache_counters, perf_event_set::L3_CACHE_EVENTS, S_csr, Ti, Tk);
            #elif defined(CACHE_STALL_STATS)
                RUN_PERF_HELPER(spmm_csr_jstream, cache_stall_perf, cache_stall_counters, perf_event_set::CACHE_STALL_EVENTS, S_csr, Ti, Tk);
            #elif defined(TLB_STATS)
                RUN_PERF_HELPER(spmm_csr_jstream, tlb_perf, tlb_counters, perf_event_set::TLB_EVENTS, S_csr, Ti, Tk);
            #elif defined(DRAM_PRECHARGE_RD_STATS)
                RUN_DRAM_HELPER(spmm_csr_jstream, dram_precharge_rd_perf, dram_precharge_rd_counters, perf_event_set::DRAM_PRECHARGE_RD_EVENTS, S_csr, Ti, Tk);
            #elif defined(DRAM_PRECHARGE_WR_STATS)
                RUN_DRAM_HELPER(spmm_csr_jstream, dram_precharge_wr_perf, dram_precharge_wr_counters, perf_event_set::DRAM_PRECHARGE_WR_EVENTS, S_csr, Ti, Tk);
            #elif defined(DRAM_ACTIVATE_RD_STATS)
                RUN_DRAM_HELPER(spmm_csr_jstream, dram_activate_rd_perf, dram_activate_rd_counters, perf_event_set::DRAM_ACTIVATE_RD_EVENTS, S_csr, Ti, Tk);
            #elif defined(DRAM_ACTIVATE_WR_STATS)
                RUN_DRAM_HELPER(spmm_csr_jstream, dram_activate_wr_perf, dram_activate_wr_counters, perf_event_set::DRAM_ACTIVATE_WR_EVENTS, S_csr, Ti, Tk);
            #endif
        #else
            {
                #ifdef TRACK_PER_CORE_RUNTIME
                    RUN_PER_PANEL_TIMING(spmm_csr_jstream, S_csr, Ti, Tk);
                #else
                    RUN_TIMING(spmm_csr_jstream, S_csr, Ti, Tk);
                #endif
                MEM_RESET(C, countC, layers);
            }
            #ifndef TRACK_CACHE_ONLY
                {
                    RUN_PERF_HELPER(spmm_csr_jstream, reg_perf, reg_counters, perf_event_set::REG_EVENTS, S_csr, Ti, Tk);
                }
                MEM_RESET(C, countC, layers);
                {
                    RUN_PERF_HELPER(spmm_csr_jstream, cache_perf, cache_counters, perf_event_set::CACHE_EVENTS, S_csr, Ti, Tk);
                }
                MEM_RESET(C, countC, layers);
                {
                    RUN_PERF_HELPER(spmm_csr_jstream, stall_perf, stall_counters, perf_event_set::STALL_EVENTS, S_csr, Ti, Tk);
                }
                MEM_RESET(C, countC, layers);
                {
                    RUN_DRAM_HELPER(spmm_csr_jstream, dram_perf, dram_counters, perf_event_set::DRAM_EVENTS, S_csr, Ti, Tk);
                }
            #else
                {
                    RUN_PERF_HELPER(spmm_csr_jstream, cache_only_perf, cache_only_counters, perf_event_set::CACHE_ONLY_EVENTS, S_csr, Ti, Tk);
                }
            #endif
        #endif // TRACK_STATS_NEW
    #endif // RUN_CSR_JSTREAM_EXPR


    // Reset all the stat objects
    // {
    //     #ifndef TRACK_CACHE_ONLY
    //         RUN_PERF_RESET(reg_counters, perf::NUM_EVENTS_PER_SET);
    //         RUN_PERF_RESET(cache_counters, perf::NUM_EVENTS_PER_SET);
    //         RUN_PERF_RESET(stall_counters, perf::NUM_EVENTS_PER_SET);
    //         RUN_DRAM_RESET(dram_counters);
    //     #else
    //         RUN_PERF_RESET(cache_only_counters, perf::NUM_EVENTS_PER_SET);
    //     #endif
    // }

    #if defined(RUN_CSR_KSTREAM_EXPR)
        PRINT_TYPE("CSR_KSTREAM");

        RUN_CHECK(spmm_kstream_compiler_vectorized, S_stm, Ti, Tj);

        MEM_RESET(C, countC, layers);
        #ifdef TRACK_PER_CORE_RUNTIME
            #ifdef TRACK_PER_PANEL_RUNTIME
                RUN_PER_PANEL_WARMUP_EXPR(spmm_kstream_compiler_vectorized, S_stm, Ti, Tj);
            #else
                RUN_PER_CORE_WARMUP_EXPR(spmm_kstream_compiler_vectorized, S_stm, Ti, Tj);
            #endif
        #else
            RUN_WARMUP_EXPR(spmm_kstream_compiler_vectorized, S_stm, Ti, Tj);
        #endif

        MEM_RESET(C, countC, layers);

        #if defined(TRACK_STATS_NEW)
            #if defined(REG_STATS)
                RUN_PERF_HELPER(spmm_kstream_compiler_vectorized, reg_perf, reg_counters, perf_event_set::REG_EVENTS, S_stm, Ti, Tj);
            #elif defined(CACHE_STATS)
                RUN_PERF_HELPER(spmm_kstream_compiler_vectorized, cache_perf, cache_counters, perf_event_set::CACHE_EVENTS, S_stm, Ti, Tj);
            #elif defined(STALL_STATS)
                RUN_PERF_HELPER(spmm_kstream_compiler_vectorized, stall_perf, stall_counters, perf_event_set::STALL_EVENTS, S_stm, Ti, Tj);
            #elif defined(DRAM_STATS)
                RUN_DRAM_HELPER(spmm_kstream_compiler_vectorized, dram_perf, dram_counters, perf_event_set::DRAM_EVENTS, S_stm, Ti, Tj);
            #elif defined(CACHE_STALL_STATS)
                RUN_PERF_HELPER(spmm_kstream_compiler_vectorized, cache_stall_perf, cache_stall_counters, perf_event_set::CACHE_STALL_EVENTS, S_stm, Ti, Tj);
            #elif defined(DRAM_PRECHARGE_RD_STATS)
                RUN_DRAM_HELPER(spmm_kstream_compiler_vectorized, dram_precharge_rd_perf, dram_precharge_rd_counters, perf_event_set::DRAM_PRECHARGE_RD_EVENTS, S_stm, Ti, Tj);
            #elif defined(DRAM_PRECHARGE_WR_STATS)
                RUN_DRAM_HELPER(spmm_kstream_compiler_vectorized, dram_precharge_wr_perf, dram_precharge_wr_counters, perf_event_set::DRAM_PRECHARGE_WR_EVENTS, S_stm, Ti, Tj);
            #elif defined(DRAM_ACTIVATE_RD_STATS)
                RUN_DRAM_HELPER(spmm_kstream_compiler_vectorized, dram_activate_rd_perf, dram_activate_rd_counters, perf_event_set::DRAM_ACTIVATE_RD_EVENTS, S_stm, Ti, Tj);
            #elif defined(DRAM_ACTIVATE_WR_STATS)
                RUN_DRAM_HELPER(spmm_kstream_compiler_vectorized, dram_activate_wr_perf, dram_activate_wr_counters, perf_event_set::DRAM_ACTIVATE_WR_EVENTS, S_stm, Ti, Tj);
            #endif

        #else
            {
                #ifdef TRACK_PER_CORE_RUNTIME
                    #ifdef TRACK_PER_PANEL_RUNTIME
                        RUN_PER_PANEL_TIMING(spmm_kstream_compiler_vectorized, S_stm, Ti, Tj);
                    #else
                        RUN_PER_CORE_TIMING(spmm_kstream_compiler_vectorized, S_stm, Ti, Tj);
                    #endif
                #else
                    RUN_TIMING(spmm_kstream_compiler_vectorized, S_stm, Ti, Tj);
                #endif
                MEM_RESET(C, countC, layers);
            }
            #ifndef TRACK_CACHE_ONLY
                {
                    RUN_PERF_HELPER(spmm_kstream_compiler_vectorized, reg_perf, reg_counters, perf_event_set::REG_EVENTS, S_stm, Ti, Tj);
                }
                MEM_RESET(C, countC, layers);
                {
                    RUN_PERF_HELPER(spmm_kstream_compiler_vectorized, cache_perf, cache_counters, perf_event_set::CACHE_EVENTS, S_stm, Ti, Tj);
                }
                MEM_RESET(C, countC, layers);
                {
                    RUN_PERF_HELPER(spmm_kstream_compiler_vectorized, stall_perf, stall_counters, perf_event_set::STALL_EVENTS, S_stm, Ti, Tj);
                }
                MEM_RESET(C, countC, layers);
                {
                    RUN_DRAM_HELPER(spmm_kstream_compiler_vectorized, dram_perf, dram_counters, perf_event_set::DRAM_EVENTS, S_stm, Ti, Tj);
                }
            #else
                {
                    RUN_PERF_HELPER(spmm_kstream_compiler_vectorized, cache_only_perf, cache_only_counters, perf_event_set::CACHE_ONLY_EVENTS, S_stm, Ti, Tj);
                }
            #endif
        #endif // TRACK_STATS_NEW
    #endif // RUN_CSR_KSTREAM_EXPR







    #if defined(RUN_CSR_ATM_KSTREAM)
        PRINT_TYPE("CSR_ATM_KSTREAM");

        Tj = Tk;

        RUN_CHECK(spmm_atm_kstream_compiler_vectorized, S_atm, Ti, Tj);

        MEM_RESET(C, countC, layers);
        #ifdef TRACK_PER_CORE_RUNTIME
            #ifdef TRACK_PER_PANEL_RUNTIME
                RUN_PER_PANEL_WARMUP_EXPR(spmm_atm_kstream_compiler_vectorized, S_atm, Ti, Tj);
            #else
                RUN_PER_CORE_WARMUP_EXPR(spmm_atm_kstream_compiler_vectorized, S_atm, Ti, Tj);
            #endif
        #else
            RUN_WARMUP_EXPR(spmm_atm_kstream_compiler_vectorized, S_atm, Ti, Tj);
        #endif

        MEM_RESET(C, countC, layers);

        #if defined(TRACK_STATS_NEW)
            #if defined(REG_STATS)
                RUN_PERF_HELPER(spmm_atm_kstream_compiler_vectorized, reg_perf, reg_counters, perf_event_set::REG_EVENTS, S_atm, Ti, Tj);
            #elif defined(CACHE_STATS)
                RUN_PERF_HELPER(spmm_atm_kstream_compiler_vectorized, cache_perf, cache_counters, perf_event_set::CACHE_EVENTS, S_atm, Ti, Tj);
            #elif defined(STALL_STATS)
                RUN_PERF_HELPER(spmm_atm_kstream_compiler_vectorized, stall_perf, stall_counters, perf_event_set::NAMED_STALL_EVENTS, S_atm, Ti, Tj);
            #elif defined(DRAM_STATS)
                RUN_DRAM_HELPER(spmm_atm_kstream_compiler_vectorized, dram_perf, dram_counters, perf_event_set::DRAM_EVENTS, S_atm, Ti, Tj);
            #elif defined(L3_CACHE_STATS)
                RUN_PERF_HELPER(spmm_atm_kstream_compiler_vectorized, l3_cache_perf, l3_cache_counters, perf_event_set::L3_CACHE_EVENTS, S_atm, Ti, Tj);
            #elif defined(CACHE_STALL_STATS)
                RUN_PERF_HELPER(spmm_atm_kstream_compiler_vectorized, cache_stall_perf, cache_stall_counters, perf_event_set::CACHE_STALL_EVENTS, S_atm, Ti, Tj);
            #elif defined(DRAM_PRECHARGE_RD_STATS)
                RUN_DRAM_HELPER(spmm_atm_kstream_compiler_vectorized, dram_precharge_rd_perf, dram_precharge_rd_counters, perf_event_set::DRAM_PRECHARGE_RD_EVENTS, S_atm, Ti, Tj);
            #elif defined(DRAM_PRECHARGE_WR_STATS)
                RUN_DRAM_HELPER(spmm_atm_kstream_compiler_vectorized, dram_precharge_wr_perf, dram_precharge_wr_counters, perf_event_set::DRAM_PRECHARGE_WR_EVENTS, S_atm, Ti, Tj);
            #elif defined(DRAM_ACTIVATE_RD_STATS)
                RUN_DRAM_HELPER(spmm_atm_kstream_compiler_vectorized, dram_activate_rd_perf, dram_activate_rd_counters, perf_event_set::DRAM_ACTIVATE_RD_EVENTS, S_atm, Ti, Tj);
            #elif defined(DRAM_ACTIVATE_WR_STATS)
                RUN_DRAM_HELPER(spmm_atm_kstream_compiler_vectorized, dram_activate_wr_perf, dram_activate_wr_counters, perf_event_set::DRAM_ACTIVATE_WR_EVENTS, S_atm, Ti, Tj);
            #endif

        #else
            #ifdef TRACK_PER_CORE_RUNTIME
                #ifdef TRACK_PER_PANEL_RUNTIME
                    RUN_PER_PANEL_TIMING(spmm_atm_kstream_compiler_vectorized, S_atm, Ti, Tj);
                #else
                    RUN_PER_CORE_TIMING(spmm_atm_kstream_compiler_vectorized, S_atm, Ti, Tj);
                #endif
            #else
                RUN_TIMING(spmm_atm_kstream_compiler_vectorized, S_atm, Ti, Tj);
            #endif
            MEM_RESET(C, countC, layers);
        #endif // TRACK_STATS_NEW
    #endif // RUN_CSR_KSTREAM_EXPR





    #if defined(RUN_CSF_EXPR)
        PRINT_TYPE("CSF_KSTREAM");

        RUN_CHECK(spmm_csf_compiler_vectorized, S_csf, Ti, Tj);

        MEM_RESET(C, countC, layers);
        #ifdef TRACK_PER_CORE_RUNTIME
            #ifdef TRACK_PER_PANEL_RUNTIME
                RUN_PER_PANEL_WARMUP_EXPR(spmm_csf_compiler_vectorized, S_csf, Ti, Tj);
            #else
                RUN_PER_CORE_WARMUP_EXPR(spmm_csf_compiler_vectorized, S_csf, Ti, Tj);
            #endif
        #else
            RUN_WARMUP_EXPR(spmm_csf_compiler_vectorized, S_csf, Ti, Tj);
        #endif

        MEM_RESET(C, countC, layers);

        #if defined(TRACK_STATS_NEW)
            #if defined(REG_STATS)
                RUN_PERF_HELPER(spmm_csf_compiler_vectorized, reg_perf, reg_counters, perf_event_set::REG_EVENTS, S_csf, Ti, Tj);
            #elif defined(CACHE_STATS)
                RUN_PERF_HELPER(spmm_csf_compiler_vectorized, cache_perf, cache_counters, perf_event_set::CACHE_EVENTS, S_csf, Ti, Tj);
            #elif defined(STALL_STATS)
                RUN_PERF_HELPER(spmm_csf_compiler_vectorized, stall_perf, stall_counters, perf_event_set::NAMED_STALL_EVENTS, S_csf, Ti, Tj);
            #elif defined(DRAM_STATS)
                RUN_DRAM_HELPER(spmm_csf_compiler_vectorized, dram_perf, dram_counters, perf_event_set::DRAM_EVENTS, S_csf, Ti, Tj);
            #elif defined(L3_CACHE_STATS)
                RUN_PERF_HELPER(spmm_csf_compiler_vectorized, l3_cache_perf, l3_cache_counters, perf_event_set::L3_CACHE_EVENTS, S_csf, Ti, Tj);
            #elif defined(CACHE_STALL_STATS)
                RUN_PERF_HELPER(spmm_csf_compiler_vectorized, cache_stall_perf, cache_stall_counters, perf_event_set::CACHE_STALL_EVENTS, S_csf, Ti, Tj);
            #elif defined(DRAM_PRECHARGE_RD_STATS)
                RUN_DRAM_HELPER(spmm_csf_compiler_vectorized, dram_precharge_rd_perf, dram_precharge_rd_counters, perf_event_set::DRAM_PRECHARGE_RD_EVENTS, S_csf, Ti, Tj);
            #elif defined(DRAM_PRECHARGE_WR_STATS)
                RUN_DRAM_HELPER(spmm_csf_compiler_vectorized, dram_precharge_wr_perf, dram_precharge_wr_counters, perf_event_set::DRAM_PRECHARGE_WR_EVENTS, S_csf, Ti, Tj);
            #elif defined(DRAM_ACTIVATE_RD_STATS)
                RUN_DRAM_HELPER(spmm_csf_compiler_vectorized, dram_activate_rd_perf, dram_activate_rd_counters, perf_event_set::DRAM_ACTIVATE_RD_EVENTS, S_csf, Ti, Tj);
            #elif defined(DRAM_ACTIVATE_WR_STATS)
                RUN_DRAM_HELPER(spmm_csf_compiler_vectorized, dram_activate_wr_perf, dram_activate_wr_counters, perf_event_set::DRAM_ACTIVATE_WR_EVENTS, S_csf, Ti, Tj);
            #endif

        #else
            #ifdef TRACK_PER_CORE_RUNTIME
                #ifdef TRACK_PER_PANEL_RUNTIME
                    RUN_PER_PANEL_TIMING(spmm_csf_compiler_vectorized, S_csf, Ti, Tj);
                #else
                    RUN_PER_CORE_TIMING(spmm_csf_compiler_vectorized, S_csf, Ti, Tj);
                #endif
            #else
                RUN_TIMING(spmm_csf_compiler_vectorized, S_csf, Ti, Tj);
            #endif
            MEM_RESET(C, countC, layers);
        #endif // TRACK_STATS_NEW
    #endif



    #if defined(RUN_HYBRID_JSTREAM_EXPR)
        PRINT_TYPE("HYBRID_JSTREAM");

        // run worklist based workload

    #endif // RUN_HYBRID_JSTREAM_EXPR

    #if defined(RUN_SPLIT_JSTREAM_EXPR)
        PRINT_TYPE("SPLIT CSR JSTREAM");

        RUN_CHECK(spmm_split_csr_jstream, S_split, Ti, Tk);

        MEM_RESET(C, countC, layers);
        RUN_WARMUP_EXPR(spmm_split_csr_jstream, S_split, Ti, Tk);
        MEM_RESET(C, countC, layers);
        {
            RUN_TIMING(spmm_split_csr_jstream, S_split, Ti, Tk);
        }
    #endif


    #if defined (RUN_CSR_WORKLIST_EXPR)
        PRINT_TYPE("CSR ADAPTIVE PANEL EXPERIMENT");

        #ifdef RUN_CORRECTNESS_CHECK
        {
            T *check_O = (T *) std::aligned_alloc( ALLOC_ALIGNMENT,  nrows * (feature + PADDING_C) * sizeof(T) );
            T *O = (T *) std::aligned_alloc( ALLOC_ALIGNMENT,  nrows * (feature + PADDING_C) * sizeof(T) );
            std::memset( check_O, 0, sizeof(T) * (nrows * feature) );
            std::memset( O, 0, sizeof(T) * (nrows * feature) );

            spmm_csr_worklist( *S_csr, B[0], O, feature, pairs_worklist, num_panels_per_thread, chunk_size);

            // spmm_csr_jstream_worklist(*S_csr, B, O, feature, num_panels_per_thread, work_list, num_threads, chunk_size);
            std::cout << "Is correct? " << check_simple( *S_csr, B[0], check_O, feature, O ) << std::endl;

            std::free(check_O);
            std::free(O);
        }
        #endif // RUN_CORRECTNESS_CHECK


        MEM_RESET(C, countC, layers);
        #ifdef TRACK_PER_CORE_RUNTIME
            #ifdef TRACK_PER_PANEL_RUNTIME
                RUN_HYBRID_PER_PANEL_WARMUP_EXPR( spmm_csr_worklist , S_csr, pairs_worklist, num_panels_per_thread );
            #else
                RUN_HYBRID_PER_CORE_WARMUP_EXPR( spmm_csr_worklist, S_csr, pairs_worklist, num_panels_per_thread );
            #endif
        #else
            RUN_HYBRID_WARMUP_EXPR( spmm_csr_worklist, S_csr, pairs_worklist, num_panels_per_thread );
        #endif
        MEM_RESET(C, countC, layers);
        #if defined(TRACK_STATS_NEW)
            #if defined(REG_STATS)
                RUN_HYBRID_PERF_HELPER(spmm_csr_worklist, reg_perf, reg_counters, perf_event_set::REG_EVENTS, S_csr, pairs_worklist, num_panels_per_thread );
            #elif defined(CACHE_STATS)
                RUN_HYBRID_PERF_HELPER(spmm_csr_worklist, cache_perf, cache_counters, perf_event_set::CACHE_EVENTS, S_csr, pairs_worklist, num_panels_per_thread );
            #elif defined(STALL_STATS)
                RUN_HYBRID_PERF_HELPER(spmm_csr_worklist, stall_perf, stall_counters, perf_event_set::NAMED_STALL_EVENTS, S_csr, pairs_worklist, num_panels_per_thread );
            #elif defined(DRAM_STATS)
                RUN_HYBRID_DRAM_HELPER(spmm_csr_worklist, dram_perf, dram_counters, perf_event_set::DRAM_EVENTS, S_csr, pairs_worklist, num_panels_per_thread );
            #elif defined(CACHE_STALL_STATS)
                RUN_HYBRID_PERF_HELPER(spmm_csr_worklist, cache_stall_perf, cache_stall_counters, perf_event_set::CACHE_STALL_EVENTS, S_csr, pairs_worklist, num_panels_per_thread );
            #elif defined(DRAM_PRECHARGE_RD_STATS)
                RUN_HYBRID_DRAM_HELPER(spmm_csr_worklist, dram_precharge_rd_perf, dram_precharge_rd_counters, perf_event_set::DRAM_PRECHARGE_RD_EVENTS, S_csr, pairs_worklist, num_panels_per_thread );
            #elif defined(DRAM_PRECHARGE_WR_STATS)
                RUN_HYBRID_DRAM_HELPER(spmm_csr_worklist, dram_precharge_wr_perf, dram_precharge_wr_counters, perf_event_set::DRAM_PRECHARGE_WR_EVENTS, S_csr, pairs_worklist, num_panels_per_thread );
            #elif defined(DRAM_ACTIVATE_RD_STATS)
                RUN_HYBRID_DRAM_HELPER(spmm_csr_worklist, dram_activate_rd_perf, dram_activate_rd_counters, perf_event_set::DRAM_ACTIVATE_RD_EVENTS, S_csr, pairs_worklist, num_panels_per_thread );
            #elif defined(DRAM_ACTIVATE_WR_STATS)
                RUN_HYBRID_DRAM_HELPER(spmm_csr_worklist, dram_activate_wr_perf, dram_activate_wr_counters, perf_event_set::DRAM_ACTIVATE_WR_EVENTS, S_csr, pairs_worklist, num_panels_per_thread );
            #elif defined(TLB_STATS)
                RUN_HYBRID_PERF_HELPER(spmm_csr_worklist, tlb_perf, tlb_counters, perf_event_set::TLB_EVENTS, S_csr, pairs_worklist, num_panels_per_thread );
            #endif

        #else
            #ifdef TRACK_PER_CORE_RUNTIME
                #ifdef TRACK_PER_PANEL_RUNTIME
                    RUN_HYBRID_PER_PANEL_TIMING( spmm_csr_worklist, S_csr, pairs_worklist, num_panels_per_thread );
                #else
                    RUN_HYBRID_PER_CORE_TIMING( spmm_cscr_worklist, S_csr, pairs_worklist, num_panels_per_thread );
                #endif
            #else
                RUN_HYBRID_TIMING( spmm_csr_worklist, S_csr, pairs_worklist, num_panels_per_thread );
            #endif
        #endif


    #endif

    #if defined(RUN_HYB_EXPR)
        PRINT_TYPE("HYBRID PANEL EXPERIMENT");

        MEM_RESET(C, countC, layers);
        #ifdef TRACK_PER_CORE_RUNTIME
            #ifdef TRACK_PER_PANEL_RUNTIME
                RUN_HYBRID_PER_PANEL_WARMUP_EXPR( spmm_hybrid_worklist, S_dcsh, pairs_worklist, num_panels_per_thread );
            #else
                RUN_HYBRID_PER_CORE_WARMUP_EXPR( spmm_hybrid_worklist, S_dcsh, pairs_worklist, num_panels_per_thread );
            #endif
        #else
            RUN_HYBRID_WARMUP_EXPR( spmm_hybrid_worklist, S_dcsh, pairs_worklist, num_panels_per_thread );
        #endif
        MEM_RESET(C, countC, layers);
        #if defined(TRACK_STATS_NEW)
            #if defined(REG_STATS)
                RUN_HYBRID_PERF_HELPER(spmm_hybrid_worklist, reg_perf, reg_counters, perf_event_set::REG_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(CACHE_STATS)
                RUN_HYBRID_PERF_HELPER(spmm_hybrid_worklist, cache_perf, cache_counters, perf_event_set::CACHE_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(STALL_STATS)
                RUN_HYBRID_PERF_HELPER(spmm_hybrid_worklist, stall_perf, stall_counters, perf_event_set::NAMED_STALL_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(DRAM_STATS)
                RUN_HYBRID_DRAM_HELPER(spmm_hybrid_worklist, dram_perf, dram_counters, perf_event_set::DRAM_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(CACHE_STALL_STATS)
                RUN_HYBRID_PERF_HELPER(spmm_hybrid_worklist, cache_stall_perf, cache_stall_counters, perf_event_set::CACHE_STALL_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(DRAM_PRECHARGE_RD_STATS)
                RUN_HYBRID_DRAM_HELPER(spmm_hybrid_worklist, dram_precharge_rd_perf, dram_precharge_rd_counters, perf_event_set::DRAM_PRECHARGE_RD_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(DRAM_PRECHARGE_WR_STATS)
                RUN_HYBRID_DRAM_HELPER(spmm_hybrid_worklist, dram_precharge_wr_perf, dram_precharge_wr_counters, perf_event_set::DRAM_PRECHARGE_WR_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(DRAM_ACTIVATE_RD_STATS)
                RUN_HYBRID_DRAM_HELPER(spmm_hybrid_worklist, dram_activate_rd_perf, dram_activate_rd_counters, perf_event_set::DRAM_ACTIVATE_RD_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(DRAM_ACTIVATE_WR_STATS)
                RUN_HYBRID_DRAM_HELPER(spmm_hybrid_worklist, dram_activate_wr_perf, dram_activate_wr_counters, perf_event_set::DRAM_ACTIVATE_WR_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(TLB_STATS)
                RUN_HYBRID_PERF_HELPER(spmm_hybrid_worklist, tlb_perf, tlb_counters, perf_event_set::TLB_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #endif

        #else
            #ifdef TRACK_PER_CORE_RUNTIME
                #ifdef TRACK_PER_PANEL_RUNTIME
                    RUN_HYBRID_PER_PANEL_TIMING( spmm_hybrid_worklist, S_dcsh, pairs_worklist, num_panels_per_thread );
                #else
                    RUN_HYBRID_PER_CORE_TIMING( spmm_hybrid_worklist, S_dcsh, pairs_worklist, num_panels_per_thread );
                #endif
            #else
                RUN_HYBRID_TIMING( spmm_hybrid_worklist, S_dcsh, pairs_worklist, num_panels_per_thread );
            #endif
        #endif

    #endif // RUN_HYB_EXPR

    #if defined(RUN_DYN_HYB_EXPR)
        PRINT_TYPE("DYNAMIC HYBRID PANEL EXPERIMENT");

        #ifdef RUN_CORRECTNESS_CHECK
        {
            T *dcsh_output = (T *) std::aligned_alloc( ALLOC_ALIGNMENT, S_dcsh->nrows * (feature + PADDING_C) * sizeof(T) );
            T *simple_output = (T *) std::aligned_alloc( ALLOC_ALIGNMENT, S_dcsh->nrows * (feature + PADDING_C) * sizeof(T) );

            #pragma omp parallel for
            for ( ITYPE i = 0; i < (S_dcsh->nrows * (feature + PADDING_C)); i++ ) {
                dcsh_output[i] = 0;
                simple_output[i] = 0;
            }

            spmm_hybrid_worklist_dynamic( *S_dcsh, B[0], dcsh_output, feature, pairs_worklist, num_panels_per_thread, chunk_size);

            bool is_correct = check_simple( *S_csr, B[0], simple_output, feature, dcsh_output );

            // spmm_csr_jstream_worklist(*S_csr, B, O, feature, num_panels_per_thread, work_list, num_threads, chunk_size);
            std::cout << "Is correct? " << is_correct << std::endl;
            if ( !is_correct ) {
                std::cout << "FAILURE MATRIX: " << mtx_filename << std::endl;
                std::exit(EXIT_FAILURE);
            }

            std::free(simple_output);
            std::free(dcsh_output);
        }
        #endif // RUN_CORRECTNESS_CHECK

        MEM_RESET(C, countC, layers);
        #ifdef TRACK_PER_CORE_RUNTIME
            #ifdef TRACK_PER_PANEL_RUNTIME
                RUN_HYBRID_PER_PANEL_WARMUP_EXPR( spmm_hybrid_worklist_dynamic, S_dcsh, pairs_worklist, num_panels_per_thread );
            #else
                RUN_HYBRID_PER_CORE_WARMUP_EXPR( spmm_hybrid_worklist_dynamic, S_dcsh, pairs_worklist, num_panels_per_thread );
            #endif
        #else
            RUN_HYBRID_WARMUP_EXPR( spmm_hybrid_worklist_dynamic, S_dcsh, pairs_worklist, num_panels_per_thread );
        #endif
        MEM_RESET(C, countC, layers);
        #if defined(TRACK_STATS_NEW)
            #if defined(REG_STATS)
                RUN_HYBRID_PERF_HELPER(spmm_hybrid_worklist_dynamic, reg_perf, reg_counters, perf_event_set::REG_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(CACHE_STATS)
                RUN_HYBRID_PERF_HELPER(spmm_hybrid_worklist_dynamic, cache_perf, cache_counters, perf_event_set::CACHE_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread  );
            #elif defined(STALL_STATS)
                RUN_HYBRID_PERF_HELPER(spmm_hybrid_worklist_dynamic, stall_perf, stall_counters, perf_event_set::NAMED_STALL_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(L3_CACHE_STATS)
                RUN_HYBRID_PERF_HELPER(spmm_hybrid_worklist_dynamic, l3_cache_perf, l3_cache_counters, perf_event_set::L3_CACHE_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(FILL_STATS)
                RUN_HYBRID_PERF_HELPER(spmm_hybrid_worklist_dynamic, fill_perf, fill_counters, perf_event_set::FILL_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(DRAM_STATS)
                RUN_HYBRID_DRAM_HELPER(spmm_hybrid_worklist_dynamic, dram_perf, dram_counters, perf_event_set::DRAM_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(CACHE_STALL_STATS)
                RUN_HYBRID_PERF_HELPER(spmm_hybrid_worklist_dynamic, cache_stall_perf, cache_stall_counters, perf_event_set::CACHE_STALL_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(DRAM_PRECHARGE_RD_STATS)
                RUN_HYBRID_DRAM_HELPER(spmm_hybrid_worklist_dynamic, dram_precharge_rd_perf, dram_precharge_rd_counters, perf_event_set::DRAM_PRECHARGE_RD_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(DRAM_PRECHARGE_WR_STATS)
                RUN_HYBRID_DRAM_HELPER(spmm_hybrid_worklist_dynamic, dram_precharge_wr_perf, dram_precharge_wr_counters, perf_event_set::DRAM_PRECHARGE_WR_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(DRAM_ACTIVATE_RD_STATS)
                RUN_HYBRID_DRAM_HELPER(spmm_hybrid_worklist_dynamic, dram_activate_rd_perf, dram_activate_rd_counters, perf_event_set::DRAM_ACTIVATE_RD_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(DRAM_ACTIVATE_WR_STATS)
                RUN_HYBRID_DRAM_HELPER(spmm_hybrid_worklist_dynamic, dram_activate_wr_perf, dram_activate_wr_counters, perf_event_set::DRAM_ACTIVATE_WR_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(TLB_STATS)
                RUN_HYBRID_PERF_HELPER(spmm_hybrid_worklist_dynamic, tlb_perf, tlb_counters, perf_event_set::TLB_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #endif

        #else
            #ifdef TRACK_PER_CORE_RUNTIME
                #ifdef TRACK_PER_PANEL_RUNTIME
                    RUN_HYBRID_PER_PANEL_TIMING( spmm_hybrid_worklist_dynamic, S_dcsh, pairs_worklist, num_panels_per_thread );
                #else
                    RUN_HYBRID_PER_CORE_TIMING( spmm_hybrid_worklist_dynamic, S_dcsh, pairs_worklist, num_panels_per_thread );
                #endif
            #else
                RUN_HYBRID_TIMING( spmm_hybrid_worklist_dynamic, S_dcsh, pairs_worklist, num_panels_per_thread );
            #endif
        #endif

    #endif // RUN_DYN_HYB_EXPR



    #if defined(RUN_DYN_HYB_COMP_VEC_EXPR)
        PRINT_TYPE("DYNAMIC HYBRID PANEL COMPILER VECTORIZED");

        #ifdef RUN_CORRECTNESS_CHECK
        {
            T *check_O = (T *) std::aligned_alloc( ALLOC_ALIGNMENT,  nrows * (feature + PADDING_C) * sizeof(T) );
            T *O = (T *) std::aligned_alloc( ALLOC_ALIGNMENT,  nrows * (feature + PADDING_C) * sizeof(T) );
            std::memset( check_O, 0, sizeof(T) * (nrows * feature) );
            std::memset( O, 0, sizeof(T) * (nrows * feature) );

            spmm_hybrid_worklist_dynamic_compiler_vectorized( *S_dcsh, B[0], O, feature, pairs_worklist, num_panels_per_thread, chunk_size);

            // spmm_csr_jstream_worklist(*S_csr, B, O, feature, num_panels_per_thread, work_list, num_threads, chunk_size);
            std::cout << "Is correct? " << check_simple( *S_csr, B[0], check_O, feature, O ) << std::endl;

            std::free(check_O);
            std::free(O);
        }
        #endif // RUN_CORRECTNESS_CHECK

        MEM_RESET(C, countC, layers);
        #ifdef TRACK_PER_CORE_RUNTIME
            #ifdef TRACK_PER_PANEL_RUNTIME
                RUN_HYBRID_PER_PANEL_WARMUP_EXPR( spmm_hybrid_worklist_dynamic_compiler_vectorized, S_dcsh, pairs_worklist, num_panels_per_thread );
            #else
                RUN_HYBRID_PER_CORE_WARMUP_EXPR( spmm_hybrid_worklist_dynamic_compiler_vectorized, S_dcsh, pairs_worklist, num_panels_per_thread );
            #endif
        #else
            RUN_HYBRID_WARMUP_EXPR( spmm_hybrid_worklist_dynamic_compiler_vectorized, S_dcsh, pairs_worklist, num_panels_per_thread );
        #endif
        MEM_RESET(C, countC, layers);
        #if defined(TRACK_STATS_NEW)
            #if defined(REG_STATS)
                RUN_HYBRID_PERF_HELPER(spmm_hybrid_worklist_dynamic_compiler_vectorized, reg_perf, reg_counters, perf_event_set::REG_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(CACHE_STATS)
                RUN_HYBRID_PERF_HELPER(spmm_hybrid_worklist_dynamic_compiler_vectorized, cache_perf, cache_counters, perf_event_set::CACHE_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread  );
            #elif defined(STALL_STATS)
                RUN_HYBRID_PERF_HELPER(spmm_hybrid_worklist_dynamic_compiler_vectorized, stall_perf, stall_counters, perf_event_set::NAMED_STALL_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(DRAM_STATS)
                RUN_HYBRID_DRAM_HELPER(spmm_hybrid_worklist_dynamic_compiler_vectorized, dram_perf, dram_counters, perf_event_set::DRAM_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(CACHE_STALL_STATS)
                RUN_HYBRID_PERF_HELPER(spmm_hybrid_worklist_dynamic_compiler_vectorized, cache_stall_perf, cache_stall_counters, perf_event_set::CACHE_STALL_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(DRAM_PRECHARGE_RD_STATS)
                RUN_HYBRID_DRAM_HELPER(spmm_hybrid_worklist_dynamic_compiler_vectorized, dram_precharge_rd_perf, dram_precharge_rd_counters, perf_event_set::DRAM_PRECHARGE_RD_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(DRAM_PRECHARGE_WR_STATS)
                RUN_HYBRID_DRAM_HELPER(spmm_hybrid_worklist_dynamic_compiler_vectorized, dram_precharge_wr_perf, dram_precharge_wr_counters, perf_event_set::DRAM_PRECHARGE_WR_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(DRAM_ACTIVATE_RD_STATS)
                RUN_HYBRID_DRAM_HELPER(spmm_hybrid_worklist_dynamic_compiler_vectorized, dram_activate_rd_perf, dram_activate_rd_counters, perf_event_set::DRAM_ACTIVATE_RD_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(DRAM_ACTIVATE_WR_STATS)
                RUN_HYBRID_DRAM_HELPER(spmm_hybrid_worklist_dynamic_compiler_vectorized, dram_activate_wr_perf, dram_activate_wr_counters, perf_event_set::DRAM_ACTIVATE_WR_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(TLB_STATS)
                RUN_HYBRID_PERF_HELPER(spmm_hybrid_worklist_dynamic_compiler_vectorized, tlb_perf, tlb_counters, perf_event_set::TLB_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #endif

        #else
            #ifdef TRACK_PER_CORE_RUNTIME
                #ifdef TRACK_PER_PANEL_RUNTIME
                    RUN_HYBRID_PER_PANEL_TIMING( spmm_hybrid_worklist_dynamic_compiler_vectorized, S_dcsh, pairs_worklist, num_panels_per_thread );
                #else
                    RUN_HYBRID_PER_CORE_TIMING( spmm_hybrid_worklist_dynamic_compiler_vectorized, S_dcsh, pairs_worklist, num_panels_per_thread );
                #endif
            #else
                RUN_HYBRID_TIMING( spmm_hybrid_worklist_dynamic_compiler_vectorized, S_dcsh, pairs_worklist, num_panels_per_thread );
            #endif
        #endif

    #endif // RUN_DYN_HYB_EXPR






    // Data movement experiment
    #if defined(RUN_DYN_HYB_DM_EXPR)
        PRINT_TYPE("DYNAMIC HYBRID PANEL EXPERIMENT");

        MEM_RESET(C, countC, layers);
        #ifdef TRACK_PER_CORE_RUNTIME
            #ifdef TRACK_PER_PANEL_RUNTIME
                RUN_HYBRID_PER_PANEL_WARMUP_EXPR( spmm_hybrid_worklist_dynamic_data_movement_only, S_dcsh, pairs_worklist, num_panels_per_thread );
            #else
                RUN_HYBRID_PER_CORE_WARMUP_EXPR( spmm_hybrid_worklist_dynamic_data_movement_only, S_dcsh, pairs_worklist, num_panels_per_thread );
            #endif
        #else
            RUN_HYBRID_WARMUP_EXPR( spmm_hybrid_worklist_dynamic_data_movement_only, S_dcsh, pairs_worklist, num_panels_per_thread );
        #endif
        MEM_RESET(C, countC, layers);
        #if defined(TRACK_STATS_NEW)
            #if defined(REG_STATS)
                RUN_HYBRID_PERF_HELPER(spmm_hybrid_worklist_dynamic_data_movement_only, reg_perf, reg_counters, perf_event_set::REG_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(CACHE_STATS)
                RUN_HYBRID_PERF_HELPER(spmm_hybrid_worklist_dynamic_data_movement_only, cache_perf, cache_counters, perf_event_set::CACHE_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(STALL_STATS)
                RUN_HYBRID_PERF_HELPER(spmm_hybrid_worklist_dynamic_data_movement_only, stall_perf, stall_counters, perf_event_set::NAMED_STALL_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(DRAM_STATS)
                RUN_HYBRID_DRAM_HELPER(spmm_hybrid_worklist_dynamic_data_movement_only, dram_perf, dram_counters, perf_event_set::DRAM_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(CACHE_STALL_STATS)
                RUN_HYBRID_PERF_HELPER(spmm_hybrid_worklist_dynamic_data_movement_only, cache_stall_perf, cache_stall_counters, perf_event_set::CACHE_STALL_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(DRAM_PRECHARGE_RD_STATS)
                RUN_HYBRID_DRAM_HELPER(spmm_hybrid_worklist_dynamic_data_movement_only, dram_precharge_rd_perf, dram_precharge_rd_counters, perf_event_set::DRAM_PRECHARGE_RD_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(DRAM_PRECHARGE_WR_STATS)
                RUN_HYBRID_DRAM_HELPER(spmm_hybrid_worklist_dynamic_data_movement_only, dram_precharge_wr_perf, dram_precharge_wr_counters, perf_event_set::DRAM_PRECHARGE_WR_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(DRAM_ACTIVATE_RD_STATS)
                RUN_HYBRID_DRAM_HELPER(spmm_hybrid_worklist_dynamic_data_movement_only, dram_activate_rd_perf, dram_activate_rd_counters, perf_event_set::DRAM_ACTIVATE_RD_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(DRAM_ACTIVATE_WR_STATS)
                RUN_HYBRID_DRAM_HELPER(spmm_hybrid_worklist_dynamic_data_movement_only, dram_activate_wr_perf, dram_activate_wr_counters, perf_event_set::DRAM_ACTIVATE_WR_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #elif defined(TLB_STATS)
                RUN_HYBRID_PERF_HELPER(spmm_hybrid_worklist_dynamic_data_movement_only, tlb_perf, tlb_counters, perf_event_set::TLB_EVENTS, S_dcsh, pairs_worklist, num_panels_per_thread );
            #endif

        #else
            #ifdef TRACK_PER_CORE_RUNTIME
                #ifdef TRACK_PER_PANEL_RUNTIME
                    RUN_HYBRID_PER_PANEL_TIMING( spmm_hybrid_worklist_dynamic, S_dcsh, pairs_worklist, num_panels_per_thread );
                #else
                    RUN_HYBRID_PER_CORE_TIMING( spmm_hybrid_worklist_dynamic, S_dcsh, pairs_worklist, num_panels_per_thread );
                #endif
            #else
                RUN_HYBRID_TIMING( spmm_hybrid_worklist_dynamic_data_movement_only, S_dcsh, pairs_worklist, num_panels_per_thread );
            #endif
        #endif

    #endif // RUN_DYN_HYB_EXPR



    #if defined(RUN_DCSC_WORKLIST)
    {
        PRINT_TYPE("RUN_DCSC_WORKLIST");
        #ifdef RUN_CORRECTNESS_CHECK
        {
            T *check_O = (T *) std::aligned_alloc( ALLOC_ALIGNMENT,  nrows * feature * sizeof(T) );
            T *O = (T *) std::aligned_alloc( ALLOC_ALIGNMENT,  nrows * feature * sizeof(T) );

            reset_matrix_helper<T, ITYPE>( check_O, (nrows * feature) );
            // reset_matrix_helper<T>( O, (nrows * feature) );
            MEM_RESET(C, countC, layers);


            // f<T, ITYPE>( *A, B, O, feature, Ti, Tk, chunk_size );

            spmm_dcsc_jstream_worklist( *S_dcsc, B, O, feature, num_panels_per_thread, panel_worklist );

            // spmm_csr_jstream_worklist(*S_csr, B, O, feature, num_panels_per_thread, work_list, num_threads, chunk_size);
            std::cout << "Is correct? " << check_simple( *S_csr, B, check_O, feature, O ) << std::endl;

            std::free(check_O);
            std::free(O);
        }
        #endif // RUN_CORRECTNESS_CHECK

        {
            stats_t<long long, ITYPE> cycle_counts;
            cycle_counts.name = "Cycle Counts";
            long long per_core_cycle_count[num_threads];

            #ifdef TRACK_PER_CORE_RUNTIME
                stats_t<long long, ITYPE> per_core_cycle_count_stats[num_threads];
                stats_t<double, ITYPE> per_core_execution_time_stats[num_threads];
                for (ITYPE i = 0; i < num_threads; i++) {
                    per_core_cycle_count_stats[i].name = "core:" + std::to_string(i);
                    per_core_execution_time_stats[i].name = "core:" + std::to_string(i);
                }
            #endif

            #ifdef TRACK_PER_PANEL_RUNTIME
                ITYPE max_panels_per_thread = 0;
                ITYPE total_panels = 0;
                for ( ITYPE i = 0; i < num_threads; i++ ) {
                    total_panels += num_panels_per_thread[i];
                    if ( max_panels_per_thread < num_panels_per_thread[i] ) {
                        max_panels_per_thread = num_panels_per_thread[i];
                    }
                }
                long long *per_panel_cycle_counts = new long long[ num_threads * max_panels_per_thread ];
                stats_t<long long, ITYPE> *per_panel_cycle_count_stats = new stats_t<long long, ITYPE>[num_threads];
                stats_t<double, ITYPE> *per_panel_execution_time_stats = new stats_t<double, ITYPE>[num_threads];
                for ( ITYPE tid = 0; tid < num_threads; tid++ ) {
                    ITYPE num_records_expected = n * num_panels_per_thread[tid];
                    per_panel_cycle_count_stats[tid].name = "per panel core: " + std::to_string(tid);
                    per_panel_execution_time_stats[tid].name = "per panel core: " + std::to_string(tid);
                    per_panel_cycle_count_stats[tid].resize( num_records_expected );
                    per_panel_execution_time_stats[tid].resize( num_records_expected );
                }
            #endif

            for ( ITYPE i = 0; i < n; i++ ) {

                T *input = &B[ (i % layers) * (ncols * feature) ];
                T *output = &C[ (i % layers) * (nrows * feature) ];

                CACHE_FLUSH;

                // long long duration = spmm_csr_jstream_worklist(*S_csr, input, output, feature, num_panels_per_thread, work_list, num_threads, chunk_size, per_core_cycle_count);
                #ifdef TRACK_PER_PANEL_RUNTIME
                    long long duration = spmm_csr_jstream_worklist(*S_csr, input, output, feature, num_panels_per_thread, work_list, num_threads, chunk_size, per_core_cycle_count, per_panel_cycle_counts);
                #else
                    long long duration = spmm_dcsc_jstream_worklist( *S_dcsc, input, output, feature, num_panels_per_thread, panel_worklist, per_core_cycle_count, nullptr, num_threads );
                #endif

                // Add cycle counts to the stats array
                #ifdef TRACK_PER_CORE_RUNTIME
                    for (ITYPE i = 0; i < num_threads; i++) {
                        per_core_cycle_count_stats[i].insert(per_core_cycle_count[i]);
                    }
                #endif

                #ifdef TRACK_PER_PANEL_RUNTIME
                    // insert into the corresponding stats object
                    for ( ITYPE tid = 0; tid < num_threads; tid++ ) {
                        for ( ITYPE panel_id = 0; panel_id < num_panels_per_thread[tid]; panel_id++ ) {
                            per_panel_cycle_count_stats[tid].insert( per_panel_cycle_counts[ num_threads * panel_id + tid ] );
                        }
                    }
                #endif

                cycle_counts.insert(duration);

            }
            cycle_counts.process();
            cycle_counts.print();
            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;
            std::cout << "Warmup Median Time: " << median_time << std::endl;

            #ifdef TRACK_PER_PANEL_RUNTIME
                // insert into the corresponding stats object
                for ( ITYPE tid = 0; tid < num_threads; tid++ ) {
                    per_panel_cycle_count_stats[tid].process();
                    // per_panel_cycle_count_stats[tid].print();
                }

                // lets free memory - wohoo!
            #endif
        } // End warmup iteration set

        // reset_matrix_helper(C, sizeC);
        MEM_RESET(C, countC, layers);

        {
            stats_t<long long, ITYPE> cycle_counts;
            cycle_counts.name = "Cycle Counts";
            long long per_core_cycle_count[num_threads];

            #ifdef TRACK_PER_CORE_RUNTIME
                stats_t<long long, ITYPE> per_core_cycle_count_stats[num_threads];
                stats_t<double, ITYPE> per_core_execution_time_stats[num_threads];
                for (ITYPE i = 0; i < num_threads; i++) {
                    per_core_cycle_count_stats[i].name = "core:" + std::to_string(i);
                    per_core_execution_time_stats[i].name = "core:" + std::to_string(i);
                }
            #endif

            #ifdef TRACK_PER_PANEL_RUNTIME
                ITYPE max_panels_per_thread = 0;
                ITYPE total_panels = 0;
                for ( ITYPE i = 0; i < num_threads; i++ ) {
                    total_panels += num_panels_per_thread[i];
                    if ( max_panels_per_thread < num_panels_per_thread[i] ) {
                        max_panels_per_thread = num_panels_per_thread[i];
                    }
                }
                long long *per_panel_cycle_counts = new long long[ num_threads * max_panels_per_thread ];
                stats_t<long long, ITYPE> *per_panel_cycle_count_stats = new stats_t<long long, ITYPE>[num_threads];
                stats_t<double, ITYPE> *per_panel_execution_time_stats = new stats_t<double, ITYPE>[num_threads];
                for ( ITYPE tid = 0; tid < num_threads; tid++ ) {
                    ITYPE num_records_expected = n * num_panels_per_thread[tid];
                    per_panel_cycle_count_stats[tid].name = "per panel core: " + std::to_string(tid);
                    per_panel_execution_time_stats[tid].name = "per panel core: " + std::to_string(tid);
                    per_panel_cycle_count_stats[tid].resize( num_records_expected );
                    per_panel_execution_time_stats[tid].resize( num_records_expected );
                }
            #endif

            for ( ITYPE i = 0; i < n; i++ ) {

                T *input = &B[ (i % layers) * (ncols * feature) ];
                T *output = &C[ (i % layers) * (nrows * feature) ];

                CACHE_FLUSH;

                // long long duration = spmm_csr_jstream_worklist(*S_csr, input, output, feature, num_panels_per_thread, work_list, num_threads, chunk_size, per_core_cycle_count);
                #ifdef TRACK_PER_PANEL_RUNTIME
                    long long duration = spmm_csr_jstream_worklist(*S_csr, input, output, feature, num_panels_per_thread, work_list, num_threads, chunk_size, per_core_cycle_count, per_panel_cycle_counts);
                #else
                    long long duration = spmm_dcsc_jstream_worklist( *S_dcsc, input, output, feature, num_panels_per_thread, panel_worklist, per_core_cycle_count, nullptr, num_threads );
                #endif

                // Add cycle counts to the stats array
                #ifdef TRACK_PER_CORE_RUNTIME
                    for (ITYPE i = 0; i < num_threads; i++) {
                        per_core_cycle_count_stats[i].insert(per_core_cycle_count[i]);
                    }
                #endif

                #ifdef TRACK_PER_PANEL_RUNTIME
                    // insert into the corresponding stats object
                    for ( ITYPE tid = 0; tid < num_threads; tid++ ) {
                        for ( ITYPE panel_id = 0; panel_id < num_panels_per_thread[tid]; panel_id++ ) {
                            per_panel_cycle_count_stats[tid].insert( per_panel_cycle_counts[ num_threads * panel_id + tid ] );
                        }
                    }
                #endif

                cycle_counts.insert(duration);

            }
            cycle_counts.process();
            cycle_counts.print();
            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;
            std::cout << "Median Time: " << median_time << std::endl;

            #ifdef TRACK_PER_CORE_RUNTIME
                double min_time = std::numeric_limits<double>::max();
                double max_time = 0;
                for (ITYPE tid = 0; tid < num_threads; tid++) {
                    per_core_cycle_count_stats[tid].process();
                    // per_core_cycle_count_stats[i].print();
                    for ( ITYPE record_id = 0; record_id < per_core_cycle_count_stats[tid].size; record_id++ ) {
                        double execution_time = ( (double) per_core_cycle_count_stats[tid].records[record_id]) / CLOCK_FREQUENCY;
                        per_core_execution_time_stats[tid].insert( execution_time );
                    }
                    per_core_execution_time_stats[tid].process();
                    // per_core_execution_time_stats[tid].print();
                    // std::cout << "Core: " << tid << " -- " << per_core_execution_time_stats[tid].median << std::endl;
                    if (per_core_execution_time_stats[tid].median < min_time) {
                        min_time = per_core_execution_time_stats[tid].median;
                    }
                    if (per_core_execution_time_stats[tid].median > max_time) {
                        max_time = per_core_execution_time_stats[tid].median;
                    }
                }
                std::cout << "Min Time: " << min_time << std::endl;
                std::cout << "Max Time: " << max_time << std::endl;
                double total_core_time = 0;
                for ( ITYPE tid = 0; tid < num_threads; tid++ ) {
                    double diff_to_median = median_time - per_core_execution_time_stats[tid].median;
                    double idle_perc = diff_to_median / max_time * 100.0;
                    // std::cout << "Core: " << tid << " -- " << per_core_execution_time_stats[tid].median << " -- diff: " << idle_perc << std::endl;
                    std::cout << "core: " << tid << ", " << per_core_execution_time_stats[tid].median << ", " << idle_perc << std::endl;
                    total_core_time += per_core_execution_time_stats[tid].median;
                }
                std::cout << "Total Core Time: " << total_core_time << std::endl;
                std::cout << "Ideal Avg Time: " << (total_core_time / ((double) num_threads)) << std::endl;
            #endif
        } // End iteration set


    }

    #endif

    #if defined(RUN_CSR_WORKLIST)
        PRINT_TYPE("CSR_WORKLIST");

        // long long spmm_csr_jstream_worklist( CSR<T, ITYPE> &S, T *I, T *O, ITYPE feature, ITYPE *num_panels, struct worklist *worklist, ITYPE num_threads, ITYPE chunk_size = 1 )

        #ifdef RUN_CORRECTNESS_CHECK
        {
            T *check_O = (T *) std::aligned_alloc( ALLOC_ALIGNMENT,  nrows * feature * sizeof(T) );
            T *O = (T *) std::aligned_alloc( ALLOC_ALIGNMENT,  nrows * feature * sizeof(T) );

            reset_matrix_helper<T, ITYPE>( check_O, (nrows * feature) );
            // reset_matrix_helper<T>( O, (nrows * feature) );
            MEM_RESET(C, countC, layers);

            // f<T, ITYPE>( *A, B, O, feature, Ti, Tk, chunk_size );


            spmm_csr_jstream_worklist(*S_csr, B, O, feature, num_panels_per_thread, work_list, num_threads, chunk_size);
            std::cout << "Is correct? " << check_simple( *S_csr, B, check_O, feature, O ) << std::endl;

            std::free(check_O);
            std::free(O);
        }
        #endif // RUN_CORRECTNESS_CHECK

        {
            stats_t<long long, ITYPE> cycle_counts;
            cycle_counts.name = "Cycle Counts";
            long long per_core_cycle_count[num_threads];

            #ifdef TRACK_PER_CORE_RUNTIME
                stats_t<long long, ITYPE> per_core_cycle_count_stats[num_threads];
                stats_t<double, ITYPE> per_core_execution_time_stats[num_threads];
                for (ITYPE i = 0; i < num_threads; i++) {
                    per_core_cycle_count_stats[i].name = "core:" + std::to_string(i);
                    per_core_execution_time_stats[i].name = "core:" + std::to_string(i);
                }
            #endif

            #ifdef TRACK_PER_PANEL_RUNTIME
                ITYPE max_panels_per_thread = 0;
                ITYPE total_panels = 0;
                for ( ITYPE i = 0; i < num_threads; i++ ) {
                    total_panels += num_panels_per_thread[i];
                    if ( max_panels_per_thread < num_panels_per_thread[i] ) {
                        max_panels_per_thread = num_panels_per_thread[i];
                    }
                }
                long long *per_panel_cycle_counts = new long long[ num_threads * max_panels_per_thread ];
                stats_t<long long, ITYPE> *per_panel_cycle_count_stats = new stats_t<long long, ITYPE>[num_threads];
                stats_t<double, ITYPE> *per_panel_execution_time_stats = new stats_t<double, ITYPE>[num_threads];
                for ( ITYPE tid = 0; tid < num_threads; tid++ ) {
                    ITYPE num_records_expected = n * num_panels_per_thread[tid];
                    per_panel_cycle_count_stats[tid].name = "per panel core: " + std::to_string(tid);
                    per_panel_execution_time_stats[tid].name = "per panel core: " + std::to_string(tid);
                    per_panel_cycle_count_stats[tid].resize( num_records_expected );
                    per_panel_execution_time_stats[tid].resize( num_records_expected );
                }
            #endif

            for ( ITYPE i = 0; i < n; i++ ) {

                T *input = &B[ (i % layers) * (ncols * feature) ];
                T *output = &C[ (i % layers) * (nrows * feature) ];

                CACHE_FLUSH;

                // long long duration = spmm_csr_jstream_worklist(*S_csr, input, output, feature, num_panels_per_thread, work_list, num_threads, chunk_size, per_core_cycle_count);
                #ifdef TRACK_PER_PANEL_RUNTIME
                    long long duration = spmm_csr_jstream_worklist(*S_csr, input, output, feature, num_panels_per_thread, work_list, num_threads, chunk_size, per_core_cycle_count, per_panel_cycle_counts);
                #else
                    long long duration = spmm_csr_jstream_worklist(*S_csr, input, output, feature, num_panels_per_thread, work_list, num_threads, chunk_size, per_core_cycle_count);
                #endif

                // Add cycle counts to the stats array
                #ifdef TRACK_PER_CORE_RUNTIME
                    for (ITYPE i = 0; i < num_threads; i++) {
                        per_core_cycle_count_stats[i].insert(per_core_cycle_count[i]);
                    }
                #endif

                #ifdef TRACK_PER_PANEL_RUNTIME
                    // insert into the corresponding stats object
                    for ( ITYPE tid = 0; tid < num_threads; tid++ ) {
                        for ( ITYPE panel_id = 0; panel_id < num_panels_per_thread[tid]; panel_id++ ) {
                            per_panel_cycle_count_stats[tid].insert( per_panel_cycle_counts[ num_threads * panel_id + tid ] );
                        }
                    }
                #endif

                cycle_counts.insert(duration);

            }
            cycle_counts.process();
            cycle_counts.print();
            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;
            std::cout << "Warmup Median Time: " << median_time << std::endl;

            #ifdef TRACK_PER_PANEL_RUNTIME
                // insert into the corresponding stats object
                for ( ITYPE tid = 0; tid < num_threads; tid++ ) {
                    per_panel_cycle_count_stats[tid].process();
                    // per_panel_cycle_count_stats[tid].print();
                }

                // lets free memory - wohoo!
            #endif

            // #ifdef TRACK_PER_CORE_RUNTIME
            //     for (ITYPE tid = 0; tid < num_threads; tid++) {
            //         per_core_cycle_count_stats[tid].process();
            //         // per_core_cycle_count_stats[i].print();
            //         for ( ITYPE n = 0; n < per_core_cycle_count_stats[tid].size; n++ ) {
            //             double execution_time = ( (double) per_core_cycle_count_stats[tid].records[n]) / CLOCK_FREQUENCY;
            //             per_core_execution_time_stats[tid].insert( execution_time );
            //         }
            //         per_core_execution_time_stats[tid].process();
            //         per_core_execution_time_stats[tid].print();
            //     }
            // #endif
        } // End warmup iteration set

        // et_matrix_helper(C, sizeC);
        MEM_RESET(C, countC, layers);

        {
            stats_t<long long, ITYPE> cycle_counts;
            cycle_counts.name = "Cycle Counts";
            long long per_core_cycle_count[num_threads];

            #ifdef TRACK_PER_CORE_RUNTIME
                stats_t<long long, ITYPE> per_core_cycle_count_stats[num_threads];
                stats_t<double, ITYPE> per_core_execution_time_stats[num_threads];
                for (ITYPE i = 0; i < num_threads; i++) {
                    per_core_cycle_count_stats[i].name = "core:" + std::to_string(i);
                    per_core_execution_time_stats[i].name = "core:" + std::to_string(i);
                }
            #endif

            #ifdef TRACK_PER_PANEL_RUNTIME
                ITYPE max_panels_per_thread = 0;
                ITYPE total_panels = 0;
                for ( ITYPE i = 0; i < num_threads; i++ ) {
                    total_panels += num_panels_per_thread[i];
                    if ( max_panels_per_thread < num_panels_per_thread[i] ) {
                        max_panels_per_thread = num_panels_per_thread[i];
                    }
                }
                long long *per_panel_cycle_counts = new long long[ num_threads * max_panels_per_thread ];
                stats_t<long long, ITYPE> *per_panel_cycle_count_stats = new stats_t<long long, ITYPE>[num_threads];
                stats_t<double, ITYPE> *per_panel_execution_time_stats = new stats_t<double, ITYPE>[num_threads];
                for ( ITYPE tid = 0; tid < num_threads; tid++ ) {
                    ITYPE num_records_expected = n * num_panels_per_thread[tid];
                    per_panel_cycle_count_stats[tid].name = "per panel core: " + std::to_string(tid);
                    per_panel_execution_time_stats[tid].name = "per panel core: " + std::to_string(tid);
                    per_panel_cycle_count_stats[tid].resize( num_records_expected );
                    per_panel_execution_time_stats[tid].resize( num_records_expected );
                }
            #endif

            for ( ITYPE i = 0; i < n; i++ ) {

                T *input = &B[ (i % layers) * (ncols * feature) ];
                T *output = &C[ (i % layers) * (nrows * feature) ];

                CACHE_FLUSH;

                // long long duration = spmm_csr_jstream_worklist(*S_csr, input, output, feature, num_panels_per_thread, work_list, num_threads, chunk_size, per_core_cycle_count);
                #ifdef TRACK_PER_PANEL_RUNTIME
                    long long duration = spmm_csr_jstream_worklist(*S_csr, input, output, feature, num_panels_per_thread, work_list, num_threads, chunk_size, per_core_cycle_count, per_panel_cycle_counts);
                #else
                    long long duration = spmm_csr_jstream_worklist(*S_csr, input, output, feature, num_panels_per_thread, work_list, num_threads, chunk_size, per_core_cycle_count);
                #endif

                // Add cycle counts to the stats array
                #ifdef TRACK_PER_CORE_RUNTIME
                    for (ITYPE i = 0; i < num_threads; i++) {
                        per_core_cycle_count_stats[i].insert(per_core_cycle_count[i]);
                    }
                #endif

                #ifdef TRACK_PER_PANEL_RUNTIME
                    // insert into the corresponding stats object
                    for ( ITYPE tid = 0; tid < num_threads; tid++ ) {
                        for ( ITYPE panel_id = 0; panel_id < num_panels_per_thread[tid]; panel_id++ ) {
                            per_panel_cycle_count_stats[tid].insert( per_panel_cycle_counts[ num_threads * panel_id + tid ] );
                        }
                    }
                #endif

                cycle_counts.insert(duration);

            }
            cycle_counts.process();
            cycle_counts.print();
            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;
            std::cout << "Median Time: " << median_time << std::endl;


            #ifdef TRACK_PER_CORE_RUNTIME
                double min_time = std::numeric_limits<double>::max();
                double max_time = 0;
                for (ITYPE tid = 0; tid < num_threads; tid++) {
                    per_core_cycle_count_stats[tid].process();
                    // per_core_cycle_count_stats[i].print();
                    for ( ITYPE record_id = 0; record_id < per_core_cycle_count_stats[tid].size; record_id++ ) {
                        double execution_time = ( (double) per_core_cycle_count_stats[tid].records[record_id]) / CLOCK_FREQUENCY;
                        per_core_execution_time_stats[tid].insert( execution_time );
                    }
                    per_core_execution_time_stats[tid].process();
                    // per_core_execution_time_stats[tid].print();
                    // std::cout << "Core: " << tid << " -- " << per_core_execution_time_stats[tid].median << std::endl;
                    if (per_core_execution_time_stats[tid].median < min_time) {
                        min_time = per_core_execution_time_stats[tid].median;
                    }
                    if (per_core_execution_time_stats[tid].median > max_time) {
                        max_time = per_core_execution_time_stats[tid].median;
                    }
                }
                std::cout << "Min Time: " << min_time << std::endl;
                std::cout << "Max Time: " << max_time << std::endl;
                for ( ITYPE tid = 0; tid < num_threads; tid++ ) {
                    double diff_to_median = median_time - per_core_execution_time_stats[tid].median;
                    double idle_perc = diff_to_median / max_time * 100.0;
                    // std::cout << "Core: " << tid << " -- " << per_core_execution_time_stats[tid].median << " -- diff: " << idle_perc << std::endl;
                    std::cout << "core: " << tid << ", " << per_core_execution_time_stats[tid].median << ", " << idle_perc << std::endl;
                }
            #endif

            #ifdef TRACK_PER_PANEL_RUNTIME
                // insert into the corresponding stats object
                for ( ITYPE tid = 0; tid < num_threads; tid++ ) {
                    per_panel_cycle_count_stats[tid].process();
                    for ( ITYPE record_id = 0; record_id < per_panel_cycle_count_stats[tid].size; record_id++ ) {
                        double panel_execution_time = ((double) per_panel_cycle_count_stats[tid].records[record_id]) / CLOCK_FREQUENCY;
                        per_panel_execution_time_stats[tid].insert( panel_execution_time );
                    }
                    // per_panel_cycle_count_stats[tid].print();
                    // per_panel_execution_time_stats[tid].print();
                }

                // do the medianization of the per panel stats
                long long *processed_per_panel_cycle_counts = new long long[ num_threads * max_panels_per_thread ];
                double *processed_per_panel_execution_times = new double[ num_threads * max_panels_per_thread ];

                stats_t<long long, ITYPE> temp_per_panel_cycle;
                stats_t<double, ITYPE> processed_panel_runtimes[num_threads];
                for ( ITYPE tid = 0; tid < num_threads; tid++ ) {
                    processed_panel_runtimes[tid].name = "core:" + std::to_string(tid);
                    for ( ITYPE panel_id = 0; panel_id < num_panels_per_thread[tid]; panel_id++ ) {
                        temp_per_panel_cycle.reset();

                        for ( ITYPE record_id = 0; record_id < n; record_id++ ) {
                            // ITYPE record_offset = (panel_id * n) + record_id;
                            ITYPE record_offset = panel_id + record_id * num_panels_per_thread[tid];

                            temp_per_panel_cycle.insert( per_panel_cycle_count_stats[tid].records[record_offset] );
                        }

                        temp_per_panel_cycle.process();
                        processed_per_panel_cycle_counts[ panel_id * num_threads + tid ] = temp_per_panel_cycle.median;
                        double panel_execution_time = ( (double) temp_per_panel_cycle.median ) / CLOCK_FREQUENCY;
                        processed_per_panel_execution_times[ panel_id * num_threads + tid ] = panel_execution_time;
                        processed_panel_runtimes[tid].insert(panel_execution_time);
                    }
                }

                for ( ITYPE tid = 0; tid < num_threads; tid++ ) {
                    // std::cout << "core: " << tid << " -- ";
                    std::cout << "core: " << tid << ", ";
                    double core_total_time = 0;
                    for ( ITYPE panel_id = 0; panel_id < num_panels_per_thread[tid]; panel_id++ ) {
                        ITYPE worklist_index = panel_id * num_threads + tid;
                        std::cout << processed_per_panel_execution_times[ worklist_index ] << ",";
                        // std::cout << work_list[ worklist_index ].panel_id << ":" << processed_per_panel_execution_times[ worklist_index ] << "; ";
                        core_total_time += processed_per_panel_execution_times[ worklist_index ];
                    }
                    std::cout << std::endl;
                    // std::cout << "total: " << core_total_time << std::endl;

                }


                auto size_mtx_name = mtx_filename.find_last_of(".") - mtx_filename.find_last_of("/") - 1;
                auto start_mtx_name = mtx_filename.find_last_of("/") + 1;
                std::string csv_file_name = std::string("/users/ajain324/new-output/imbalance-bytes-score-I-cached-only-sweep/") + mtx_filename.substr(start_mtx_name, size_mtx_name);

                std::string message = "CSR_WORKLIST -- Core Timing -- alpha: " + std::to_string(alpha_val) + " -- beta: " + std::to_string(beta_val);
                stats_arr_to_csv( csv_file_name, message, num_threads, processed_panel_runtimes );

                // lets free memory - wohoo!
            #endif

        }

    #endif


    #ifdef INTEL_MKL

    {
        PRINT_TYPE("INTEL MKL SPMM");

        sparse_matrix_t mkl_S;
        struct matrix_descr mkl_S_desc;
        mkl_S_desc.type = SPARSE_MATRIX_TYPE_GENERAL;
        T mkl_alpha = 1.0;
        T mkl_beta =  0.0;

        MKL_INT mkl_n = n;
        MKL_INT mkl_nrows = S_csr->nrows;
        MKL_INT mkl_ncols = S_csr->ncols;
        MKL_INT mkl_feature = feature;
        MKL_INT mkl_nnzs = S_csr->nnzs;
        MKL_INT *mkl_row_start = (MKL_INT *) std::aligned_alloc(ALLOC_ALIGNMENT, sizeof(MKL_INT) * mkl_nrows);
        MKL_INT *mkl_row_end = (MKL_INT *) std::aligned_alloc(ALLOC_ALIGNMENT, sizeof(MKL_INT) * mkl_nrows);
        MKL_INT *mkl_cols = (MKL_INT *) std::aligned_alloc(ALLOC_ALIGNMENT, sizeof(MKL_INT) * mkl_nnzs);

        for ( ITYPE i = 0; i < mkl_nrows; i++ ) {
            mkl_row_start[i] = S_csr->row_ptr[i];
            mkl_row_end[i] = S_csr->row_ptr[i + 1];
        }

        for ( ITYPE i = 0; i < mkl_nnzs; i++ ) {
            mkl_cols[i] = S_csr->cols[i];
        }

        // create the MKL csr matrix
        sparse_status_t status = mkl_sparse_d_create_csr( &mkl_S, SPARSE_INDEX_BASE_ZERO, mkl_nrows, mkl_ncols, mkl_row_start, mkl_row_end, mkl_cols, S_csr->vals );
        if ( status != SPARSE_STATUS_SUCCESS ) {
            print_error_exit("Could not create mkl sparse matrix");
        }

        // Warmup iteration
        // status = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, mkl_alpha, mkl_S, mkl_S_desc, SPARSE_LAYOUT_ROW_MAJOR, B, mkl_ncols, mkl_feature, mkl_beta, C, mkl_nrows );

        auto hint_start_time = std::chrono::high_resolution_clock::now();
        status = mkl_sparse_set_mm_hint( mkl_S, SPARSE_OPERATION_NON_TRANSPOSE, mkl_S_desc, SPARSE_LAYOUT_ROW_MAJOR, mkl_feature, mkl_n );
        auto hint_end_time = std::chrono::high_resolution_clock::now();
        std::cerr << "MKL hint execution time: " << std::chrono::duration<double>( hint_end_time - hint_start_time ).count() << std::endl;

        #ifdef RUN_WARMUP
        {
            stats_t<long long, ITYPE> cycle_counts;
            cycle_counts.name = "Cycle Counts";
            for ( ITYPE i = 0; i < n/WARMUP_DIVIDER; i++ ) {
                T *input = &(B[ (i % layers) ][0]);
                T *output = &(C[ (i % layers) ][0]);
                CACHE_FLUSH;
                // PERF_START(p);

                // Need to do the timing here
                auto start_cycle = readTSC();
                // mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, mkl_alpha, mkl_S, mkl_S_desc, SPARSE_LAYOUT_ROW_MAJOR, input, mkl_feature, mkl_ncols, mkl_beta, output, mkl_nrows );
                mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, mkl_alpha, mkl_S, mkl_S_desc, SPARSE_LAYOUT_ROW_MAJOR, input, mkl_feature, mkl_feature, mkl_beta, output, mkl_feature );
                auto end_cycle = readTSC();

                long long duration = end_cycle - start_cycle;

                // PERF_STOP(p, s);
                cycle_counts.insert(duration);
            }
            cycle_counts.process();
            cycle_counts.print();
            double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;
            std::cout << "Warmup Median Time: " << median_time << std::endl;
            // PERF_PRINT(e, s, num_threads, num_events_to_print);
            double flop_count = 2 * mkl_nnzs * feature;
            double gflops = flop_count / median_time / 1E9;
            // std::cout << "GFLOPS: " << gflops << std::endl;

        }
        MEM_RESET(C, countC, layers);
        #endif

        #if defined(TRACK_STATS_NEW)
            #if defined(REG_STATS)
                RUN_MKL_PERF_HELPER( reg_perf, reg_counters, perf_event_set::REG_EVENTS );
            #elif defined(CACHE_STATS)
                RUN_MKL_PERF_HELPER( cache_perf, cache_counters, perf_event_set::CACHE_EVENTS );
            #elif defined(STALL_STATS)
                RUN_MKL_PERF_HELPER( stall_perf, stall_counters, perf_event_set::NAMED_STALL_EVENTS );
            #elif defined(L3_CACHE_STATS)
                RUN_MKL_PERF_HELPER( l3_cache_perf, l3_cache_counters, perf_event_set::L3_CACHE_EVENTS );
                // RUN_MKL_DRAM_HELPER( dram_perf, dram_counters, perf_event_set::DRAM_EVENTS );
            #elif defined(CACHE_STALL_STATS)
                RUN_MKL_PERF_HELPER( cache_stall_perf, cache_stall_counters, perf_event_set::CACHE_STALL_EVENTS );
            #elif defined(DRAM_PRECHARGE_RD_STATS)
                RUN_MKL_PERF_HELPER( dram_precharge_rd_perf, dram_precharge_rd_counters, perf_event_set::DRAM_PRECHARGE_RD_EVENTS );
            #elif defined(DRAM_PRECHARGE_WR_STATS)
            #elif defined(DRAM_ACTIVATE_RD_STATS)
            #elif defined(DRAM_ACTIVATE_WR_STATS)
            #elif defined(TLB_STATS)
                RUN_MKL_PERF_HELPER( tlb_perf, tlb_counters, perf_event_set::TLB_EVENTS );
            #endif
        #else
            RUN_MKL_TIMING;
        #endif

            // RUN_MKL_PERF_HELPER(p,s,e)
        /*
        ITYPE num_events_to_print = perf::NUM_EVENTS_PER_SET;
        if (e == perf_event_set::CACHE_STALL_EVENTS) {
            num_events_to_print = NUM_CACHE_STALL_EVENTS;
        } else if (e == perf_event_set::NAMED_CACHE_EVENTS) {
            num_events_to_print = NUM_NAMED_CACHE_EVENTS;
        } else if (e == perf_event_set::L3_CACHE_EVENTS) {
            num_events_to_print = NUM_L3_CACHE_EVENTS;
        } else if (e == perf_event_set::NAMED_STALL_EVENTS) {
            num_events_to_print = NUM_NAMED_STALL_EVENTS;
        }
        */
    }


        // Let's only use doubles



    #endif // INTEL_MKL




    // Free counters and stuff
    // #ifdef TRACK_PARALLEL_STATS
    //     #ifndef TRACK_CACHE_ONLY
    //         free_perf_counters_helper<T, ITYPE>(reg_perf, reg_counters, perf::NUM_EVENTS_PER_SET, num_threads);
    //         free_perf_counters_helper<T, ITYPE>(cache_perf, cache_counters, perf::NUM_EVENTS_PER_SET, num_threads);
    //         free_perf_counters_helper<T, ITYPE>(stall_perf, stall_counters, perf::NUM_EVENTS_PER_SET, num_threads);
    //         free_dram_counters_helper<T, ITYPE>(dram_perf, dram_counters, NUM_MEMORY_CHANNELS * 2, num_threads);
    //     #else
    //         free_perf_counters_helper<T, ITYPE>(cache_only_perf, cache_only_counters, perf::NUM_EVENTS_PER_SET, num_threads);
    //     #endif
    // #endif

#ifdef LARGE_ARRAY
    release_memory(B);
    release_memory(C);
#else
    for (ITYPE l = 0; l < layers; l++) {
        release_memory(B[l]);
        release_memory(C[l]);
    }
#endif

    #ifdef RUN_DCSC_JSTREAM_EXPR
        delete S_dcsc;
    #endif

    #ifdef RUN_CSR_KSTREAM_EXPR
        delete S_stm;
    #endif

    #ifdef RUN_HYB_EXPR
        delete S_dcsh;
        std::free(pairs_worklist);  // Was allocated only for this particular matrix type
    #endif

    #ifdef RUN_CSR_ATM_EXPR
        delete S_atm;
    #endif

    if (!using_global) {
        if (locs) { delete[] locs; }
        if (vals) { delete[] vals; }
    }
    if (S_csr) { delete S_csr; }

    return;
}


#define RUN_SDDMM_KERNEL_WARMUP(f, s, o)                                                                \
                                {                                                                       \
                                    stats_t<long long, ITYPE> cycle_counts;                             \
                                    cycle_counts.name = "Cycle Count";                                  \
                                    for ( ITYPE i = 0; i < (n/10); i++ ) {                              \
                                        T *d1 = &B[ (i % layers) ][0];                                  \
                                        T *d2 = &C[ (i % layers) ][0];                                  \
                                        T *output = &o[ (i % layers) ][0];                              \
                                        CACHE_FLUSH;                                                    \
                                        long long duration = f( *s, d1, d2, output, feature, Ti, Tk, chunk_size );  \
                                        cycle_counts.insert( duration );                                \
                                    }                                                                   \
                                    cycle_counts.process();                                             \
                                    cycle_counts.print();                                               \
                                    double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;\
                                    std::cout << "Warmup Median Time: " << median_time << std::endl;    \
                                }


#ifdef TRACK_PARALLEL_STATS

// p - perf object
// s - stats object
// e - stats type
#define RUN_SDDMM_KERNEL(f, sp, o, p, s, e)                                                             \
                                {                                                                       \
                                    stats_t<long long, ITYPE> cycle_counts;                             \
                                    cycle_counts.name = "Cycle Count";                                  \
                                    for ( ITYPE i = 0; i < n; i++ ) {                                   \
                                        T *d1 = &B[ (i % layers) ][0];                                  \
                                        T *d2 = &C[ (i % layers) ][0];                                  \
                                        T *output = &o[ (i % layers) ][0];                              \
                                        CACHE_FLUSH;                                                    \
                                        PERF_START(p);                                                  \
                                        long long duration = f( *sp, d1, d2, output, feature, Ti, Tk, chunk_size );  \
                                        PERF_STOP(p, s);                                                \
                                        cycle_counts.insert( duration );                                \
                                    }                                                                   \
                                    cycle_counts.process();                                             \
                                    cycle_counts.print();                                               \
                                    double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;\
                                    std::cout << "Median Time: " << median_time << std::endl;           \
                                    ITYPE num_events_to_print = perf::NUM_EVENTS_PER_SET;               \
                                    if (e == perf_event_set::CACHE_STALL_EVENTS) {                      \
                                        num_events_to_print = NUM_CACHE_STALL_EVENTS;                   \
                                    } else if (e == perf_event_set::NAMED_CACHE_EVENTS) {               \
                                        num_events_to_print = NUM_NAMED_CACHE_EVENTS;                   \
                                    } else if (e == perf_event_set::L3_CACHE_EVENTS) {                  \
                                        num_events_to_print = NUM_L3_CACHE_EVENTS;                      \
                                    } else if (e == perf_event_set::NAMED_STALL_EVENTS) {               \
                                        num_events_to_print = NUM_NAMED_STALL_EVENTS;                   \
                                    }                                                                   \
                                    auto ret_stats = PERF_PRINT(e, s, num_threads, num_events_to_print);    \
                                    if ( e == perf_event_set::NAMED_STALL_EVENTS ) {                    \
                                        for ( ITYPE counter = 0; counter < NUM_NAMED_STALL_EVENTS; counter++ ) {    \
                                            double frac_stalled = ((double) CEIL( ret_stats[counter].median, num_threads )) / ((double) cycle_counts.median); \
                                            double perc_stalled = frac_stalled * 100.0;                 \
                                            std::cout << "Fraction -- " << ret_stats[counter].name << " -- " << frac_stalled << std::endl; \
                                            std::cout << "Percentage Stalled -- " << perc_stalled << std::endl; \
                                        }                                                               \
                                    }                                                                   \
                                    size_t flop_count = ((size_t)feature) * 2 * ((size_t)sp->nnzs) + ((size_t) sp->nnzs); \
                                    std::cout << "FLOP COUNT: " << flop_count << std::endl;             \
                                    double gflops = ((double)flop_count) / median_time / 1E9;           \
                                    std::cout << "GFLOPS: " << gflops << std::endl;                     \
                                }

#else

#define RUN_SDDMM_KERNEL(f, s, o)                                                                       \
                                {                                                                       \
                                    stats_t<long long, ITYPE> cycle_counts;                             \
                                    cycle_counts.name = "Cycle Count";                                  \
                                    for ( ITYPE i = 0; i < n; i++ ) {                                   \
                                        T *d1 = &B[ (i % layers) ][0];                                  \
                                        T *d2 = &C[ (i % layers) ][0];                                  \
                                        T *output = &o[ (i % layers) ][0];                              \
                                        CACHE_FLUSH;                                                    \
                                        long long duration = f( *s, d1, d2, output, feature, Ti, Tk, chunk_size );  \
                                        cycle_counts.insert( duration );                                \
                                    }                                                                   \
                                    cycle_counts.process();                                             \
                                    cycle_counts.print();                                               \
                                    double median_time = (double) cycle_counts.median / CLOCK_FREQUENCY;\
                                    std::cout << "Median Time: " << median_time << std::endl;           \
                                    size_t flop_count = ((size_t)feature) * 2 * ((size_t)s->nnzs) + ((size_t) s->nnzs); \
                                    double gflops = ((double)flop_count) / median_time / 1E9;           \
                                    std::cout << "GFLOPS: " << gflops << std::endl;                     \
                                }

#endif // TRACK_PARALLEL_STATS

#define SDDMM_CORRECTNESS_CHECK(f,S)                                                                      \
                                {                                                                       \
                                    T *O_correct = generate_zeroes<T, ITYPE>( countO );                 \
                                    T *O_check = generate_zeroes<T, ITYPE>( countO );                   \
                                                                                                        \
                                    sddmm_simple( *S_csr, &B[0][0], &C[0][0], O_correct, feature );     \
                                    f( *S, &B[0][0], &C[0][0], O_check, feature, Ti, Tk, chunk_size );  \
                                                                                                        \
                                    auto correct = are_equal( O_correct, O_check, countO );             \
                                    std::cout << "Correctness check passed? " << (correct==0) << std::endl;  \
                                    std::free(O_correct);                                               \
                                    std::free(O_check);                                                 \
                                }


template <typename T, typename ITYPE>
void data_movement_experiment_sddmm( std::string mtx_filename, ITYPE feature, ITYPE Ti, ITYPE Tj, ITYPE Tk, ITYPE n, ITYPE num_threads = 1, ITYPE chunk_size = 1, ITYPE layers = 1, ITYPE num_panels = 0, ITYPE *num_panels_per_thread = nullptr, struct workitem *work_list = nullptr, ITYPE *panel_worklist = nullptr, ITYPE num_partitions = 2, std::vector<struct panel_t> *adaptive_panels = nullptr, ITYPE fixed_nnzs = 0 )
{
    // create sparse array
    ITYPE nrows, ncols, nnzs;
    std::pair<ITYPE, ITYPE> *locs = nullptr;
    T* vals = nullptr;
    bool using_global = false;

    CSR<T, ITYPE> *S_csr = nullptr;
    DCSC<T, ITYPE> *S_dcsc = nullptr;
    ATM<T, ITYPE> *S_atm = nullptr;
    CSF<T, ITYPE> *S_csf = nullptr;

    if ( global_locs && global_vals ) {
        locs = global_locs;
        vals = global_vals;
        nrows = global_nrows;
        ncols = global_ncols;
        nnzs = global_nnzs;
        using_global = true;
    } else {
        read_mtx_matrix_into_arrays(mtx_filename.c_str(), &locs, &vals, &nrows, &ncols, &nnzs);
        assert( locs && vals && (nnzs > 0) && (ncols > 0) && (nnzs > 0) && "Could not read mtx file" );
    }

    S_csr = new CSR<T, ITYPE>( nrows, ncols, nnzs, locs, vals );

    std::cout << "M: " << S_csr->nrows << std::endl;
    std::cout << "N: " << S_csr->ncols << std::endl;
    std::cout << "NNZ: " << S_csr->nnzs << std::endl;

    if (chunk_size == -1) {
        ITYPE num_panels = CEIL(S_csr->nrows, Ti);
        chunk_size = CEIL(num_panels, num_threads);
    }
    std::cout << "Chunk Size: " << chunk_size << std::endl;

    #if defined(RUN_DCSC_JSTREAM_EXPR) || defined(RUN_DCSC_WORKLIST) || defined(RUN_DCSC_JSTREAM_COMPILER_EXPR)
        S_dcsc = new DCSC<T, ITYPE>( nrows, ncols, nnzs, locs, vals, Ti );
    #endif

    #if defined(RUN_CSR_ATM_KSTREAM)
        // S_atm = new ATM<T, ITYPE>( nrows, ncols, nnzs, locs, vals, adaptive_panels->size(), *adaptive_panels );
        S_atm = new ATM<T, ITYPE>( nrows, ncols, nnzs, locs, vals, *adaptive_panels );
        bool atm_correct = verify_matrix_structure( *S_csr, *S_atm );
        std::cout << "ATM Matrix correct? " << atm_correct << std::endl;
    #endif

    #if defined(RUN_CSF_EXPR)
        // First generate the tiled CSF coordinate ordering, then call the constructor
        ITYPE *C1, *C2, *C3, *C4;
        ITYPE num_csf_panels = 0;

        if ( (*adaptive_panels).size() > 0 ) {
            print_status("[CSF] Building rassm csf matrix\n");
            num_csf_panels = generate_coo_representation_rassm(nnzs, locs, vals, *adaptive_panels, &C1, &C2, &C3, &C4);
        } else {
            if (fixed_nnzs > 0) {
                print_status("[CSF] Building fixed nnzs csf matrix\n");
                num_csf_panels = generate_fixed_nnzs_coo_representation(nnzs, locs, vals, fixed_nnzs, &C1, &C2, &C3, &C4);
            } else {
                print_status("[CSF] Building fixed size csf matrix\n");
                num_csf_panels = generate_coo_representation(Ti, Tj, nnzs, locs, vals, &C1, &C2, &C3, &C4);
            }
        }

        std::cout << "Unique Tile Rows: " << num_csf_panels << std::endl;

        S_csf = new CSF<T, ITYPE>( nrows, ncols, nnzs, num_csf_panels, C1, C2, C3, C4, vals );

        // verify the constructed matrix via a function call
        bool csf_correct = verify_matrix_structure( *S_csr, *S_csf );
        std::cout << "CSF Matrix correct? " << csf_correct << std::endl;
    #endif

    #if defined(RUN_CSR_KSTREAM_EXPR)
        S_stm = new STM<T, ITYPE>( nrows, ncols, nnzs, locs, vals, Ti, Tj );
    #endif

    #if defined(RUN_SPLIT_JSTREAM_EXPR)
        S_split = new SPLIT_CSR<T, ITYPE>( nrows, ncols, nnzs, locs, vals, num_partitions );
    #endif

    #if defined(RUN_DCSC_JSTREAM_EXPR) && defined(RUN_CSR_KSTREAM_EXPR)
        std::cout << "Matrices are correct?: " << verify_matrices(S_csr, S_dcsc, S_stm) << std::endl;
    #endif

    #if defined(RUN_HYB_EXPR) || defined(RUN_DYN_HYB_EXPR) || defined(RUN_DYN_HYB_DM_EXPR)
        DCSH<T, ITYPE> *S_dcsh = nullptr;
        // DCSH<T, ITYPE> **O_dcsh = nullptr;
        DCSH<T, ITYPE> *O_dcsh[layers];

        S_dcsh = new DCSH<T, ITYPE>( nrows, ncols, nnzs, locs, vals, num_panels, work_list );

        // O_dcsh = new DCSH<T, ITYPE>*[ layers ];
        for ( ITYPE i = 0; i < layers; i++ ) {
            O_dcsh[i] = new DCSH<T, ITYPE>( *S_dcsh );
        }
        // O_dcsh = new DCSH<T, ITYPE>( *S_dcsh );

        //  new DCSH<T, ITYPE>(nrows, ncols, nnz, locs, vals, num_panels, panel_type, panel_offset);
        ITYPE max_panels_per_thread = 0;
        for (ITYPE i = 0; i < num_threads; i++) {
            if (num_panels_per_thread[i] > max_panels_per_thread) {
                max_panels_per_thread = num_panels_per_thread[i];
            }
        }

        ITYPE temp_panel_Tk[num_panels];

        std::pair<ITYPE, ITYPE> *pairs_worklist = (std::pair<ITYPE, ITYPE> *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(std::pair<ITYPE, ITYPE>) * max_panels_per_thread * num_threads );
        for (ITYPE tid = 0; tid < num_threads; tid++) {
            for (ITYPE p = 0; p < num_panels_per_thread[tid]; p++) {
                ITYPE panel_id = panel_worklist[ p * num_threads + tid ];
                pairs_worklist[ p * num_threads + tid ] = { panel_id, work_list[panel_id].Tk };
                temp_panel_Tk[ panel_id ] = work_list[panel_id].Tk;
            }
        }

        S_dcsh->set_per_panel_Tk( temp_panel_Tk );

        std::cout << "DCSH Matrix correct? : " << verify_dcsh_matrix<T, ITYPE>( S_csr, S_dcsh ) << std::endl;
    #endif

    #ifdef TRACK_PARALLEL_STATS
        perf::init_papi(); // Initialize PAPI
        ITYPE temp;
        #ifndef TRACK_CACHE_ONLY
            perf *dram_perf;
            perf *reg_perf;
            perf *cache_perf;
            perf *stall_perf;
            perf *tlb_perf;
            perf *cache_stall_perf;
            perf *l3_cache_perf;

            perf *dram_precharge_rd_perf;
            perf *dram_precharge_wr_perf;
            perf *dram_activate_rd_perf;
            perf *dram_activate_wr_perf;

            stats_t<long long, ITYPE> **dram_counters;
            stats_t<long long, ITYPE> **reg_counters;
            stats_t<long long, ITYPE> **cache_counters;
            stats_t<long long, ITYPE> **stall_counters;
            stats_t<long long, ITYPE> **tlb_counters;
            stats_t<long long, ITYPE> **cache_stall_counters;
            stats_t<long long, ITYPE> **l3_cache_counters;

            stats_t<long long, ITYPE> **dram_precharge_rd_counters;
            stats_t<long long, ITYPE> **dram_precharge_wr_counters;

            stats_t<long long, ITYPE> **dram_activate_rd_counters;
            stats_t<long long, ITYPE> **dram_activate_wr_counters;

            #if defined(REG_STATS)
                setup_perf_counters_helper<T, ITYPE>(perf_event_set::REG_EVENTS, num_threads, &reg_perf, &reg_counters, &temp);
                RUN_PERF_RESET(reg_counters, perf::NUM_EVENTS_PER_SET);
            #elif defined(CACHE_STATS)
                setup_perf_counters_helper<T, ITYPE>(perf_event_set::CACHE_EVENTS, num_threads, &cache_perf, &cache_counters, &temp);
                // setup_native_perf_counters_helper<T, ITYPE>(perf_event_set::NAMED_CACHE_EVENTS, num_threads, &cache_perf, &cache_counters, &temp);
                RUN_PERF_RESET(cache_counters, perf::NUM_EVENTS_PER_SET);
            #elif defined(STALL_STATS)
                // setup_perf_counters_helper<T, ITYPE>(perf_event_set::STALL_EVENTS, num_threads, &stall_perf, &stall_counters, &temp);
                setup_native_perf_counters_helper<T, ITYPE>(perf_event_set::NAMED_STALL_EVENTS, num_threads, &stall_perf, &stall_counters, &temp);
                RUN_PERF_RESET(stall_counters, NUM_NAMED_STALL_EVENTS);
            #elif defined(DRAM_STATS)
                setup_dram_counters_helper<T, ITYPE>(num_threads, &dram_perf, &dram_counters, &temp);
                RUN_DRAM_RESET(dram_counters);
            #elif defined(CACHE_STALL_STATS)
                setup_native_perf_counters_helper<T, ITYPE>(perf_event_set::CACHE_STALL_EVENTS, num_threads, &cache_stall_perf, &cache_stall_counters, &temp);
                RUN_PERF_RESET(cache_stall_counters, NUM_CACHE_STALL_EVENTS);
            #elif defined(L3_CACHE_STATS)
                setup_native_perf_counters_helper<T, ITYPE>(perf_event_set::L3_CACHE_EVENTS, num_threads, &l3_cache_perf, &l3_cache_counters, &temp);
            #elif defined(TLB_STATS)
                setup_perf_counters_helper<T, ITYPE>(perf_event_set::TLB_EVENTS, num_threads, &tlb_perf, &tlb_counters, &temp);
                RUN_PERF_RESET(tlb_counters, perf::NUM_EVENTS_PER_SET);
            #elif defined(DRAM_PRECHARGE_RD_STATS)
                setup_native_perf_counters_helper<T, ITYPE>(perf_event_set::DRAM_PRECHARGE_RD_EVENTS, num_threads, &dram_precharge_rd_perf, &dram_precharge_rd_counters, &temp);
                RUN_DRAM_RESET(dram_precharge_rd_counters);
            #elif defined(DRAM_PRECHARGE_WR_STATS)
                setup_native_perf_counters_helper<T, ITYPE>(perf_event_set::DRAM_PRECHARGE_WR_EVENTS, num_threads, &dram_precharge_wr_perf, &dram_precharge_wr_counters, &temp);
                RUN_DRAM_RESET(dram_precharge_wr_counters);
            #elif defined(DRAM_ACTIVATE_RD_STATS)
                setup_native_perf_counters_helper<T, ITYPE>(perf_event_set::DRAM_ACTIVATE_RD_EVENTS, num_threads, &dram_activate_rd_perf, &dram_activate_rd_counters, &temp);
                RUN_DRAM_RESET(dram_activate_rd_counters);
            #elif defined(DRAM_ACTIVATE_WR_STATS)
                setup_native_perf_counters_helper<T, ITYPE>(perf_event_set::DRAM_ACTIVATE_WR_EVENTS, num_threads, &dram_activate_wr_perf, &dram_activate_wr_counters, &temp);
                RUN_DRAM_RESET(dram_activate_wr_counters);
            #endif

        #endif // TRACK_CACHE_ONLY
    #endif


    // input dense arrays
    size_t sizeB = ncols * (feature + PADDING_B) * sizeof(T);
    size_t sizeB_rounded = CEIL(sizeB, PAGE_SIZE) * PAGE_SIZE;
    size_t countB = CEIL(sizeB_rounded, sizeof(T));

    size_t sizeC = nrows * (feature + PADDING_C) * sizeof(T);
    size_t sizeC_rounded = CEIL(sizeC, PAGE_SIZE) * PAGE_SIZE;
    size_t countC = CEIL(sizeC_rounded, sizeof(T));

    size_t sizeO = nnzs * sizeof(T);
    size_t sizeO_rounded = CEIL(sizeO, PAGE_SIZE) * PAGE_SIZE;
    size_t countO = CEIL(sizeO_rounded, sizeof(T));

    #ifdef LARGE_ARRAY
        size_t countC_alloc = countC * (size_t) layers;
        size_t countB_alloc = countB * (size_t) layers;

        T *B = generate_dense<T, ITYPE>( countB_alloc );
        T *C = generate_zeroes<T, ITYPE>( countC_alloc );
    #else
        T *B[layers];
        T *C[layers];
        T *O[layers];

        for ( ITYPE l = 0; l < layers; l++ ) {
            B[l] = generate_dense<T, ITYPE>( countB );
            C[l] = generate_dense<T, ITYPE>( countC );
            O[l] = generate_zeroes<T, ITYPE>( countO );
        }
    #endif

    // #ifdef RUN_CORRECTNESS_CHECK
    //     T *O_correct = generate_zeroes<T, ITYPE>( countO );
    //     T *O_check = generate_zeroes<T, ITYPE>( countO );

    //     sddmm_simple( *S_csr, &B[0][0], &C[0][0], O_correct, feature );
    //     sddmm_rstream( *S_csr, &B[0][0], &C[0][0], O_check, feature, Ti, Tk, chunk_size );

    //     if ( are_equal( O_correct, O_check, countO ) == 0 ) {
    //         std::cout << "Correctness check passed" << std::endl;
    //     } else {
    //         std::cout << "Correctness check failed" << std::endl;
    //     }
    // #endif

    /*
    {
        // CSR<T, ITYPE> *gold
        DCSH<T, ITYPE> *verif_O = new DCSH<T, ITYPE>( *S_dcsh );

        CSR<T, ITYPE> *simple_O = new CSR<T, ITYPE>( *S_csr );
        CSR<T, ITYPE> *verif_O_rstream = new CSR<T, ITYPE>( *S_csr );
        CSR<T, ITYPE> *verif_O_simple_parallel = new CSR<T, ITYPE>( *S_csr );

        DCSC<T, ITYPE> *verif_O_dcsc = new DCSC<T, ITYPE>( *S_dcsc );

        assert( S_dcsc->is_equal_structure( *verif_O_dcsc ) && "structure of copy is not the same" );
        assert( verify_matrix_structure( *S_csr, *verif_O ) && "structure of dsch is not the same as csr" );

        sddmm_simple( *S_csr, &B[0][0], &C[0][0], *simple_O, feature );
        sddmm_simple_parallel( *S_csr, &B[0][0], &C[0][0], *verif_O_simple_parallel, feature, chunk_size );

        if ( are_equal( *simple_O, *verif_O_simple_parallel ) ) {
            std::cout << "Simple parallel has errors" << std::endl;
        } else {
            std::cout << "Simple parallel is correct" << std::endl;
        }


        std::cout << "Checking SDDMM r-stream" << std::endl;
        sddmm_rstream( *S_csr, &B[0][0], &C[0][0], *verif_O_rstream, feature, Ti, Tk, chunk_size );
        if ( are_equal( *simple_O, *verif_O_rstream ) ) {
            std::cout << "Something went wrong in rstream SDDMM function" << std::endl;
        } else {
            std::cout << "rstream SDDMM looks good" << std::endl;
        }

        sddmm_jstream( *S_dcsc &B[0][0], &C[0][0], *verif_O_dcsc, feature, Ti, Tk, chunk_size );
        std::cout << "Checking SDDMM j-stream" << std::endl;
        if ( are_equal( *simple_O, *verif_O_dcsc ) ) {
            std::cout << "Something went wrong in the jstream SDDMM function" << std::endl;
        } else {
            std::cout << "jstream SDDMM looks good" << std::endl;
        }

        std::cout << "Checking adaptive hybrid" << std::endl;
        sddmm_hybrid( *S_dcsh, &B[0][0], &C[0][0], *verif_O, feature, chunk_size );
        if ( are_equal( *simple_O, *verif_O ) ) {
            std::cout << "Something went wrong in the adaptive SDDMM function" << std::endl;
        } else {
            std::cout << "hybrid SDDMM looks good" << std::endl;
        }

        delete simple_O;
        delete verif_O_rstream;
        delete verif_O;
    */

    //
    #ifdef RUN_SIMPLE_CSR
        PRINT_TYPE("SIMPLE_CSR");
        #ifdef RUN_CORRECTNESS_CHECK
            SDDMM_CORRECTNESS_CHECK(sddmm_parallel, S_csr);
        #endif
        RUN_SDDMM_KERNEL_WARMUP(sddmm_parallel, S_csr, O);
        #ifdef TRACK_PARALLEL_STATS
            #if defined(CACHE_STATS)
                RUN_SDDMM_KERNEL(sddmm_parallel, S_csr, O, cache_perf, cache_counters, perf_event_set::CACHE_EVENTS);
            #elif defined(STALL_STATS)
                RUN_SDDMM_KERNEL(sddmm_parallel, S_csr, O, stall_perf, stall_counters, perf_event_set::NAMED_STALL_EVENTS);
            #elif defined(L3_CACHE_STATS)
                RUN_SDDMM_KERNEL(sddmm_parallel, S_csr, O, l3_cache_perf, l3_cache_counters, perf_event_set::L3_CACHE_EVENTS);
            #elif defined(REG_STATS)
                RUN_SDDMM_KERNEL(sddmm_parallel, S_csr, O, reg_perf, reg_counters, perf_event_set::REG_EVENTS);
            #endif
        #else
            RUN_SDDMM_KERNEL(sddmm_parallel, S_csr, O);
        #endif // TRACK_PARALLEL_STATS
    #endif

    #ifdef RUN_ROW_PANEL_COMPILER
        PRINT_TYPE("CSR COMPILER");
        #ifdef RUN_CORRECTNESS_CHECK
            SDDMM_CORRECTNESS_CHECK(sddmm_csr_row_panel_compiler_vectorized, S_csr);
        #endif
        RUN_SDDMM_KERNEL_WARMUP(sddmm_csr_row_panel_compiler_vectorized, S_csr, O);
        MEM_RESET(O, countO, layers);
        #ifdef TRACK_PARALLEL_STATS
            #if defined(CACHE_STATS)
                RUN_SDDMM_KERNEL(sddmm_csr_row_panel_compiler_vectorized, S_csr, O, cache_perf, cache_counters, perf_event_set::CACHE_EVENTS);
            #elif defined(STALL_STATS)
                RUN_SDDMM_KERNEL(sddmm_csr_row_panel_compiler_vectorized, S_csr, O, stall_perf, stall_counters, perf_event_set::NAMED_STALL_EVENTS);
            #elif defined(L3_CACHE_STATS)
                RUN_SDDMM_KERNEL(sddmm_csr_row_panel_compiler_vectorized, S_csr, O, l3_cache_perf, l3_cache_counters, perf_event_set::L3_CACHE_EVENTS);
            #elif defined(REG_STATS)
                RUN_SDDMM_KERNEL(sddmm_csr_row_panel_compiler_vectorized, S_csr, O, reg_perf, reg_counters, perf_event_set::REG_EVENTS);
            #endif
        #else
            RUN_SDDMM_KERNEL(sddmm_csr_row_panel_compiler_vectorized, S_csr, O);
        #endif // TRACK_PARALLEL_STATS
    #endif



    #ifdef RUN_CSR_JSTREAM_EXPR
        RUN_SDDMM_KERNEL_WARMUP(sddmm_rstream, S_csr, O);
        #ifdef TRACK_PARALLEL_STATS
            #if defined(CACHE_STATS)
                RUN_SDDMM_KERNEL(sddmm_rstream, S_csr, O, cache_perf, cache_counters, perf_event_set::CACHE_EVENTS);
            #elif defined(STALL_STATS)
                RUN_SDDMM_KERNEL(sddmm_rstream, S_csr, O, stall_perf, stall_counters, perf_event_set::NAMED_STALL_EVENTS);
            #elif defined(L3_CACHE_STATS)
                RUN_SDDMM_KERNEL(sddmm_rstream, S_csr, O, l3_cache_perf, l3_cache_counters, perf_event_set::L3_CACHE_EVENTS);
            #elif defined(REG_STATS)
                RUN_SDDMM_KERNEL(sddmm_rstream, S_csr, O, reg_perf, reg_counters, perf_event_set::REG_EVENTS);
            #endif
        #else
            RUN_SDDMM_KERNEL(sddmm_rstream, S_csr, O);
        #endif // TRACK_PARALLEL_STATS
    #endif

    #ifdef RUN_CSR_ATM_KSTREAM
        #ifdef RUN_CORRECTNESS_CHECK
            SDDMM_CORRECTNESS_CHECK(sddmm_kstream, S_atm);
        #endif

        RUN_SDDMM_KERNEL_WARMUP(sddmm_kstream, S_atm, O);
        MEM_RESET(O, countO, layers);
        #ifdef TRACK_PARALLEL_STATS
            #if defined(CACHE_STATS)
                RUN_SDDMM_KERNEL(sddmm_kstream, S_atm, O, cache_perf, cache_counters, perf_event_set::CACHE_EVENTS);
            #elif defined(STALL_STATS)
                RUN_SDDMM_KERNEL(sddmm_kstream, S_atm, O, stall_perf, stall_counters, perf_event_set::NAMED_STALL_EVENTS);
            #elif defined(L3_CACHE_STATS)
                RUN_SDDMM_KERNEL(sddmm_kstream, S_atm, O, l3_cache_perf, l3_cache_counters, perf_event_set::L3_CACHE_EVENTS);
            #elif defined(REG_STATS)
                RUN_SDDMM_KERNEL(sddmm_kstream, S_atm, O, reg_perf, reg_counters, perf_event_set::REG_EVENTS);
            #endif
        #else
            RUN_SDDMM_KERNEL(sddmm_kstream, S_atm, O);
        #endif // TRACK_PARALLEL_STATS
    #endif


    #ifdef RUN_CSF_EXPR
        #ifdef RUN_CORRECTNESS_CHECK
            SDDMM_CORRECTNESS_CHECK(sddmm_csf, S_csf);
        #endif
        RUN_SDDMM_KERNEL_WARMUP(sddmm_csf, S_csf, O);
        MEM_RESET(O, countO, layers);
        #ifdef TRACK_PARALLEL_STATS
            #if defined(CACHE_STATS)
                RUN_SDDMM_KERNEL(sddmm_csf, S_csf, O, cache_perf, cache_counters, perf_event_set::CACHE_EVENTS);
            #elif defined(STALL_STATS)
                RUN_SDDMM_KERNEL(sddmm_csf, S_csf, O, stall_perf, stall_counters, perf_event_set::NAMED_STALL_EVENTS);
            #elif defined(L3_CACHE_STATS)
                RUN_SDDMM_KERNEL(sddmm_csf, S_csf, O, l3_cache_perf, l3_cache_counters, perf_event_set::L3_CACHE_EVENTS);
            #elif defined(REG_STATS)
                RUN_SDDMM_KERNEL(sddmm_csf, S_csf, O, reg_perf, reg_counters, perf_event_set::REG_EVENTS);
            #endif
        #else
            RUN_SDDMM_KERNEL(sddmm_csf, S_csf, O);
        #endif // TRACK_PARALLEL_STATS

    #endif // RUN_CSF_EXPR


    #ifdef RUN_DCSC_JSTREAM_EXPR
        #ifdef RUN_CORRECTNESS_CHECK
            SDDMM_CORRECTNESS_CHECK(sddmm_jstream, S_dcsc);
        #endif

        RUN_SDDMM_KERNEL_WARMUP(sddmm_jstream, S_dcsc, O);
        MEM_RESET(O, countO, layers);
        #ifdef TRACK_PARALLEL_STATS
            #if defined(CACHE_STATS)
                RUN_SDDMM_KERNEL(sddmm_jstream, S_dcsc, O, cache_perf, cache_counters, perf_event_set::CACHE_EVENTS);
            #elif defined(STALL_STATS)
                RUN_SDDMM_KERNEL(sddmm_jstream, S_dcsc, O, stall_perf, stall_counters, perf_event_set::NAMED_STALL_EVENTS);
            #elif defined(L3_CACHE_STATS)
                RUN_SDDMM_KERNEL(sddmm_jstream, S_dcsc, O, l3_cache_perf, l3_cache_counters, perf_event_set::L3_CACHE_EVENTS);
            #elif defined(REG_STATS)
                RUN_SDDMM_KERNEL(sddmm_jstream, S_dcsc, O, reg_perf, reg_counters, perf_event_set::REG_EVENTS);
                // RUN_SDDMM_KERNEL(sddmm_jstream, S_dcsc, O, l3_cache_perf, l3_cache_counters, perf_event_set::L3_CACHE_EVENTS);
            #endif
        #else
            RUN_SDDMM_KERNEL(sddmm_jstream, S_dcsc, O);
        #endif // TRACK_PARALLEL_STATS
    #endif

    #ifdef RUN_DYN_HYB_EXPR
        RUN_SDDMM_KERNEL_WARMUP(sddmm_hybrid, S_dcsh, O);
        #ifdef TRACK_PARALLEL_STATS
            #if defined(CACHE_STATS)
                RUN_SDDMM_KERNEL(sddmm_hybrid, S_dcsh, O, cache_perf, cache_counters, perf_event_set::CACHE_EVENTS);
            #elif defined(STALL_STATS)
                RUN_SDDMM_KERNEL(sddmm_hybrid, S_dcsh, O, stall_perf, stall_counters, perf_event_set::NAMED_STALL_EVENTS);
            #elif defined(L3_CACHE_STATS)
                RUN_SDDMM_KERNEL(sddmm_hybrid, S_dcsh, O, l3_cache_perf, l3_cache_counters, perf_event_set::L3_CACHE_EVENTS);
            #endif
        #else
            RUN_SDDMM_KERNEL(sddmm_hybrid, S_dcsh, O);
        #endif // TRACK_PARALLEL_STATS
    #endif

    #ifdef LARGE_ARRAY
        release_memory(B);
        release_memory(C);
    #else
        for (ITYPE l = 0; l < layers; l++) {
            release_memory(B[l]);
            release_memory(C[l]);
            release_memory(O[l]);
        }
    #endif


    if (!using_global) {
        if (locs) { delete[] locs; }
        if (vals) { delete[] vals; }
    }
    if (S_csr) { delete S_csr; }

}

#endif // EXPERIMENTS_H

