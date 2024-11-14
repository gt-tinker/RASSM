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
    #include "spmm/taco.h"

    #include "sddmm/tiled.h"
    #include "sddmm/simple.h"
#else   // Might be using a compiler that does not recoganize Intel intrinsics
    #include "spmm/jstream.h"
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

// #define RUN_CSF_EXPR


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


#define MEM_RESET(arr, size, layers) reset_matrix_helper<T, ITYPE>(arr, size, layers)

#pragma intel optimization_level 0
void reset_matrix_helper_slow(double *arr, size_t size, int num_threads = 20)
{
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (size_t i = 0; i < size; i++) {
        arr[i] = (double) 0;
    }

}


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


// f - spmm function to call
// A - The sparse matrix pointer
// Ti - Row panel height
// Tk - Tk
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

#define PRINT_TYPE(T) std::cout << "T: " << T << std::endl;

#define LARGE_ARRAY_SIZE ( 1024 * 1024 * 1024 / sizeof(double) )
#define ERROR_THRESHOLD 0.03    // We can tolerate a 3% error

template <typename T, typename ITYPE>
void data_movement_experiment( std::string mtx_filename, ITYPE feature, ITYPE Ti, ITYPE Tj, ITYPE Tk, ITYPE n, ITYPE num_threads = 1, ITYPE chunk_size = 1, ITYPE layers = 1, ITYPE num_panels = 0, ITYPE *num_panels_per_thread = nullptr, struct workitem *work_list = nullptr, ITYPE *panel_worklist = nullptr, ITYPE num_partitions = 2, std::vector<struct panel_t> *adaptive_panels = nullptr, ITYPE fixed_nnzs = 0, runtype run_mode = runtype::RASSM )
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

    std::cout << "M: " << S_csr->nrows << std::endl;
    std::cout << "N: " << S_csr->ncols << std::endl;
    std::cout << "NNZ: " << S_csr->nnzs << std::endl;

    if (chunk_size == -1) {
        ITYPE num_panels = CEIL(S_csr->nrows, Ti);
        chunk_size = CEIL(num_panels, num_threads);
    }
    print_status("Chunk size for openmp: %d\n", chunk_size);
    // std::cout << "Chunk Size: " << chunk_size << std::endl;

    if (run_mode == runtype::JSTREAM) {
        S_dcsc = new DCSC<T, ITYPE>( nrows, ncols, nnzs, locs, vals, Ti );
        print_status("Matrix structure verified?: %d\n", verify_matrices( *S_csr, *S_dcsc ));
    } else if (run_mode == runtype::CSR_32) {

    } else if (run_mode == runtype::CSF_US || run_mode == runtype::CSF_UO) {

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

    } else if (run_mode == runtype::RASSM) {
        S_atm = new ATM<T, ITYPE>( nrows, ncols, nnzs, locs, vals, *adaptive_panels );
        auto atm_correct = verify_matrix_structure( *S_csr, *S_atm );
        std::cout << "ATM Matrix correct? " << atm_correct << std::endl;
    } else {

    }


    size_t sizeB = ncols * (feature + PADDING_B) * sizeof(T);
    size_t sizeB_rounded = CEIL(sizeB, PAGE_SIZE) * PAGE_SIZE;
    size_t countB = CEIL(sizeB_rounded, sizeof(T));

    size_t sizeC = nrows * (feature + PADDING_C) * sizeof(T);
    size_t sizeC_rounded = CEIL(sizeC, PAGE_SIZE) * PAGE_SIZE;
    size_t countC = CEIL(sizeC_rounded, sizeof(T));


    T *B[layers];
    T *C[layers];

    for ( ITYPE l = 0; l < layers; l++ ) {
        B[l] = generate_dense<T, ITYPE>( countB );
        C[l] = generate_zeroes<T, ITYPE>( countC );
    }



    if (run_mode == runtype::CSR_32) {
        PRINT_TYPE("CSR_ROW_PANEL_COMPILER");

        RUN_CHECK(csr_row_panel_compiler_vectorized, S_csr, Ti, Tk);

        RUN_WARMUP_EXPR(csr_row_panel_compiler_vectorized, S_csr, Ti, Tk);

        MEM_RESET(C, countC, layers);

        RUN_TIMING(csr_row_panel_compiler_vectorized, S_csr, Ti, Tk);
    } else if (run_mode == runtype::JSTREAM) {
        PRINT_TYPE("DCSC_JSTREAM");

        RUN_CHECK(spmm_dcsc_jstream, S_dcsc, Ti, Tk);

        RUN_WARMUP_EXPR(spmm_dcsc_jstream, S_dcsc, Ti, Tk);

        MEM_RESET(C, countC, layers);

        RUN_TIMING(spmm_dcsc_jstream, S_dcsc, Ti, Tk);
    } else if (run_mode == runtype::CSF_US || run_mode == runtype::CSF_UO) {
        PRINT_TYPE("CSF_KSTREAM");

        RUN_CHECK(spmm_csf_compiler_vectorized, S_csf, Ti, Tj);

        RUN_WARMUP_EXPR(spmm_csf_compiler_vectorized, S_csf, Ti, Tj);

        MEM_RESET(C, countC, layers);

        RUN_TIMING(spmm_csf_compiler_vectorized, S_csf, Ti, Tj);
    } else if (run_mode == runtype::RASSM) {
        PRINT_TYPE("CSR_ATM_KSTREAM");
        Tj = Tk;

        RUN_CHECK(spmm_atm_kstream_compiler_vectorized, S_atm, Ti, Tj);

        RUN_WARMUP_EXPR(spmm_atm_kstream_compiler_vectorized, S_atm, Ti, Tj);

        MEM_RESET(C, countC, layers);

        RUN_TIMING(spmm_atm_kstream_compiler_vectorized, S_atm, Ti, Tj);
    } else {

    }


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

        RUN_WARMUP_EXPR(spmm_dcsc_jstream, S_dcsc, Ti, Tk);

        MEM_RESET(C, countC, layers);

        RUN_TIMING(spmm_dcsc_jstream, S_dcsc, Ti, Tk);
    #endif // RUN_DCSC_JSTREAM_EXPR




    #if defined(RUN_CSR_ATM_KSTREAM)
        PRINT_TYPE("CSR_ATM_KSTREAM");
        Tj = Tk;

        RUN_CHECK(spmm_atm_kstream_compiler_vectorized, S_atm, Ti, Tj);

        RUN_WARMUP_EXPR(spmm_atm_kstream_compiler_vectorized, S_atm, Ti, Tj);

        MEM_RESET(C, countC, layers);

        RUN_TIMING(spmm_atm_kstream_compiler_vectorized, S_atm, Ti, Tj);
    #endif // RUN_CSR_KSTREAM_EXPR



    #if defined(RUN_CSF_EXPR)
        PRINT_TYPE("CSF_KSTREAM");

        RUN_CHECK(spmm_csf_compiler_vectorized, S_csf, Ti, Tj);

        RUN_WARMUP_EXPR(spmm_csf_compiler_vectorized, S_csf, Ti, Tj);

        MEM_RESET(C, countC, layers);

        RUN_TIMING(spmm_csf_compiler_vectorized, S_csf, Ti, Tj);
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
    }
    #endif // INTEL_MKL

    // Free all allocated memory
    for (ITYPE l = 0; l < layers; l++) {
        release_memory(B[l]);
        release_memory(C[l]);
    }

    if (S_dcsc) { delete S_dcsc; }
    if (S_csr) { delete S_csr; }
    if (S_csf) { delete S_csf; }
    if (S_atm) { delete S_atm; }

    if (!using_global) {
        if (locs) { delete[] locs; }
        if (vals) { delete[] vals; }
    }

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

