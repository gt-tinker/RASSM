#ifndef CONFIG_H
#define CONFIG_H

#include <cstdint>
#include <cstddef>

using ITYPE = int32_t;
// using ITYPE = int64_t;
using TYPE = double;

#if defined(__x86_64__) || defined(__i386__)
    #ifdef INTEL_COMPILER
        #include <immintrin.h>

        /* Instruction definition for manually vectorized code */
        // AVX512 instructions
        #define rtype512 __m512d
        #define vload512 _mm512_load_pd
        #define vstore512 _mm512_store_pd
        #define vfma512 _mm512_fmadd_pd
        #define vset512 _mm512_set1_pd

        // AVX2 instructions
        #define rtype __m256d
        #define vload _mm256_loadu_pd

        // #define vstore _mm256_store_pd
        #define vstore _mm256_storeu_pd
        #define vstore_nt _mm256_stream_pd

        #define vfma _mm256_fmadd_pd
        #define vadd(a,b) _mm256_add_pd(a,b)
        #define vset _mm256_set1_pd
        #define vsetzero _mm256_setzero_pd
        #define vhreduce(a)\
        a=_mm256_hadd_pd(a,a); \
        a=_mm256_hadd_pd(a,a);
        // #define vnop _mm256_blend_pd
        #define vnop _mm256_fmadd_pd

        #define lfence _mm_lfence
        #define mfence _mm_mfence
        #define rdtsc __rdtsc

        inline double hsum_double_avx(__m256d v) {
            __m128d vlow  = _mm256_castpd256_pd128(v);
            __m128d vhigh = _mm256_extractf128_pd(v, 1); // high 128
                    vlow  = _mm_add_pd(vlow, vhigh);     // reduce down to 128

            __m128d high64 = _mm_unpackhi_pd(vlow, vlow);
            return  _mm_cvtsd_f64(_mm_add_sd(vlow, high64));  // reduce to scalar
        }

    #else // not using an intel compiler try some hacks
        #include <x86intrin.h>
        #include <immintrin.h>

        // Instruction definition for manually vectorized code
        #define rtype __m256d
        #define vfma _mm256_fmadd_pd

        #define lfence __builtin_ia32_lfence
        #define mfence __builtin_ia32_mfence
        #define rdtsc  __builtin_ia32_rdtsc
    #endif
#else
    #define mfence()
    #define lfence()
#endif

// #define DEBUG
// #define DEBUG_BITSET
// #define DEBUG_STM
// #define DEBUG_RESIDUE
// #define DEBUG_SPARSE_MATRIX
// #define DEBUG_RESIDUE_COMBINATION

#define DEFAULT_CACHE_FLUSH_SIZE 1024 * 1024 * 8 // 4 Mi
#define CACHE_BLOCK_SIZE 64

// #define CACHE_NUM_WAYS 8
extern ITYPE CACHE_NUM_WAYS;

#define PAGE_SIZE ((size_t)(4 * 1024))  // 4 KiB
// #define PAGE_SIZE ((size_t) (128 * 1024 * 2))

#define DEFAULT_LLC 1024 * 1024
#define CREATE_MULTI_OUTPUT_THRESHOLD 10
#define ALIGNED_ACCESS 64

#define WORD_SIZE 8 // 8 bytes in a word

#define ALLOC_ALIGNMENT 64
#define WARMUP_DIVIDER 6

// #define PADDING_B 8
// #define PADDING_C 8

#define PADDING_B 0
#define PADDING_C 0

#define SPECIAL_THRESHOLD 128   // MAGIC NUMBER -- BECAUSE WHO DOESN'T LOVE SOME MAGIC
#define BIN_SEARCH_THRESHOLD 128

// #define LARGE_ARRAY      // uncomment this to allocate a single large array of layers * matrix size

// #define CLOCK_FREQUENCY ((double) (2.2E9))
#define CLOCK_FREQUENCY ((double) (2.6E9))

/////////////////////// System memory configuration ///////////////////////
const size_t MEMORY_CAPACITY = ((size_t)128 * (size_t)1024 * (size_t)1024 * (size_t)1024); // 128 GiB

extern double alpha_val;    // latency impact of the MAC operation
extern double beta_val;     // latency impact of the memory operation
extern bool global_debug;


// using ITYPE = int32_t;
// using TYPE = float;



/////////////////////// Runtime configurations for which methods to call ///////////////////////

// Runtime configurations for experiments using hardware performance counters
// #define RUN_BAND_MATRIX_EXPERIMENT


// #define GENERATE_RESIDUE_PROPERTIES
// #define GENERATE_RESIDUE_STATS
// #define DEBUG_RESIDUE_CONSTRUCTION



// #define RUN_CHARACTERIZATION
#define RUN_RESIDUE_GENERATED_TILE_SIZE // Generate tile sizes using the residue matrix model

// #define RUN_PINTOOL_MEMORY_TRACING

#define RUN_ASPT_SPECIAL
// #define RUN_ASPT_SPECIAL_DENSE_OPT
#define RUN_ADAPTIVE_2D_TILING
// #define ASPT_SPECIAL_SIMD_PARALLEL

// #define RUN_CSF_CHARACTERIZATION

#define DATA_MOVEMENT_EXPERIMENT


// #define PRINT_HYBRID_PANEL_CONSTRUCTION

// #define COMPUTE_ONLY_LATENCY
// #define PER_PANEL_PERF_EVENTS
// #define TRACK_CACHE_ONLY
// #define TRACK_STATS_NEW

// #define DEBUG_PRINT_PAPI_STATS

// #define OMP_SCHEDULE static
#define OMP_SCHEDULE dynamic

// #define TRACK_PER_CORE_RUNTIME // use this to capture per core runtime -- i.e. track load imbalance across the machine
// #define TRACK_PER_PANEL_RUNTIME // track per panel runtime to find what is going on

// #define CACHE_STATS

// #define VALIDATE_PERF_COUNTERS  // validate perf counters

// Runtime configurations for runtime experiments on input mtx matrices
#ifdef INTEL_COMPILER
    // #define RUN_CSR_BASELINE
    // #define RUN_CSR_JSTREAM
    // #define RUN_CSR_STM
    // #define RUN_CSC_STM
    // #define RUN_DCSC_JSTREAM
#endif

#define ONLY_PERF

// data structures for configuration and such
template<typename ITYPE, typename TYPE>
struct SpMM_Config {
    static constexpr ITYPE kBlockWidth = 32 * 8;
};

#endif // CONFIG_H
