#ifndef UTIL_H
#define UTIL_H

#include <cassert>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

#include <sys/time.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>

#ifdef INTEL_COMPILER
    #include <immintrin.h> // For rdtsc intrinsic
#endif

#include "config.h"

extern TYPE *A;
extern TYPE *B;

#define ALLOC_ALIGNED_MEMORY
#ifdef ALLOC_ALIGNED_MEMORY
    #define ALLOC_MEMORY(size) std::aligned_alloc(ALLOC_ALIGNMENT, size)
#else
    #define ALLOC_MEMORY(size) std::malloc(size)
#endif

#define FREE_MEMORY(ptr) if (ptr) { std::free(ptr); }

#define RELEASE_MEMORY(var) if(var) { delete var; } var = NULL;
#define RELEASE_MEMORY_ARR(var) if(var) { delete[] var; } var = NULL;

#define MIN(a, b) (a <= b ? a : b)
#define MAX(a, b) ( (a >= b) ? a : b)
#define CEIL(a,b) (((a)+(b)-1)/(b))
#define FLOOR(a, b) ((a) / (b))

#define KIB 1024
#define MIB 1024 * 1024
#define GIB 1024 * 1024 * 1024

// For debugging
struct v_struct {
    ITYPE row, col, grp;
    TYPE val;

    ITYPE col_grp; // For STM
};

template<typename T, typename ITYPE>
void print_arr(T* arr, ITYPE size, std::string name = "", int flag = 0)
{
    std::ostream pstream( (flag ? std::cout.rdbuf() : std::cerr.rdbuf()) );

    if (name.size() > 0) { pstream << name << ": "; }
    pstream << "[";
    for (ITYPE i = 0; i < size; i++) {
        pstream << arr[i] << (i < (size-1) ? ", " : "");
    }
    pstream << "]" << std::endl;
    // pstream << "]";
}

template <typename T, typename ITYPE>
T* read_arr(std::string filename, ITYPE num_entries)
{
    std::ifstream istream( filename );

    T* arr = new T[ num_entries ];
    T temp;
    ITYPE size = 0;
    while (istream >> temp) {

        assert( temp < num_entries && "something in the row ordering is not correct");

        arr[size++] = temp;
    }

    assert( size == num_entries );

    istream.close();

    return arr;
}

inline void print_error_exit(const char *msg, ...)
{
    char temp[1024];
    va_list argptr;
    va_start(argptr, msg);
    vsnprintf( temp, sizeof(temp), msg, argptr );
    va_end(argptr);

    fprintf(stderr, "[ERROR]: %s", temp);
    exit(EXIT_FAILURE);
}

inline void print_debug(const char *msg, ...)
{
    char temp[1024];
    va_list argptr;
    va_start(argptr, msg);
    vsnprintf( temp, sizeof(temp), msg, argptr );
    va_end(argptr);

    fprintf(stderr, "[DEBUG]: %s", temp);
}

inline void print_status(const char *msg, ...)
{
    char temp[1024];
    va_list argptr;
    va_start(argptr, msg);
    vsnprintf( temp, sizeof(temp), msg, argptr );
    va_end(argptr);

    fprintf(stderr, "[STATUS]: %s", temp);
}

inline void print_debug_temp(const struct v_struct *t)
{
    std::cerr << "r: " << t->row << " c: " << t->col << " g: " << t->grp << " cg: " << t->col_grp << std::endl;
}

inline long get_time()
{
    timeval t;
    gettimeofday(&t, NULL);
    return ((t.tv_sec * 1000) + (t.tv_usec / 1000));
}

// Intel intrinsic for reading the timestamp counter
inline unsigned long long readTSC()
{
    unsigned long long tsc;
    // #ifdef INTEL_COMPILER
    #if defined(__x86_64__) || defined(__i386__)
        // _mm_lfence();  // optionally wait for earlier insns to retire before reading the clock

        mfence();
        lfence();

        // _mm_mfence();
        // _mm_lfence();
        // tsc = __rdtsc();
        tsc = rdtsc();
        // return __rdtscp();
        // _mm_lfence();  // optionally block later instructions until rdtsc retires
        lfence();
        // _mm_mfence();
    #else
        asm volatile("mrs %0, cntvct_el0" : "=r" (tsc));
    #endif

    return tsc;
}

inline __attribute__((always_inline)) unsigned long long rdtscp()
{
    unsigned long long tsc;
    #ifdef INTEL_COMPILER
        unsigned long a, d, c;

        __asm__ volatile("rdtscp" : "=a" (a), "=d" (d), "=c" (c));

        tsc = (a | (d << 32));
    #else
        asm volatile("mrs %0, cntvct_el0" : "=r" (tsc));
    #endif
    return tsc;
}

inline bool comp_double(double A, double B, double eps = 0.005)
{
    A = A < 0 ? (A * -1) : A;
    B = B < 0 ? (B * -1) : B;
    double max = A > B ? A : B;
    double diff = A > B ? (A - B) : (B - A);
    return (diff/max) < eps;
}

struct comp {
    template<typename ITYPE>
    bool operator() (const ITYPE &l, const ITYPE &r) const
    {
        if (l.first == r.first) {
            return l.second < r.second;
        }
        return l.first < r.first;
    }
};


// For sorting in row major format
int compare1(const void *a, const void *b);

// For sorting in column major format
int compare2(const void *a, const void *b);

template<typename ITYPE>
bool compare_pair_col(const std::pair<ITYPE, ITYPE> &a,
              const std::pair<ITYPE, ITYPE> &b)
{
    return (a.second < b.second);
}

template <typename T>
T find_median(T *arr, size_t size)
{
    std::sort(arr, arr + size);
    T median;
    if (size & 0x1) {
        median = arr[ size / 2 ];
    } else {
        median = (arr[ (size-1) / 2 ] + arr[ size / 2 ]) / 2;
    }
    return median;
}

template <typename T>
T find_mean(T *arr, size_t size)
{
    T mean = 0.0;
    for (size_t i = 0; i < size; i++) {
        mean += arr[i];
    }
    return mean / size;
}

template <typename T>
T find_std_dev( T* arr, size_t size, T mean )
{
    T x = 0;
    for ( size_t i = 0; i < size; i++ ) {
        x += ( arr[i] - mean ) * ( arr[i] - mean );
    }
    x = x / size;

    return std::sqrt(x);
}

template<typename T, typename ITYPE>
void set_zero(T *arr, ITYPE size)
{
    #pragma omp parallel for schedule(static)
    for ( ITYPE i = 0; i < size; i++ ) {
        arr[i] = 0;
    }
}

// For flushing the caches with an array of 'size' bytes
void cache_flush(ITYPE num_threads = 1);
void init_cache_flush(ITYPE size = DEFAULT_CACHE_FLUSH_SIZE);
void free_cache_flush();

// For tracing via a pintool
void __attribute__((noinline)) PIN_ROI_BEGIN();
void __attribute__((noinline)) PIN_ROI_END();

#endif // UTIL_H
