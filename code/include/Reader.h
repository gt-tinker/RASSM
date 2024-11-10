#ifndef READER_H
#define READER_H

#include <cassert>
#include <cstring>
#include <fstream>
#include <random>

#include "matrices/ATM.h"
#include "matrices/CSR.h"
#include "matrices/CSC.h"
#include "matrices/CSF.h"
#include "matrices/DCSC.h"
#include "matrices/Matrix.h"


#define LIMIT 10

// #define DEBUG_MTX_READER

template <typename T, typename ITYPE>
void read_mtx_matrix_into_arrays(const char *mtx_file_name, std::pair<ITYPE, ITYPE> **locs, T **vals,
        ITYPE *nr, ITYPE *nc, ITYPE *nz)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    *locs = NULL;
    *vals = NULL;

    std::ifstream mtx_file(mtx_file_name);
    char line[256];

    bool has_vals = true;
    bool is_complex = false;
    bool is_symmetric = false;

    // Read the first line and process header
    mtx_file.getline(line, 256);
    std::string head = std::string(line);
    if ( head.find("pattern") != std::string::npos ) {
        has_vals = false;
    }

    if ( head.find("complex") != std::string::npos ) {
        is_complex = true;
        has_vals = false; // Make has vals false -- only 1 of complex or vals possible
    }

    if ( head.find("symmetric") != std::string::npos ) {
        is_symmetric = true;
    }

    print_status("Matrix header: is_symmetric %d, is_complex %d, is_pattern %d\n", is_symmetric, is_complex, !has_vals);

    char c = mtx_file.get();
    while (c == '%') {
        mtx_file.getline(line, 256);
        c = mtx_file.get();
    }
    mtx_file.unget();

    // Read matrix metadata
    ITYPE nrows = 0, ncols = 0, nnz = 0;
    mtx_file >> nrows >> ncols >> nnz;

    // Read into buffers -- assume (row, col) pairs
    if (is_symmetric) {
        nnz = nnz * 2; // allocate twice the space for symmetric matrices
    }
    print_status("Matrix header: nrows %d, ncols %d, nnzs %d\n", nrows, ncols, nnz);

    *nr = nrows; *nc = ncols; *nz = nnz;

    *locs = new std::pair<ITYPE, ITYPE>[nnz]();
    *vals = new T[nnz]();

    if (! (*locs) || ! (*vals) ) {
        print_error_exit("Could not allocate memory\n");
    }

    ITYPE i = 0;
    while (!mtx_file.eof() && i < nnz) {
        ITYPE t_row = -1;
        ITYPE t_col = -1;
        T t_val = -1;
        if (has_vals) {
            mtx_file >> t_row >> t_col >> t_val;
        } else if (is_complex) {
            double temp1, temp2;
            mtx_file >> t_row >> t_col >> temp1 >> temp2;
        } else {
            mtx_file >> t_row >> t_col;
        }

        // if we didn't read anything in this lot, break;
        if (t_row == -1 || t_col == -1) { break; }

        (*locs)[i].first = t_row;
        (*locs)[i].second = t_col;
        (*locs)[i].first--;
        (*locs)[i].second--;
        (*vals)[i] = t_val != -1 ? t_val : ((T) (rand()%1048576))/ ((T)1048576) ;

        if ( is_symmetric && (*locs)[i].first != (*locs)[i].second ) {
            (*locs)[i + 1].first = (*locs)[i].second;
            (*locs)[i + 1].second = (*locs)[i].first;
            (*vals)[i + 1] = (*vals)[i];
            i++;
        }

        #ifdef DEBUG_MTX_READER
            std::cerr << i << ": " << (*locs)[i].first << " " << (*locs)[i].second << std::endl;
        #endif
        i++;
    }
    if (i > nnz) { print_error_exit("Error reading matrix file\n"); }

    // make sure to update the returned nnz count
    *nz = i;

    // close the matrix file
    mtx_file.close();

    auto end_time = std::chrono::high_resolution_clock::now();
    print_status( "MTX read from file -- M: %ld, N: %ld, NNZ: %ld -- time: %f\n", nrows, ncols, nnz, std::chrono::duration<double>(end_time - start_time).count() );
}

template <typename T, typename ITYPE>
void read_smtx_matrix_into_arrays(const char *mtx_file_name, std::pair<ITYPE, ITYPE> **locs, T **vals, ITYPE *nr, ITYPE *nc, ITYPE *nz)
{
    auto start_time = std::chrono::high_resolution_clock::now();

    *locs = NULL;
    *vals = NULL;

    std::ifstream mtx_file(mtx_file_name);
    char line[256];

    // read the first line -- rows cols nnzs
    ITYPE nrows = 0, ncols = 0, nnz = 0;
    mtx_file >> (*nr) >> (*nc) >> (*nz);

    *locs = new std::pair<ITYPE, ITYPE>[*nz]();
    *vals = new T[*nz]();

    if (! (*locs) || ! (*vals) ) {
        print_error_exit("Could not allocate memory\n");
    }

    ITYPE i = 0;
    ITYPE t_row = 0;
    while (!mtx_file.eof() && i < *nz) {
        ITYPE t_col = 0;
        mtx_file >> t_col;
        if (t_col == 0) {
            t_row++;    // increment the row -- wrap around
        }

        T t_val = ((T) (i % 100)) / 100;

        (*locs)[i].first = t_row;
        (*locs)[i].second = t_col;
        (*vals)[i] = t_val;

        #ifdef DEBUG_MTX_READER
            std::cerr << i << ": " << (*locs)[i].first << " " << (*locs)[i].second << std::endl;
        #endif
        i++;
    }

    assert(i == *nz);

    // close the matrix file
    mtx_file.close();
    auto end_time = std::chrono::high_resolution_clock::now();
    print_status( "S-MTX read from file -- M: %ld, N: %ld, NNZ: %ld -- time: %f\n", nrows, ncols, nnz, std::chrono::duration<double>(end_time - start_time).count() );
}

template <typename T, typename ITYPE>
CSR<T, ITYPE> *read_mtx_matrix(const char *mtx_file_name, bool inplace = false)
{
    ITYPE nrow, ncol, nnz;
    std::pair<ITYPE, ITYPE> *locs;
    T* vals;
    read_mtx_matrix_into_arrays(mtx_file_name, &locs, &vals, &nrow, &ncol, &nnz);

    // Call constructor for the CSR
    CSR<T, ITYPE> *spm = new CSR<T, ITYPE>(nrow, ncol, nnz, locs, vals, inplace);

    if (locs) { delete[] locs; }

    if (vals) { delete[] vals; }

    return spm;
}

template<typename T, typename ITYPE>
void parse_matrix_header(const char *mtx_file_name, ITYPE *nrows, ITYPE *ncols, ITYPE *nnzs)
{
    std::ifstream mtx_file(mtx_file_name);
    char line[256];

    // Read the first line and process header
    mtx_file.getline(line, 256);
    std::string head = std::string(line);

    char c = mtx_file.get();
    while (c == '%') {
        mtx_file.getline(line, 256);
        c = mtx_file.get();
    }
    mtx_file.unget();

    // Read the matrix size, etc
    mtx_file >> *nrows >> *ncols >> *nnzs;
    mtx_file.close();
}

template<typename T, typename ITYPE>
CSC<T, ITYPE> *read_mtx_matrix_csc(const char *mtx_file_name)
{
    ITYPE nrow, ncol, nnz;
    std::pair<ITYPE, ITYPE> *locs;
    T *vals;

    read_mtx_matrix_into_arrays(mtx_file_name, &locs, &vals, &nrow, &ncol, &nnz);
    CSC<T, ITYPE> *csc = new CSC<T, ITYPE>(nrow, ncol, nnz, locs, vals);

    if (locs) { delete[] locs; }
    if (vals) { delete[] vals; }

    return csc;
}

template <typename T, typename ITYPE>
std::pair<std::pair<ITYPE, ITYPE> *, T *> generate_raw_band_matrix( ITYPE nrows, ITYPE ncols, ITYPE b, ITYPE *nnzs, double sparsity = 1.0 )
{
    std::vector<std::pair<ITYPE, ITYPE>> locs;
    std::vector<T> vals;
    ITYPE nnz_count = 0;
    double upper_bound = nrows * ncols;
    double lower_bound = 0.0;
    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    std::default_random_engine re;

    std::uniform_real_distribution<double> coin_toss(0, 1.0);
    std::default_random_engine coin_toss_re;

    for ( ITYPE i = 0; i < nrows; i++ ) {
        ITYPE y_lower_bound = MAX( 0, i - b );
        ITYPE y_upper_bound = MIN( ncols - 1, i + b );
        for ( ITYPE j = y_lower_bound; j <= y_upper_bound; j++ ) {
            if ( coin_toss( coin_toss_re ) <= sparsity ) {
                locs.push_back( std::pair<ITYPE, ITYPE>(i, j) );
                vals.push_back( unif(re) );
                nnz_count++;
            }
        }
    }

    *nnzs = nnz_count;
    T *raw_vals = new T[nnz_count];
    std::pair<ITYPE, ITYPE> *raw_locs = new std::pair<ITYPE, ITYPE>[nnz_count];
    assert(raw_vals != NULL && raw_locs != NULL);

    std::memcpy( raw_vals, vals.data(), sizeof(T) * nnz_count );
    std::memcpy( raw_locs, locs.data(), sizeof(std::pair<ITYPE, ITYPE>) * nnz_count );

    return std::make_pair(raw_locs, raw_vals);
}


template <typename T, typename ITYPE>
CSR<T, ITYPE> *generate_csr_band_matrix(ITYPE nrows, ITYPE ncols, ITYPE b)
{
    std::vector<std::pair<ITYPE, ITYPE>> locs;
    std::vector<T> vals;
    ITYPE nnz_count = 0;
    double upper_bound = nrows * ncols;
    double lower_bound = 0.1;
    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    std::default_random_engine re;

    for ( ITYPE i = 0; i < nrows; i++ ) {

        ITYPE y_lower_bound = MAX( 0, i - b );
        ITYPE y_upper_bound = MIN( ncols - 1, i + b );

        for ( ITYPE j = 0; j < ncols; j++ ) {
            if ( j >= y_lower_bound && j <= y_upper_bound ) {
                locs.push_back( std::pair<ITYPE, ITYPE>(i, j) );
                // vals.push_back( unif(re) );
                vals.push_back( rand() );
                nnz_count++;
                // std::cout << "(" << locs.back().first << ", " << locs.back().second << ")" << std::endl;
            }
        }
    }

    CSR<T, ITYPE> *ret = new CSR<T, ITYPE>(nrows, ncols, nnz_count, locs.data(), vals.data());

    return ret;
}

template<typename T, typename ITYPE>
CSC<T, ITYPE> *read_csc(const char *file_name)
{
    ITYPE nrows, ncols, nnzs;
    std::pair<ITYPE, ITYPE> *locs;
    T *vals;
    read_mtx_matrix_into_arrays(file_name, &locs, &vals, &nrows, &ncols, &nnzs);

    CSC<T, ITYPE> *csc = new CSC<T, ITYPE>(nrows, ncols, nnzs, locs, vals);

    if (locs) { delete[] locs; }
    if (vals) { delete[] vals; }

    return csc;
}

template<typename T, typename ITYPE>
DCSC<T, ITYPE> *read_mtx_matrix_dcsc(const char *mtx_file_name, ITYPE segment_size)
{
    ITYPE nrows, ncols, nnz;
    std::pair<ITYPE, ITYPE> *locs;
    T *vals;

    std::cerr << "[DCSC Reader] -- segment size " << segment_size << std::endl;
    read_mtx_matrix_into_arrays(mtx_file_name, &locs, &vals, &nrows, &ncols, &nnz);
    DCSC<T, ITYPE> *dcsc = new DCSC<T, ITYPE>(nrows, ncols, nnz, locs, vals, segment_size);

    if (locs) { delete[] locs; }
    if (vals) { delete[] vals; }

    return dcsc;
}

template<typename T, typename ITYPE>
ATM<T, ITYPE> *read_mtx_matrix_atm(const char *mtx_file_name, std::vector<struct panel_t> &panels)
{
    ITYPE nrows, ncols, nnz;
    std::pair<ITYPE, ITYPE> *locs;
    T *vals;

    read_mtx_matrix_into_arrays(mtx_file_name, &locs, &vals, &nrows, &ncols, &nnz);
    ATM<T, ITYPE> *atm = new ATM<T, ITYPE>( nrows, ncols, nnz, locs, vals, panels.size(), panels );

    if (locs) { delete[] locs; }
    if (vals) { delete[] vals; }

    return atm;
}

template<typename T, typename ITYPE>
SPLIT_CSR<T, ITYPE> *read_mtx_matrix_split_csr(const char *mtx_file_name, ITYPE num_partitions)
{
    ITYPE nrows, ncols, nnzs;
    std::pair<ITYPE, ITYPE> *locs;
    T *vals;

    read_mtx_matrix_into_arrays(mtx_file_name, &locs, &vals, &nrows, &ncols, &nnzs);
    SPLIT_CSR<T, ITYPE> *split_csr = new SPLIT_CSR<T, ITYPE>(nrows, ncols, nnzs, locs, vals, num_partitions);

    if (locs) { delete[] locs; }
    if (vals) { delete[] vals; }

    return split_csr;
}


// takes size in count
template<typename T, typename ITYPE>
T *generate_dense(size_t size)
{
    T *m = (T *) std::aligned_alloc(ALLOC_ALIGNMENT, sizeof(T) * size);

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        m[i] = ((T) (i % 100)) / ((T) 100);
    }

    return m;
}

template<typename T, typename ITYPE>
T *generate_zeroes(size_t size)
{
    T *m = (T *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(T) * size);

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        m[i] = (T) 0;
    }

    return m;
}

template <typename T, typename ITYPE>
T *generate_dense(ITYPE nrows, ITYPE ncols, ITYPE layers = 1)
{
    if (layers < 1) { layers = 1; }

    size_t size = (size_t) nrows * (size_t) ncols * (size_t) layers;

    // std::cout << "Per Matrix Allocation: " << ((size_t) sizeof(T) * (size_t) nrows * (size_t) ncols) << std::endl;
    // std::cout << "Is size cache block aligned: " << (((size_t) sizeof(T) * (size_t) nrows * (size_t) ncols) % ((size_t) ALLOC_ALIGNMENT)) << std::endl;

    T *m = (T *) std::aligned_alloc(ALLOC_ALIGNMENT, sizeof(T) * size);

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        m[i] = ((T) (i % 100)) / ((T) 100);
    }

    return m;
}

template <typename T>
void release_memory(T *ptr)
{
    if (ptr != nullptr) {
        std::free(ptr);
    }
}

template <typename T, typename ITYPE>
T *generate_zeroes(ITYPE nrows, ITYPE ncols, ITYPE layers = 1)
{
    if (layers < 1) { layers = 1; }
    size_t size = (size_t) nrows * (size_t) ncols * (size_t) layers;
    T *m = (T *) std::aligned_alloc(ALLOC_ALIGNMENT, sizeof(T) * size);

    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        m[i] = (T) 0;
    }

    return m;
}

template<typename T, typename ITYPE>
void fill_array(T *arr, size_t num_entries)
{
    #pragma omp parallel for
    for (size_t i = 0; i < num_entries; i++) {
        arr[i] = ((T) (i % 100)) / 100;
    }
}

template<typename T, typename ITYPE>
Matrix<T, ITYPE> *generate_dense(ITYPE nrows, ITYPE ncols)
{
    Matrix<T, ITYPE> *m = new Matrix<T, ITYPE>(nrows, ncols);
    assert(m && "could not generate dense");

    #pragma omp parallel for
    for (ITYPE i = 0; i < (nrows * ncols); i++) {
        m->_data[i] = ((T) (i % 100)) / 100;
    }

    return m;
}

template<typename T, typename ITYPE>
Matrix<T, ITYPE> *generate_dense_ones(ITYPE nrows, ITYPE ncols)
{
    Matrix<T, ITYPE> *m = new Matrix<T, ITYPE>(nrows, ncols);
    for (ITYPE r = 0; r < nrows; r++) {
        for (ITYPE c = 0; c < ncols; c++) {
            T temp = 1.0;
            m->set_element(temp, r, c);
        }
    }

    return m;
}

template<typename T, typename ITYPE>
CSR<T, ITYPE>* sparsify_matrix(Matrix<T, ITYPE> &M)
{

    ITYPE nnz = 0;
    for (ITYPE r = 0; r < M.nrows; r++) {
        for (ITYPE c = 0; c < M.ncols; c++) {
            if (M.get_element(r, c)) { nnz++; }
        }
    }

    // Compute the 3 arrays, call constructor on the sparse residue matrices
    // Allocate temp data structures
    std::pair<ITYPE, ITYPE> *pairs = new std::pair<ITYPE, ITYPE>[nnz];
    T *vals = new T[nnz];

    ITYPE count = 0;
    for (ITYPE r = 0; r < M.nrows; r++) {
        for (ITYPE c = 0; c < M.ncols; c++) {
            if (M.get_element(r, c)) {
                pairs[count].first = r;
                pairs[count].second = c;
                vals[count] = M.get_element(r, c);
                count++;
            }
        }
    }

    assert(count == nnz && "Nunber of elements added to the tmep array is not consistent with nnzs");

    CSR<T, ITYPE> *spm = new CSR<T, ITYPE>(M.nrows, M.ncols, nnz, pairs, vals);

    delete[] pairs;
    delete[] vals;

    return spm;
}

#define PRINT_DIFFERENCE
template<typename T, typename ITYPE>
ITYPE are_equal(Matrix<T, ITYPE> &A, Matrix<T, ITYPE> &B, double divider = 1.0)
{
    if (A.nrows != B.nrows || A.ncols != B.ncols) {
        return std::numeric_limits<ITYPE>::max();
    }

    ITYPE num_differences = 0;
    ITYPE zero_count = 0;

    for (ITYPE r = 0; r < A.nrows; r++) {
        for (ITYPE c = 0; c < A.ncols; c++) {
            T a = A.get_element(r, c) < 0 ? -1 * A.get_element(r, c) : A.get_element(r, c);
            a = a / divider;

            T b = B.get_element(r, c) < 0 ? -1 * B.get_element(r, c) : B.get_element(r, c);

            T diff = ((a - b) < 0 ? (b - a) : (a - b));
            if (diff / MAX(a, b) > 0.01) {
                #ifdef PRINT_DIFFERENCE
                    std::cerr << "Difference at (" << r << "," << c << ") " << A.get_element(r, c) << " " << B.get_element(r, c) << std::endl;
                #endif
                num_differences++;
            }

            if (a == 0.0 && b == 0.0) {
                zero_count++;
            }
        }
    }

    // std::cout << "Total: " << A.nrows * A.ncols  << " Zero count: " << zero_count << std::endl;w

    return num_differences;
}

template <typename T, typename ITYPE>
ITYPE are_equal( CSR<T, ITYPE> &A, DCSC<T, ITYPE> &B )
{
    if ( (A.nrows != B.nrows) || (A.ncols != B.ncols) || (A.nnzs != B.nnzs) ) {
        return std::numeric_limits<ITYPE>::max();
    }

    std::vector<ITYPE> A_locs;
    std::vector<ITYPE> B_locs;

    std::map<ITYPE, T> A_map;
    std::map<ITYPE, T> B_map;
    ITYPE num_differences = 0;

    for ( ITYPE row = 0; row < A.nrows; row++ ) {
        ITYPE row_start = A.row_ptr[row];
        ITYPE row_end = A.row_ptr[row + 1];

        for ( ITYPE ptr = row_start; ptr < row_end; ptr++ ) {
            ITYPE loc = row * A.ncols + A.cols[ptr];

            A_map[ loc ] = A.vals[ptr];

            A_locs.push_back( loc );
        }
    }

    for ( ITYPE panel = 0; panel < B.num_segments; panel++ ) {
        ITYPE panel_start = B.aux[panel];
        ITYPE panel_end = B.aux[panel + 1];
        for ( ITYPE panel_ptr = panel_start; panel_ptr < panel_end; panel_ptr++ ) {
            ITYPE col_start = B.col_ptr[ panel_ptr ];
            ITYPE col_end = B.col_ptr[ panel_ptr + 1 ];
            ITYPE col = B.cols[ panel_ptr ];
            for ( ITYPE ptr = col_start; ptr < col_end; ptr++ ) {
                ITYPE row = B.rows[ ptr ];
                ITYPE loc = row * B.ncols + col;
                B_map[ loc ] = B.vals[ ptr ];
                B_locs.push_back( loc );
            }
        }
    }

    if ( A_locs.size() != B_locs.size() ) {
        std::cout << "Missing non-zero in the DCSC output" << std::endl;
    }
    std::sort( A_locs.begin(), A_locs.end() );
    std::sort( B_locs.begin(), B_locs.end() );

    for ( ITYPE i = 0; i < A_locs.size(); i++ ) {
        if (A_locs[i] != B_locs[i]) {
            std::cout << "location miss match after " << i << " values" << std::endl;
            return std::numeric_limits<ITYPE>::max();
        }
    }

    if ( A_map.size() != B_map.size() ) {
        std::cout << "Number of non-zeros is different -- " << "E: " << A_map.size() << " -- " << "A: " << B_map.size() << std::endl;
    }

    for ( auto &itA : A_map ) {
        const auto &itB = B_map.find( itA.first );
        ITYPE row = itA.first / A.ncols;
        ITYPE col = itA.first % A.ncols;
        if ( itB == B_map.end() ) {
            std::cout << "Missing element at (" << row << ", " << col << ")" << std::endl;
            return std::numeric_limits<ITYPE>::max();
        }

        T a = itA.second < 0 ? -1 * itA.second : itA.second;
        T b = itB->second < 0 ? -1 * itB->second : itB->second;
        T diff = (a - b) < 0 ? (b - a) : (a - b);

        if ( diff / MAX(a, b) > 0.01 ) {
            std::cout << "Difference in location: (" << row << ", " << col << ")" << " -- " << "E: " << itA.second << " -- " << "A: " << itB->second << std::endl;
            num_differences++;
        }
    }

    return num_differences;
}

template <typename T, typename ITYPE>
ITYPE are_equal( T *A, T *B, ITYPE size, double divider = 1.0 )
{
    ITYPE num_differences = 0;
    ITYPE num_zero = 0;

    for ( ITYPE i = 0; i < size; i++ ) {
        T a = A[i] < 0 ? -1 * A[i] : A[i];
        T b = B[i] < 0 ? -1 * B[i] : B[i];
        T diff = (a - b) < 0 ? (b - a) : (a - b);
        if ( diff / MAX(a, b) > 0.01 ) {
            num_differences++;
        }
        if ( a == 0.0 && b == 0.0 ) {
            num_zero++;
        }
    }

    std::cout << "Total: " << size << " Zero count: " << num_zero << std::endl;

    return num_differences;
}

template <typename T, typename ITYPE>
ITYPE are_equal( CSR<T, ITYPE> &A, CSR<T, ITYPE> &B )
{
    if ( (A.nrows != B.nrows) || (A.ncols != B.ncols) ) {
        return std::numeric_limits<ITYPE>::max();
    }
    ITYPE num_differences = 0;

    for ( ITYPE row = 0; row < A.nrows; row++ ) {
        ITYPE row_start_A = A.row_ptr[row];
        ITYPE row_end_A = A.row_ptr[row + 1];
        ITYPE row_start_B = B.row_ptr[row];
        ITYPE row_end_B = B.row_ptr[row + 1];

        for ( ITYPE ptrA = row_start_A, ptrB = row_start_B; (ptrA < row_end_A) && (ptrB < row_end_B); ptrA++, ptrB++ ) {
            ITYPE col = A.cols[ptrA];

            T a = A.vals[ptrA] < 0 ? ( -1 * A.vals[ptrA] ) : A.vals[ptrA];
            T b = B.vals[ptrB] < 0 ? ( -1 * B.vals[ptrB] ) : B.vals[ptrB];
            T diff = (a > b) ? (a - b) : (b - a);
            if (diff / MAX(a, b) > 0.01) {
                std::cout << "Difference in location: (" << row << ", " << col << ")" << " -- " << "E: " << A.vals[ptrA] << " -- " << "A: " << B.vals[ptrB] << std::endl;
                num_differences++;
            }
        }
    }

    return num_differences;
}


template <typename T, typename ITYPE>
bool verify_matrix_structure( CSR<T, ITYPE> &A, ATM<T, ITYPE> &B )
{
    if ( A.nrows != B.nrows || A.ncols != B.ncols || A.nnzs != B.nnzs ) {
        return false;
    }

    ITYPE num_rows = A.nrows;
    std::vector<std::vector<std::pair<ITYPE, T>>> m_csr( num_rows, std::vector<std::pair<ITYPE, T>>() );
    std::vector<std::vector<std::pair<ITYPE, T>>> m_atm( num_rows, std::vector<std::pair<ITYPE, T>>() );

    for ( ITYPE row = 0; row < A.nrows; row++ ) {
        ITYPE row_start = A.row_ptr[row];
        ITYPE row_end = A.row_ptr[row + 1];
        for ( ITYPE row_ptr = row_start; row_ptr < row_end; row_ptr++ ) {
            ITYPE col = A.cols[row_ptr];
            m_csr[row].push_back( std::pair<ITYPE, T>(col, A.vals[row_ptr]) );
        }
    }

    for ( ITYPE panel = 0; panel < B.num_panels; panel++ ) {
        ITYPE panel_num_tiles = B.panel_ptr[panel + 1] - B.panel_ptr[panel];

        ITYPE panel_Ti = B.panel_Ti[panel];
        ITYPE panel_offset = B.panel_offset[panel];
        ITYPE panel_start = B.panel_start[panel];

        for ( ITYPE tile = 0; tile < panel_num_tiles; tile++ ) {
            for ( ITYPE row = panel_start; row < MIN(panel_start + panel_Ti, B.nrows); row++ ) {
                ITYPE i = row - panel_start;
                // ITYPE row = panel_start + i;
                ITYPE ptr = panel_offset + ( i * panel_num_tiles ) + tile;
                ITYPE row_start = B.tile_row_ptr[ptr];
                ITYPE row_end = B.tile_row_ptr[ptr + 1];
                for ( ITYPE j = row_start; j < row_end; j++ ) {
                    ITYPE col = B.cols[j];
                    m_atm[row].push_back( std::pair<ITYPE, T>(col, B.vals[j]) );
                    // std::cout << "Adding: (" << row << ", " << col << ")" << " ptr: " << j << std::endl;
                }
            }
        }
    }

    bool check = true;
    for (ITYPE row = 0; row < num_rows; row++) {
        std::sort( m_csr[row].begin(), m_csr[row].end() );
        std::sort( m_atm[row].begin(), m_atm[row].end() );

        if (m_csr[row].size() != m_atm[row].size() ) {
            std::cout << "ATM Row" << row << " mismatch in number of active cols " << std::endl;
        }

        ITYPE ncols = m_csr[row].size();

        for (ITYPE it = 0; it < ncols; it++) {
            if (m_csr[row][it].first != m_atm[row][it].first) {
                std::cout << "ATM Row: " << row << " Missing Column: " << m_csr[row][it].first << std::endl;
                check = false;
                break;
            } else if (m_csr[row][it].second != m_atm[row][it].second) {
                std::cout << "ATM Row: " << row << " Col: " << m_csr[row][it].first << " value mismatch" << " -- E: " << m_csr[row][it].second << " G: " << m_atm[row][it].second << std::endl;
                check = false;
                break;
            }
        }
    }

    return check;
}

template <typename T, typename ITYPE>
bool verify_matrix_structure(CSR<T, ITYPE> &csr, CSF<T, ITYPE> &csf)
{

    ITYPE num_rows = csr.nrows;
    std::vector<std::vector<std::pair<ITYPE, T>>> m_csr( num_rows, std::vector<std::pair<ITYPE, T>>() );
    std::vector<std::vector<std::pair<ITYPE, T>>> m_atm( num_rows, std::vector<std::pair<ITYPE, T>>() );

    for ( ITYPE row = 0; row < csr.nrows; row++ ) {
        ITYPE row_start = csr.row_ptr[row];
        ITYPE row_end = csr.row_ptr[row + 1];
        for ( ITYPE row_ptr = row_start; row_ptr < row_end; row_ptr++ ) {
            ITYPE col = csr.cols[row_ptr];
            m_csr[row].push_back( std::pair<ITYPE, T>(col, csr.vals[row_ptr]) );
        }
    }


    ITYPE *A1_pos = csf.indices[0][0];
    ITYPE *A1_crd = csf.indices[0][1];
    ITYPE *A2_pos = csf.indices[1][0];
    ITYPE *A2_crd = csf.indices[1][1];
    ITYPE *A3_pos = csf.indices[2][0];
    ITYPE *A3_crd = csf.indices[2][1];
    ITYPE *A4_pos = csf.indices[3][0];
    ITYPE *A4_crd = csf.indices[3][1];

    for (ITYPE tile_row_ptr = A1_pos[0]; tile_row_ptr < A1_pos[1]; tile_row_ptr++) {
        ITYPE tile_row = A1_crd[tile_row_ptr];
        for (ITYPE tile_col_ptr = A2_pos[tile_row_ptr]; tile_col_ptr < A2_pos[tile_row_ptr + 1]; tile_col_ptr++) {
            ITYPE tile_col = A2_crd[tile_col_ptr];
            for (ITYPE row_ptr = A3_pos[tile_col_ptr]; row_ptr < A3_pos[tile_col_ptr + 1]; row_ptr++) {
                ITYPE row = A3_crd[row_ptr];
                for (ITYPE col_ptr = A4_pos[row_ptr]; col_ptr < A4_pos[row_ptr + 1]; col_ptr++) {
                    ITYPE col = A4_crd[col_ptr];
                    m_atm[row].push_back( std::pair<ITYPE, T>(col, csf.vals[col_ptr]) );
                }
            }
        }
    }

    bool check = true;
    for (ITYPE row = 0; row < num_rows; row++) {
        std::sort( m_csr[row].begin(), m_csr[row].end() );
        std::sort( m_atm[row].begin(), m_atm[row].end() );

        if (m_csr[row].size() != m_atm[row].size() ) {
            std::cout << "ATM Row" << row << " mismatch in number of active cols " << std::endl;
        }

        ITYPE ncols = m_csr[row].size();

        for (ITYPE it = 0; it < ncols; it++) {
            if (m_csr[row][it].first != m_atm[row][it].first) {
                std::cout << "ATM Row: " << row << " Missing Column: " << m_csr[row][it].first << std::endl;
                check = false;
                break;
            } else if (m_csr[row][it].second != m_atm[row][it].second) {
                std::cout << "ATM Row: " << row << " Col: " << m_csr[row][it].first << " value mismatch" << " -- E: " << m_csr[row][it].second << " G: " << m_atm[row][it].second << std::endl;
                check = false;
                break;
            }
        }
    }

    return check;
}

template <typename T, typename ITYPE>
bool verify_matrices(CSR<T, ITYPE> &csr, DCSC<T, ITYPE> &dcsc)
{
    std::cout << "Verifying DCSC Matrix" << std::endl;

    ITYPE num_rows = csr.nrows;
    std::vector<std::vector<std::pair<ITYPE, T>>> m_csr( num_rows, std::vector<std::pair<ITYPE, T>>() );
    std::vector<std::vector<std::pair<ITYPE, T>>> m_dcsc( num_rows, std::vector<std::pair<ITYPE, T>>() );

    for (ITYPE row = 0; row < csr.nrows; row++) {
        ITYPE row_start = csr.row_ptr[row];
        ITYPE row_end = csr.row_ptr[row + 1];
        for (ITYPE ptr = row_start; ptr < row_end; ptr++) {
            ITYPE col = csr.cols[ptr];
            m_csr[row].push_back( std::pair<ITYPE, T>(col, csr.vals[ptr]) );
        }
    }

    for (ITYPE panel = 0; panel < dcsc.num_segments; panel++) {
        ITYPE panel_start = dcsc.aux[panel];
        ITYPE panel_end = dcsc.aux[panel + 1];
        for (ITYPE j = panel_start; j < panel_end; j++) {
            ITYPE col = dcsc.cols[j];
            ITYPE col_start = dcsc.col_ptr[j];
            ITYPE col_end = dcsc.col_ptr[j + 1];
            for (ITYPE ptr = col_start; ptr < col_end; ptr++) {
                ITYPE row = dcsc.rows[ptr];
                m_dcsc[row].push_back( std::pair<ITYPE, T>(col, dcsc.vals[ptr]) );
            }
        }
    }

    bool check = true;
    for (ITYPE row = 0; row < num_rows; row++) {
        std::sort( m_csr[row].begin(), m_csr[row].end() );
        std::sort( m_dcsc[row].begin(), m_dcsc[row].end() );

        if (m_csr[row].size() != m_dcsc[row].size() ) {
            std::cout << "DCSC Row" << row << " mismatch in number of active cols " << std::endl;
        }

        ITYPE ncols = m_csr[row].size();

        for (ITYPE it = 0; it < ncols; it++) {
            if (m_csr[row][it].first != m_dcsc[row][it].first) {
                std::cout << "DCSC Row: " << row << " Missing Column: " << m_csr[row][it].first << std::endl;
                check = false;
                break;
            } else if (m_csr[row][it].second != m_dcsc[row][it].second) {
                std::cout << "DCSC Row: " << row << " Col: " << m_csr[row][it].first << " value mismatch" << std::endl;
                check = false;
                break;
            }
        }
    }

    return check;
}

#endif // READER_H
