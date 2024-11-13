#ifndef RESIDUE_H
#define RESIDUE_H

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <omp.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "BitSet.h"
#include "common.h"
#include "matrices/CSC.h"
#include "matrices/CSR.h"
#include "matrices/Matrix.h"
#include "utils/Statistics.h"
#include "utils/util.h"

// #define GENERATE_FULL_BITSETS
#define GENERATE_VARIABLE_RESOLUTION_BITSET
#define DEFAULT_CACHE 8 * 32 * 1024
#define L2_CACHE_SIZE (128 * 1024 * 1024)
#define MAXIMUM_TILE_SIZE 8192
#define BINARY_SEARCH_THRESHOLD 64


template<typename T = uint64_t, typename ITYPE = size_t>
class Residue {
private:
    // resolution -- the number of rows/columns per residue bit
    void generate_sparse_residue();
    void generate_bitsets(ITYPE resolution = 1);
    void generate_sparse_bitsets(ITYPE resolution = 1);


    void verify_sparse_residue();
    void verify_sparse_bitsets(ITYPE resolution = 1);
    void generate_range_augmentation();

public:
    // Reference to matrix for which residue has to be constructed
    CSR<T, ITYPE> *_spm;
    CSC<T, ITYPE> *_csc;

    const std::string &mtx_name;

    ITYPE Ti;   // Height of sparse tile
    ITYPE Tj;   // Width of sparse tile
    ITYPE Tk;   // Width of dense input and output tiles
    ITYPE nnz;  // Number of non-zeros in the residue matrix

    ITYPE nrows, ncols; // Number of rows and cols in the residue
    ITYPE resolution;   // Number of entries per bit in the residue bitsets

    CSR<ITYPE, ITYPE> *res_raw_sparse;
    CSR<BitSet<ITYPE>, ITYPE> *bitset_active_cols_csr;
    CSR<BitSet<ITYPE>, ITYPE> *bitset_active_rows_csr;
    CSR<BitSet<ITYPE>, ITYPE> *bitset_active_reused_cols_csr;
    CSR<BitSet<ITYPE>, ITYPE> *bitset_active_reused_rows_csr;

    // range pair value is { start, end } - for that particular active column
    CSR<RangeInfo<ITYPE>, ITYPE> *active_cols_range = nullptr;
    CSR<RangeInfo<ITYPE>, ITYPE> *active_rows_range = nullptr;

    // Functions and constructors
    Residue(CSR<T, ITYPE> *spm, CSC<T, ITYPE> *csc, int64_t Ti = 64, int64_t Tj = 64, const std::string &mtx_name = 0, ITYPE resolution = 1, bool range_augmentation = false);
    ~Residue();

    // 2D adaptive tiling methods
    std::vector<struct panel_t> adaptive_2d_greedy_Ti_greedy_Tj_tile_generator( ITYPE feature, ITYPE cache_size = L2_CACHE_SIZE, ITYPE cache_split = 4, bool temporal_input = false, bool temporal_output = false, bool oi_aware = false);
};


template<typename T, typename ITYPE>
Residue<T, ITYPE>::~Residue() {
    if(res_raw_sparse) { delete res_raw_sparse;  }


    // delete augmentations
    delete bitset_active_cols_csr;
    delete bitset_active_rows_csr;
    delete bitset_active_reused_cols_csr;
    delete bitset_active_reused_rows_csr;

    if (active_cols_range && active_rows_range) {

        for ( ITYPE i = 0; i < this->active_cols_range->nnzs; i++ ) {
            active_cols_range->vals[i].~RangeInfo<ITYPE>();
            active_rows_range->vals[i].~RangeInfo<ITYPE>();
        }

        delete active_cols_range;
        delete active_rows_range;
    }
}

// Constructor for the residue matrix
template<typename T, typename ITYPE>
Residue<T, ITYPE>::Residue(CSR<T, ITYPE> *spm, CSC<T, ITYPE> *csc, int64_t Ti, int64_t Tj, const std::string &mtx_name,
                                ITYPE resolution, bool augment_temporal) : Ti(Ti), Tj(Tj), Tk(Tk), mtx_name(mtx_name), resolution(resolution) {

    if (Ti < 0 || Tj < 0) { return; }
    _spm = spm;
    _csc = csc;

    nrows = CEIL(spm->get_nrows(), Ti);
    ncols = CEIL(spm->get_ncols(), Tj);

    // Faster implementation
    auto start_time = std::chrono::high_resolution_clock::now();
    generate_sparse_residue();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff(end_time - start_time);
    double sparse_gen_time = diff.count();
    print_status("Sparse residue generation time: %f\n", sparse_gen_time);

    start_time = std::chrono::high_resolution_clock::now();
    generate_sparse_bitsets(resolution);
    end_time = std::chrono::high_resolution_clock::now();
    diff = end_time - start_time;
    double bitset_gen_time = diff.count();
    print_status("Total bitset generation time: %f\n", bitset_gen_time);

    if (augment_temporal) {
        print_status("Augmenting temporal range information\n");
        start_time = std::chrono::high_resolution_clock::now();
        generate_range_augmentation();
        end_time = std::chrono::high_resolution_clock::now();
        diff = end_time - start_time;
        double range_gen_time = diff.count();
        print_status("Total range generation time: %f\n", range_gen_time);
    }

    #ifdef DEBUG_RESIDUE_CONSTRUCTION
        verify_sparse_residue();
        verify_sparse_bitsets(resolution);
    #endif

    #ifdef DEBUG_BITSET
        verify_bitset();
    #endif


    #ifdef DEBUG_RESIDUE
        print_raw_residue();
        print_frac_residue();
    #endif
}


template<typename T, typename ITYPE>
void Residue<T, ITYPE>::generate_sparse_residue()
{
    // Go directly from the sparse matrix to the sparse residue
    auto num_threads = omp_get_max_threads();

    ITYPE num_panels = CEIL(this->_spm->nrows, this->Ti);

    this->res_raw_sparse = new CSR<ITYPE, ITYPE>(this->nrows, this->ncols);

    ITYPE *temp_panel = new ITYPE[num_threads * this->ncols]();
    bool *temp_active_tiles = new bool[num_threads * this->ncols]();
    std::vector<ITYPE> *temp_active_tiles_idx = new std::vector<ITYPE>[num_threads];

    for ( ITYPE i = 0; i < num_threads; i++ ) {
        temp_active_tiles_idx[i].resize( 40 );
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    #pragma omp parallel num_threads(num_threads)
    {
        auto my_tid = omp_get_thread_num();

        #pragma omp for schedule(dynamic)
        for (ITYPE panel = 0; panel < num_panels; panel++) {
            // auto my_tid = 0;
            bool *my_active_tiles = &temp_active_tiles[my_tid * this->ncols];
            temp_active_tiles_idx[my_tid].clear();

            ITYPE panel_start = panel * this->Ti;
            ITYPE panel_end = MIN((panel + 1) * this->Ti, _spm->nrows);
            for (ITYPE row = panel_start; row < panel_end; row++) {
                ITYPE row_start = this->_spm->row_ptr[row];
                ITYPE row_end = this->_spm->row_ptr[row + 1];
                for (ITYPE ptr = row_start; ptr < row_end; ptr++) {
                    ITYPE col = this->_spm->cols[ptr];
                    ITYPE tile = col / this->Tj;
                    if (!my_active_tiles[tile]) {
                        temp_active_tiles_idx[my_tid].push_back(tile);
                        my_active_tiles[tile] = true;
                    }
                }
            }

            for (auto &i : temp_active_tiles_idx[my_tid]) {
                this->res_raw_sparse->row_ptr[panel + 1]++;
                my_active_tiles[i] = false;
            }
        }
    }

    // prefix sum the row_ptrs
    this->res_raw_sparse->row_ptr[0] = 0;
    for (ITYPE i = 1; i < (this->nrows + 1); i++) {
        this->res_raw_sparse->row_ptr[i] += this->res_raw_sparse->row_ptr[i - 1];
    }
    // allocate memory for the non-zero residue tiles
    this->nnz = this->res_raw_sparse->nnzs = this->res_raw_sparse->row_ptr[this->nrows];
    this->res_raw_sparse->init_memory(this->nnz);

    // Do the actual computation
    #pragma omp parallel num_threads(num_threads)
    {
        auto my_tid = omp_get_thread_num();

        #pragma omp for schedule(dynamic)
        for (ITYPE panel = 0; panel < num_panels; panel++) {

            // auto my_tid = omp_get_thread_num();
            // auto my_tid = 0;
            ITYPE *my_temp_panel = &temp_panel[this->ncols * my_tid];
            temp_active_tiles_idx[my_tid].clear();

            ITYPE panel_start = panel * this->Ti;
            ITYPE panel_end = MIN( (panel_start + this->Ti) , this->_spm->nrows);
            ITYPE panel_nnz = 0;
            for (ITYPE row = panel_start; row < panel_end; row++) {
                ITYPE row_start = this->_spm->row_ptr[row];
                ITYPE row_end = this->_spm->row_ptr[row + 1];
                for (ITYPE ptr = row_start; ptr < row_end; ptr++) {
                    ITYPE col = this->_spm->cols[ptr];
                    ITYPE tile = col / this->Tj;
                    if (my_temp_panel[tile] == 0) {
                        temp_active_tiles_idx[my_tid].push_back(tile);
                    }
                    my_temp_panel[tile]++;
                    panel_nnz++;
                }
            }

            // Keep the residues ordered to ensure that the binary search for construction works!
            std::sort(temp_active_tiles_idx[my_tid].begin(), temp_active_tiles_idx[my_tid].end());

            ITYPE panel_res_ptr = this->res_raw_sparse->row_ptr[panel];
            ITYPE panel_res_nnz = 0;
            for ( auto &i : temp_active_tiles_idx[my_tid] ) {
                this->res_raw_sparse->cols[panel_res_ptr] = i;
                this->res_raw_sparse->vals[panel_res_ptr++] = my_temp_panel[i];
                panel_res_nnz += my_temp_panel[i];
                my_temp_panel[i] = 0;
            }
            assert( panel_nnz == panel_res_nnz && "something is incorrect with the counting" );
            assert( panel_res_ptr == this->res_raw_sparse->row_ptr[panel + 1] && "missing tiles in the residue constuction" );
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    print_status("Sparse Resdiue Construction Runtime: %f\n", std::chrono::duration<double>(end_time - start_time).count() );

#ifdef DEBUG_RESIDUE_CONSTRUCTION
    ITYPE residue_sum = 0;
    for (ITYPE res_row = 0; res_row < this->res_raw_sparse->nrows; res_row++) {
        for (ITYPE ptr = this->res_raw_sparse->row_ptr[res_row]; ptr < this->res_raw_sparse->row_ptr[res_row + 1]; ptr++) {
            residue_sum += this->res_raw_sparse->vals[ptr];
        }
    }
    std::cout << "Residue sum: " << residue_sum << " -- Expected sum: " << this->_spm->nnzs << std::endl;
    assert( residue_sum == this->_spm->nnzs && "Residue sparse generation failed" );
#endif // DEBUG_RESIDUE_CONSTRUCTION

    delete []temp_panel;
    delete []temp_active_tiles;
    delete []temp_active_tiles_idx;
}

// Really slow routine that relies on std vector and std map
// Don't use unless debugging!!!
template <typename T, typename ITYPE>
void Residue<T, ITYPE>::verify_sparse_residue()
{
    std::vector<std::map<ITYPE, ITYPE>> verif_residue;

    for ( ITYPE panel = 0; panel < this->nrows; panel++ ) {
        // verif_residue[panel] = std::map<ITYPE, ITYPE>();
        verif_residue.push_back( std::map<ITYPE, ITYPE>() );

        ITYPE panel_start = panel * this->Ti;
        ITYPE panel_end = MIN((panel + 1) * this->Ti, _spm->nrows);
        for ( ITYPE row = panel_start; row < panel_end; row++ ) {
            ITYPE row_start = this->_spm->row_ptr[row];
            ITYPE row_end = this->_spm->row_ptr[row + 1];
            for ( ITYPE ptr = row_start; ptr < row_end; ptr++ ) {
                ITYPE col = this->_spm->cols[ptr];
                ITYPE tile = col / this->Tj;
                if ( verif_residue[panel].find(tile) == verif_residue[panel].end() ) {
                    verif_residue[panel][tile] = 0;
                }
                verif_residue[panel][tile]++;
            }
        }
    }

    for ( ITYPE res_row = 0; res_row < this->nrows; res_row++ ) {
        ITYPE res_row_start = this->res_raw_sparse->row_ptr[res_row];
        ITYPE res_row_end = this->res_raw_sparse->row_ptr[res_row + 1];

        if ( verif_residue[res_row].size() != (res_row_end - res_row_start) ) {
            print_error_exit("Mismatch in panel sizes at panel: %ld -- expected: %ld -- found: %ld\n", res_row, verif_residue[res_row].size(), (res_row_end - res_row_start));
        }

        for ( ITYPE res_ptr = res_row_start; res_ptr < res_row_end; res_ptr++ ) {
            ITYPE res_col = this->res_raw_sparse->cols[res_ptr];
            ITYPE res_val = this->res_raw_sparse->vals[res_ptr];

            if ( verif_residue[res_row].find(res_col) == verif_residue[res_row].end() ) {
                print_error_exit("Extra non-zero col in the sparse residue: %ld\n", res_col);
            }

            if ( verif_residue[res_row][res_col] != res_val ) {
                print_error_exit("Mismatch in panel: %ld -- tile: %ld -- expected: %ld -- found: %d\n", res_row, res_col, verif_residue[res_row][res_col], res_val);
            }
        }
    }
}

// verify every single bitset entry in the sparse bitsets
template <typename T, typename ITYPE>
void Residue<T, ITYPE>::verify_sparse_bitsets(ITYPE resolution)
{
    // Let's go one panel at a time
    for ( ITYPE panel = 0; panel < this->nrows; panel++ ) {

        std::map<ITYPE, std::set<ITYPE>> panel_active_cols;
        std::map<ITYPE, std::set<ITYPE>> panel_active_rows;

        ITYPE panel_start = panel * this->Ti;
        ITYPE panel_end = MIN((panel + 1) * this->Ti, _spm->nrows);
        for ( ITYPE row = panel_start; row < panel_end; row++ ) {
            ITYPE row_start = this->_spm->row_ptr[row];
            ITYPE row_end = this->_spm->row_ptr[row + 1];
            for ( ITYPE ptr = row_start; ptr < row_end; ptr++ ) {
                ITYPE col = this->_spm->cols[ptr];
                ITYPE tile = col / this->Tj;

                if ( panel_active_cols.find(tile) == panel_active_cols.end() ) {
                    panel_active_cols[tile] = std::set<ITYPE>();
                }
                panel_active_cols[tile].insert(col);

                if ( panel_active_rows.find(tile) == panel_active_rows.end() ) {
                    panel_active_rows[tile] = std::set<ITYPE>();
                }
                panel_active_rows[tile].insert(row);
            }
        }

        // if ( bitset_active_cols_csr->row_ptr[panel + 1] - bitset_active_cols_csr->row_ptr[panel]

        for ( ITYPE bitset_ptr = this->bitset_active_cols_csr->row_ptr[panel]; bitset_ptr < this->bitset_active_cols_csr->row_ptr[panel+1]; bitset_ptr++ ) {
            ITYPE col = this->bitset_active_cols_csr->cols[bitset_ptr];
            ITYPE verif_nacs = panel_active_cols[col].size();
            ITYPE bitset_nacs = this->bitset_active_cols_csr->vals[bitset_ptr].count();
            ITYPE verif_nars = panel_active_rows[col].size();
            ITYPE bitset_nars = this->bitset_active_rows_csr->vals[bitset_ptr].count();

            if ( !verif_nacs || !verif_nars || !bitset_nacs || !bitset_nars ) {
                print_error_exit("[Sparse Bitset Verification] -- Zero size bitset found\n");
            }

            if ( bitset_nacs != verif_nacs ) {
                print_error_exit("Mismatch in active cols -- Panel: %ld -- Tile: %ld -- expected: %ld -- found: %ld\n", panel, col, verif_nacs, bitset_nacs);
            }

            if ( bitset_nars != verif_nars ) {
                print_error_exit("Mismatch in active rows -- Panel: %ld -- Tile: %ld -- expected: %ld -- found: %ld\n", panel, col, verif_nars, bitset_nars);
            }
        }
    }

    print_status("Bitset verification successful\n");
}


template<typename T, typename ITYPE>
void Residue<T, ITYPE>::generate_sparse_bitsets(ITYPE resolution)
{

    // Allocate bitsets without allocating any memory for the row_ptr and cols
    bitset_active_cols_csr = new CSR<BitSet<ITYPE>, ITYPE>(this->nrows, this->ncols, this->nnz, true); // Create active cols csr
    bitset_active_reused_cols_csr = new CSR<BitSet<ITYPE>, ITYPE>(this->nrows, this->ncols, this->nnz, true); // Create active reused cols csr
    bitset_active_rows_csr = new CSR<BitSet<ITYPE>, ITYPE>(this->nrows, this->ncols, this->nnz, true);
    bitset_active_reused_rows_csr = new CSR<BitSet<ITYPE>, ITYPE>(this->nrows, this->ncols, this->nnz, true);

    // auto num_threads = omp_get_num_threads();
    auto num_threads = omp_get_max_threads();

    ITYPE chunk_size = CEIL(this->nnz, num_threads);


    /*
    #pragma omp parallel for schedule(static, chunk_size)
    for ( ITYPE i = 0; i < this->nnz; i++ ) {
        new (&(bitset_active_cols_csr->vals[i])) BitSet<ITYPE>( Tj / resolution );
        new (&(bitset_active_reused_cols_csr->vals[i])) BitSet<ITYPE>( Tj / resolution );

        new (&(bitset_active_rows_csr->vals[i])) BitSet<ITYPE>( Ti / resolution );
        new (&(bitset_active_reused_rows_csr->vals[i])) BitSet<ITYPE>( Ti / resolution );
    }
    */


    ITYPE num_entries_cols = this->Tj / resolution;
    ITYPE num_entries_rows = this->Ti / resolution;
    ITYPE data_store_size_cols = CEIL( num_entries_cols, (sizeof(size_t) * BITS_IN_BYTE) );
    ITYPE data_store_size_rows = CEIL( num_entries_rows, (sizeof(size_t) * BITS_IN_BYTE) );


    if (num_threads > 4) {
    ITYPE num_threads_per_init_loop = CEIL(num_threads, 4);
    ITYPE work_per_thread = CEIL(this->nnz, num_threads_per_init_loop);

    #pragma omp parallel num_threads(num_threads)
    {
        auto my_tid = omp_get_thread_num();
        ITYPE my_start = (my_tid % num_threads_per_init_loop) * work_per_thread;
        ITYPE my_end = MIN( ((my_tid % num_threads_per_init_loop) + 1) * work_per_thread, this->nnz );

        if (my_tid < num_threads_per_init_loop) {
            for ( ITYPE i = my_start; i < my_end; i++ ) {
                bitset_active_cols_csr->vals[i].init( num_entries_cols, data_store_size_cols );
            }
        } else if (my_tid >= num_threads_per_init_loop && my_tid < (2 * num_threads_per_init_loop)) {
            for ( ITYPE i = my_start; i < my_end; i++ ) {
                bitset_active_rows_csr->vals[i].init( num_entries_rows, data_store_size_rows );
            }
        } else if (my_tid >= (2 * num_threads_per_init_loop) && my_tid < (3 * num_threads_per_init_loop)) {
            for ( ITYPE i = my_start; i < my_end; i++ ) {
                bitset_active_reused_cols_csr->vals[i].init( num_entries_cols, data_store_size_cols );
            }
        } else {
            for ( ITYPE i = my_start; i < my_end; i++ ) {
                bitset_active_reused_rows_csr->vals[i].init( num_entries_cols, data_store_size_rows );
            }
        }
    }

    } else {
        #pragma omp parallel for
        for ( ITYPE i = 0; i < this->nnz; i++ ) {
            bitset_active_cols_csr->vals[i].init( num_entries_cols, data_store_size_cols );
            bitset_active_rows_csr->vals[i].init( num_entries_rows, data_store_size_rows );
            bitset_active_reused_cols_csr->vals[i].init( num_entries_cols, data_store_size_cols );
            bitset_active_reused_rows_csr->vals[i].init( num_entries_cols, data_store_size_rows );
        }
    }

    bitset_active_cols_csr->row_ptr = bitset_active_rows_csr->row_ptr = bitset_active_reused_cols_csr->row_ptr = bitset_active_reused_rows_csr->row_ptr = res_raw_sparse->row_ptr;
    bitset_active_cols_csr->cols = bitset_active_rows_csr->cols = bitset_active_reused_cols_csr->cols = bitset_active_reused_rows_csr->cols = res_raw_sparse->cols;

    // START ACTUAL computation
    auto start_time_generate_bitset = std::chrono::high_resolution_clock::now();

    ITYPE num_panels = CEIL(this->_spm->nrows, this->Ti);

    #pragma omp parallel for schedule(static, 4)
    for ( ITYPE panel = 0; panel < num_panels; panel++ ) {
        ITYPE panel_start = panel * this->Ti;
        ITYPE panel_end = MIN( (panel_start + this->Ti), this->_spm->nrows );

        // ITYPE panel_num_active_tiled = this->bitset_active_cols_csr->row_ptr[panel] - this->bitset_active_cols_csr->row_ptr[panel + 1];
        for ( ITYPE row = panel_start; row < panel_end; row++ ) {
            ITYPE row_start = this->_spm->row_ptr[row];
            ITYPE row_end = this->_spm->row_ptr[row + 1];

            ITYPE residue_row_start = this->bitset_active_cols_csr->row_ptr[panel];
            ITYPE residue_row_end = this->bitset_active_cols_csr->row_ptr[panel + 1];

            // Iterate over the columns of the matrix row
            for ( ITYPE ptr = row_start; ptr < row_end; ptr++ ) {
                ITYPE col = this->_spm->cols[ptr];
                ITYPE curr_tile = col / this->Tj;

                // find tile in the current panel


                // binary search to find the col in the bitset
                ITYPE lo = residue_row_start;
                ITYPE hi = residue_row_end;
                while (lo <= hi) {
                    ITYPE mid = (lo + hi) / 2;
                    if ( curr_tile ==  this->bitset_active_cols_csr->cols[mid] ) {

                        if ( this->bitset_active_cols_csr->vals[mid].get( col % (this->Tj / resolution) ) ) {
                            this->bitset_active_reused_cols_csr->vals[mid].set( col % (this->Tj / resolution) );
                        } else {
                            this->bitset_active_cols_csr->vals[mid].set( col % (this->Tj / resolution) );
                        }

                        this->bitset_active_rows_csr->vals[mid].set( row % (this->Ti / resolution) );
                        break;
                    }

                    if ( curr_tile < this->bitset_active_cols_csr->cols[mid] ) {
                        hi = mid - 1;
                    } else {
                        lo = mid + 1;
                    }
                }

                /*
                for ( ITYPE tile_ptr = residue_row_start; tile_ptr < residue_row_end; tile_ptr++ ) {
                    ITYPE col_bitset_ndx = col % (Tj / resolution);
                    ITYPE row_bitset_ndx = row % (Ti / resolution);

                    // Find the current tile in the panel and set big high. Then break
                    if ( curr_tile == this->bitset_active_cols_csr->cols[tile_ptr] ) {
                        if ( this->bitset_active_cols_csr->vals[tile_ptr].get( col_bitset_ndx )) {
                            this->bitset_active_reused_cols_csr->vals[tile_ptr].set( col_bitset_ndx );
                        } else {
                            this->bitset_active_cols_csr->vals[tile_ptr].set( col % (Tj / resolution) );
                        }
                        this->bitset_active_rows_csr->vals[tile_ptr].set( row % (Ti / resolution) );

                        // if ( panel == 0 && curr_tile == 0 ) {
                        //     std::cout << "Setting ( " << row << ", " << col << ") -- " << col % (Tj / resolution) << std::endl;
                        //     set2.insert(col);
                        // }

                        break;
                    }
                }
                */

            }
        }
    }

    auto end_time_generate_bitset = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_generate_bitset = end_time_generate_bitset - start_time_generate_bitset;
    print_status("Time Bitset generation: %f\n", duration_generate_bitset.count());
}

template<typename T, typename ITYPE>
void Residue<T, ITYPE>::generate_bitsets(ITYPE resolution)
{
    auto start_time_generate_bitset = std::chrono::high_resolution_clock::now();

    // Sparse computation with the variable precision / resolution
    bitset_active_cols_csr = new CSR<BitSet<ITYPE>, ITYPE>(this->nrows, this->ncols, this->nnz); // Create active cols csr
    bitset_active_reused_cols_csr = new CSR<BitSet<ITYPE>, ITYPE>(this->nrows, this->ncols, this->nnz); // Create active reused cols csr

    for ( ITYPE i = 0; i < this->nnz; i++ ) {
        new (&(bitset_active_cols_csr->vals[i])) BitSet<ITYPE>( Tj / resolution );
        new (&(bitset_active_reused_cols_csr->vals[i])) BitSet<ITYPE>( Tj / resolution );
    }

    bitset_active_rows_csr = new CSR<BitSet<ITYPE>, ITYPE>(this->nrows, this->ncols, this->nnz);
    bitset_active_reused_rows_csr = new CSR<BitSet<ITYPE>, ITYPE>(this->nrows, this->ncols, this->nnz);
    for ( ITYPE i = 0; i < this->nnz; i++ ) {
        new (&(bitset_active_rows_csr->vals[i])) BitSet<ITYPE>( Ti / resolution );
        new (&(bitset_active_reused_rows_csr->vals[i])) BitSet<ITYPE>( Ti / resolution );
    }

    std::memcpy( bitset_active_cols_csr->cols, res_raw_sparse->cols, sizeof(ITYPE) * res_raw_sparse->nnzs);
    std::memcpy( bitset_active_cols_csr->row_ptr, res_raw_sparse->row_ptr, sizeof(ITYPE) * (res_raw_sparse->nrows + 1) );
    std::memcpy( bitset_active_rows_csr->cols, res_raw_sparse->cols, sizeof(ITYPE) * res_raw_sparse->nnzs);
    std::memcpy( bitset_active_rows_csr->row_ptr, res_raw_sparse->row_ptr, sizeof(ITYPE) * (res_raw_sparse->nrows + 1) );

    std::memcpy( bitset_active_reused_cols_csr->cols, res_raw_sparse->cols, sizeof(ITYPE) * res_raw_sparse->nnzs);
    std::memcpy( bitset_active_reused_cols_csr->row_ptr, res_raw_sparse->row_ptr, sizeof(ITYPE) * (res_raw_sparse->nrows + 1) );

    std::memcpy( bitset_active_reused_rows_csr->cols, res_raw_sparse->cols, sizeof(ITYPE) * res_raw_sparse->nnzs );
    std::memcpy( bitset_active_reused_rows_csr->row_ptr, res_raw_sparse->row_ptr, sizeof(ITYPE) * (res_raw_sparse->nrows + 1) );


    ITYPE num_panels = CEIL(_spm->nrows, this->Ti);

    #pragma omp parallel for schedule(dynamic, 1)
    for ( ITYPE panel = 0; panel < num_panels; panel++ ) {
        ITYPE panel_start = panel * this->Ti;
        ITYPE panel_end = MIN(((panel + 1) * this->Ti), this->_spm->nrows);

        // ITYPE panel_num_active_tiled = this->bitset_active_cols_csr->row_ptr[panel] - this->bitset_active_cols_csr->row_ptr[panel + 1];
        for ( ITYPE row = panel_start; row < panel_end; row++ ) {
            ITYPE row_start = this->_spm->row_ptr[row];
            ITYPE row_end = this->_spm->row_ptr[row + 1];

            ITYPE residue_row_start = this->bitset_active_cols_csr->row_ptr[panel];
            ITYPE residue_row_end = this->bitset_active_cols_csr->row_ptr[panel + 1];

            // Iterate over the columns of the matrix row
            for ( ITYPE ptr = row_start; ptr < row_end; ptr++ ) {
                ITYPE col = this->_spm->cols[ptr];
                ITYPE curr_tile = col / this->Tj;

                // find tile in the current panel

                // binary search to find the col in the bitset
                ITYPE lo = residue_row_start;
                ITYPE hi = residue_row_end;
                while (lo <= hi) {
                    ITYPE mid = (lo + hi) / 2;
                    if ( curr_tile ==  this->bitset_active_cols_csr->cols[mid] ) {

                        if ( this->bitset_active_cols_csr->vals[mid].get( col % (this->Tj / resolution) ) ) {
                            this->bitset_active_reused_cols_csr->vals[mid].set( col % (this->Tj / resolution) );
                        } else {
                            this->bitset_active_cols_csr->vals[mid].set( col % (this->Tj / resolution) );
                        }

                        this->bitset_active_rows_csr->vals[mid].set( row % (this->Ti / resolution) );
                        break;
                    }

                    if ( curr_tile < this->bitset_active_cols_csr->cols[mid] ) {
                        hi = mid - 1;
                    } else {
                        lo = mid + 1;
                    }
                }


                /*
                for ( ITYPE tile_ptr = residue_row_start; tile_ptr < residue_row_end; tile_ptr++ ) {

                    // Find the current tile in the panel and set big high. Then break
                    if ( curr_tile == this->bitset_active_cols_csr->cols[tile_ptr] ) {
                        this->bitset_active_cols_csr->vals[tile_ptr].set( col % (Tj / resolution) );
                        this->bitset_active_rows_csr->vals[tile_ptr].set( row % (Ti / resolution) );

                        // if ( panel == 0 && curr_tile == 0 ) {
                        //     std::cout << "Setting ( " << row << ", " << col << ") -- " << col % (Tj / resolution) << std::endl;
                        //     set2.insert(col);
                        // }

                        break;
                    }
                }
                */

            }
        }
    }

    auto end_time_generate_bitset = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_generate_bitset = end_time_generate_bitset - start_time_generate_bitset;
    print_status("Time Bitset generation: %f\n", duration_generate_bitset.count());
}

template<typename ITYPE>
bool compare_range( std::pair<ITYPE, char> &a, std::pair<ITYPE, char> &b ) {
    return a.first <= b.first;
}

template<typename T, typename ITYPE>
void Residue<T, ITYPE>::generate_range_augmentation()
{
    this->active_cols_range = new CSR<RangeInfo<ITYPE>, ITYPE>(this->nrows, this->ncols, this->nnz, true);
    this->active_rows_range = new CSR<RangeInfo<ITYPE>, ITYPE>(this->nrows, this->ncols, this->nnz, true);

    for ( ITYPE i = 0; i < this->nnz; i++ ) {
        new ( &this->active_cols_range->vals[i] ) RangeInfo<ITYPE>();
        new ( &this->active_rows_range->vals[i] ) RangeInfo<ITYPE>();
    }

    this->active_cols_range->row_ptr = this->bitset_active_cols_csr->row_ptr;
    this->active_cols_range->cols = this->bitset_active_cols_csr->cols;
    this->active_rows_range->row_ptr = this->bitset_active_cols_csr->row_ptr;
    this->active_rows_range->cols = this->bitset_active_cols_csr->cols;


    ITYPE num_panels = CEIL( this->_spm->nrows, this->Ti );

    // #pragma omp parallel for num_threads(8)
    for ( ITYPE panel = 0; panel < num_panels; panel++ ) {
        ITYPE panel_start = panel * this->Ti;
        ITYPE panel_end = MIN(panel_start + this->Ti, this->_spm->nrows);

        for ( ITYPE row = panel_start; row < panel_end; row++ ) {
            assert( row >= 0 && "row should never be negative");
            ITYPE row_start = this->_spm->row_ptr[row];
            ITYPE row_end = this->_spm->row_ptr[row + 1];

            for ( ITYPE ptr = row_start; ptr < row_end; ptr++ ) {
                ITYPE col = this->_spm->cols[ptr];
                assert( col >= 0 && "col should never be negative");

                // ITYPE tile = this->ncols * panel + (col / this->Tj);
                ITYPE tile = col / this->Tj;

                for ( ITYPE res_ptr = this->active_cols_range->row_ptr[panel]; res_ptr < this->active_cols_range->row_ptr[panel+1]; res_ptr++ ) {

                    // TODO: make this tile finder more efficient
                    if ( this->active_cols_range->cols[res_ptr] == tile ) {
                        this->active_cols_range->vals[res_ptr].addRange( col, row );
                        this->active_rows_range->vals[res_ptr].addRange( row, col );

                        break;
                    }
                }
            }
        }
    }

    #ifdef DEBUG_RESIDUE_RANGE
        // for ( ITYPE res_row = 0; res_row < this->nrows; res_row++ ) {
        for ( ITYPE res_row = 0; res_row < 1; res_row++ ) {
            ITYPE res_row_start = this->active_cols_range->row_ptr[res_row];
            ITYPE res_row_end = this->active_cols_range->row_ptr[res_row + 1];

            for ( ITYPE res_ptr = res_row_start; res_ptr < res_row_end; res_ptr++ ) {
                ITYPE tile = this->active_cols_range->cols[res_ptr];

                auto &col_range_map = this->active_cols_range->vals[res_ptr].range;
                auto &row_range_map = this->active_rows_range->vals[res_ptr].range;

                // for ( auto &it : col_range_map ) {
                //     std::cout << "Tile: " << tile << " -- " << it.first << " -- " << it.second.first << " : " << it.second.second << std::endl;
                // }
                // std::cout << "Tile: " << tile << " -- " << " Active Cols: " << this->active_cols_range->vals[res_ptr].count() << std::endl;

                for ( auto &it : row_range_map ) {
                    std::cout << "Tile: " << tile << " -- " << it.first << " -- " << it.second.first << " : " << it.second.second << std::endl;
                }
                std::cout << "Tile: " << tile << " -- " << " Active Rows: " << this->active_rows_range->vals[res_ptr].count() << std::endl;
            }
        }
    #endif
}

// Greedy 2D adaptive
template<typename T, typename ITYPE>
std::vector<struct panel_t> Residue<T, ITYPE>::adaptive_2d_greedy_Ti_greedy_Tj_tile_generator( ITYPE feature, ITYPE cache_size, ITYPE cache_split, bool temporal_input, bool temporal_output, bool oi_aware )
{
    auto greedy_panel_start_time = std::chrono::high_resolution_clock::now();

    std::vector<struct panel_t> worklist;
    std::unordered_map<ITYPE, ITYPE> panel_height_hist;

    ITYPE max_output_cache_volume = (cache_size * cache_split) / CACHE_NUM_WAYS;

    ITYPE max_nnzs_per_panel = CEIL( this->_spm->nnzs, 64 );
    // ITYPE max_panel_height = CEIL( this->nrows, 64 );
    ITYPE max_panel_height = CEIL(8192,64); // I love magic numbers
    ITYPE band_count = 0;

    // ASPLOS 25 -- Band Matrix Detection Algorithm
    ITYPE band_detection_threshold = CEIL( cache_size, sizeof(T) * feature );   // number of rows that can be stored in the cache
    ITYPE band_detection_residue_threshold = CEIL( band_detection_threshold, this->Ti ); // number of residues that can be stored in the cache

    // print the band detection thresholds
    print_status("Band detection threshold: %ld\n", band_detection_threshold);
    print_status("Band detection residue threshold: %ld\n", band_detection_residue_threshold);

    print_status("Max output cache volume: %ld\n", max_output_cache_volume);
    BitSet<ITYPE> *panel_nars = new BitSet<ITYPE>[this->nrows];
    BitSet<ITYPE> *panel_nacs = new BitSet<ITYPE>[this->ncols];
    std::set<ITYPE> panel_active_cols_set;
    for ( ITYPE rt = 0; rt < this->nrows; rt++ ) { panel_nars[rt].init(this->Ti); }
    for ( ITYPE rt = 0; rt < this->ncols; rt++ ) { panel_nacs[rt].init(this->Tj); }

    RangeInfo<ITYPE> panel_row_range;
    ITYPE num_panels = 0;
    ITYPE curr_residue_row = 0; // points to the residue row that has not been added
    ITYPE max_row_panel_height = 0;


    // Adding resiliance in the tile builder by looking ahead?
    const ITYPE lookahead = 0;

    #ifdef CACHE_CONFLICT_ANALYSIS
        ITYPE num_ways = 8;
        ITYPE num_cache_blocks_per_way = cache_size / 64 / num_ways;
        ITYPE num_cache_blocks_per_row = (feature * sizeof(T)) / 64;

        // number of unique rows possible per way -- we have 8 ways so anything higher than an 8 count will cause a conflict
        ITYPE num_rows_per_way = num_cache_blocks_per_way / num_cache_blocks_per_row;

        std::vector<BitSetCounter<T, ITYPE>> panel_conflict_array;
        std::vector<std::vector<BitSetCounter<T, ITYPE>>> tile_conflict_array;
    #endif


    // build panels until we can
    while ( curr_residue_row < this->nrows ) {

        bool grow_panel = true;
        ITYPE curr_panel_start = curr_residue_row;  // start of the panel -- inclusive
        ITYPE curr_panel_end = curr_panel_start;    // end of the panel -- not inclusive
        ITYPE curr_fitting_panel = -1;  // end of the currently fitting panel -- not inclusive
        worklist.push_back( panel_t() );
        panel_t &new_panel = worklist[num_panels];
        #ifdef CACHE_CONFLICT_ANALYSIS
            panel_conflict_array.push_back( BitSetCounter<T, ITYPE>(num_rows_per_way) );
            // new_panel.cache_conflict.init(num_rows_per_way);
        #endif

        ITYPE new_panel_height = 0;
        ITYPE curr_panel_nnzs = 0;

        double new_panel_oi = 0;
        double curr_panel_oi = 0;

        while (grow_panel) {
            ITYPE new_res_row = curr_panel_end;
            new_panel_height = new_res_row - curr_panel_start + 1;
            if ( new_res_row >= this->nrows || curr_panel_nnzs >= max_nnzs_per_panel || new_panel_height > max_panel_height ) {
                break;
            }

            // check for band matrix detection
            if ( new_panel_height > band_detection_residue_threshold ) {

                std::vector<ITYPE> band_detection_residue_count;
                band_detection_residue_count.resize(band_detection_residue_threshold);

                std::vector<ITYPE> band_detection_residue_width;
                band_detection_residue_width.resize(band_detection_residue_threshold);
                for ( ITYPE i = (curr_panel_end - band_detection_residue_threshold); i < curr_panel_end; i++ ) {
                    band_detection_residue_count[i - (curr_panel_end - band_detection_residue_threshold)] = this->bitset_active_rows_csr->row_ptr[i+1] - this->bitset_active_rows_csr->row_ptr[i];

                    // get the min and max column value of the non-zero residue
                    ITYPE active_residue_col_start = this->bitset_active_cols_csr->cols[ this->bitset_active_rows_csr->row_ptr[i] ];
                    ITYPE active_residue_col_end = this->bitset_active_cols_csr->cols[ this->bitset_active_rows_csr->row_ptr[i+1] - 1 ];

                    band_detection_residue_width[i - (curr_panel_end - band_detection_residue_threshold)] = (active_residue_col_end - active_residue_col_start);
                }

                bool band_detected = true;
                // loop through and check that all the values in the residue count and the residue width are the same
                for ( ITYPE i = 1; i < band_detection_residue_threshold; i++ ) {
                    if ( band_detection_residue_count[i] != band_detection_residue_count[0] || band_detection_residue_width[i] != band_detection_residue_width[0] ) {
                        band_detected = false;

                        // print the band detection
                        // print_status("Band detection failed: %ld -- %ld\n", band_detection_residue_count[0], band_detection_residue_count[i]);

                        // print the same for residue width as well
                        // print_status("Band detection failed: %ld -- %ld\n", band_detection_residue_width[0], band_detection_residue_width[i]);

                        break;
                    }
                }

                if ( band_detected ) {
                    band_count++;
                    break;
                }
            }

            ITYPE new_panel_nnzs = curr_panel_nnzs;
            ITYPE res_row_start = this->bitset_active_rows_csr->row_ptr[new_res_row];
            ITYPE res_row_end = this->bitset_active_rows_csr->row_ptr[new_res_row + 1];

            for ( ITYPE act_res_ptr = res_row_start; act_res_ptr < res_row_end; act_res_ptr++ ) {
                panel_nars[ new_panel_height - 1 ] |= this->bitset_active_rows_csr->vals[act_res_ptr];

                if (temporal_output) {
                    panel_row_range |= this->active_rows_range->vals[act_res_ptr];
                }

                if (oi_aware) {
                    ITYPE act_res_col = this->bitset_active_cols_csr->cols[act_res_ptr];
                    new_panel_nnzs += this->res_raw_sparse->vals[act_res_ptr];
                    panel_nacs[ act_res_col ] |= this->bitset_active_cols_csr->vals[act_res_ptr];
                    panel_active_cols_set.insert(act_res_col);
                }
            }

            // compute the output volume based on the temporal output flag
            ITYPE panel_output_volume = feature * sizeof(T);
            if (temporal_output) {
                ITYPE panel_effective_active_rows_count = panel_row_range.count();
                panel_output_volume *= panel_effective_active_rows_count;
            } else {
                ITYPE panel_active_rows_count = 0;
                for ( ITYPE i = 0; i < new_panel_height; i++ ) {
                    panel_active_rows_count += panel_nars[i].count();
                }
                panel_output_volume *= panel_active_rows_count;
            }

            ITYPE panel_input_volume = feature * sizeof(T);
            if ( oi_aware ) {
                if ( temporal_input ) {
                    ITYPE panel_effective_active_cols_count = panel_active_cols_set.size();
                    panel_input_volume *= panel_effective_active_cols_count;
                } else {
                    ITYPE panel_active_cols_count = 0;
                    for ( auto &i : panel_active_cols_set) {
                        panel_active_cols_count += panel_nacs[i].count();
                    }
                    panel_input_volume *= panel_active_cols_count;
                }
                new_panel_oi = ((double )(2 * new_panel_nnzs * feature)) / ((double) ((panel_input_volume + panel_output_volume)));
            }


            if (oi_aware) {
                // if ( new_panel_oi < curr_panel_oi ) {
                //     if (new_panel_height == 1) {
                //         curr_panel_end++;
                //         new_panel.output_data_cached = (size_t) panel_output_volume;
                //     }
                //     curr_fitting_panel = curr_panel_end;

                //     if ( new_res_row > (curr_fitting_panel + lookahead) ) {
                //         grow_panel = false;
                //         break;
                //     }
                // } else if ( panel_output_volume > max_output_cache_volume ) {
                //     if ( new_panel_height == 1) {
                //         curr_panel_end++;
                //         new_panel.output_data_cached = (size_t) panel_output_volume;
                //     }
                //     grow_panel = false;
                //     break;
                // }

                if ( new_panel_oi < curr_panel_oi || panel_output_volume > max_output_cache_volume ) {
                    if ( new_panel_height == 1) {
                        curr_panel_end++;
                        new_panel.output_data_cached = (size_t) panel_output_volume;
                    }
                    // curr_panel_end++;
                    // new_panel.output_data_cached = (size_t) panel_output_volume;
                    grow_panel = false;
                    break;
                }
            } else {
                // can't add the new res row
                if ( panel_output_volume > max_output_cache_volume ) {
                    if ( new_panel_height == 1 ) {  // a panel needs to have at least 1 residue row
                        curr_panel_end++;
                        new_panel.output_data_cached = (size_t) panel_output_volume;
                    }

                    #ifdef CACHE_CONFLICT_ANALYSIS
                        for ( ITYPE rt = 0; rt < new_panel_height - 1; rt++ ) {
                            panel_conflict_array[num_panels].insert( panel_nars[rt] );
                        }
                    #endif

                    grow_panel = false;
                    break;
                }
            }
            curr_panel_end++;
            new_panel.output_data_cached = (size_t) panel_output_volume;
            curr_panel_nnzs = new_panel_nnzs;
            curr_panel_oi = new_panel_oi;
        }

        if (curr_fitting_panel != -1) {
            std::cout << "Looking ahead failed -- " << curr_fitting_panel << std::endl;
            curr_panel_end = curr_fitting_panel;
        }

        new_panel.start_row = curr_panel_start * this->Ti;
        new_panel.end_row = curr_panel_end * this->Ti;
        new_panel.Ti = new_panel.end_row - new_panel.start_row;
        new_panel.type = panel_type_t::P_2D;
        if ( new_panel.Ti > max_row_panel_height ) {
            max_row_panel_height = new_panel.Ti;
        }

        panel_height_hist[new_panel.Ti]++;

        #ifdef CACHE_CONFLICT_ANALYSIS
            if ( panel_conflict_array[num_panels].max_count() > num_ways ) {
                std::cout << "Panel: " << num_panels << " -- " << "Height: " << new_panel.Ti << " -- " << "max output conflict: " << panel_conflict_array[num_panels].max_count() << std::endl;
            }
        #endif

        num_panels++;
        curr_residue_row += (curr_panel_end - curr_panel_start);

        // cleanup for the next iteration
        for ( ITYPE rt = 0; rt < (new_panel_height + 1); rt++ ) { panel_nars[rt].reset(); }
        for ( auto &rt : panel_active_cols_set ) { panel_nacs[rt].reset(); }
        panel_row_range.reset();
    } // row panel builder loop

    auto greedy_panel_end_time = std::chrono::high_resolution_clock::now();
    print_status("Greedy Panel Generation Time: %f\n", std::chrono::duration<double>(greedy_panel_end_time - greedy_panel_start_time).count() );

    print_status("Total panels generated: %ld\n", num_panels);
    for (auto &it : panel_height_hist) {
        print_status("Panel height: %ld -- %ld\n", it.first, it.second);
    }
    print_status("Count of Band Panels detected: %ld\n", band_count);

    /*
    std::cout << "Num panels: " << num_panels;
    for ( auto &it : worklist ) {
        std::cout << "Panel start: " << it.start_row << " -- " << it.end_row << " -- " << it.output_data_cached << std::endl;
    }
    */

    ITYPE num_threads = omp_get_max_threads();
    ITYPE max_residues_per_panel = CEIL(max_row_panel_height, this->Ti);
    ITYPE max_residues_per_tile = this->ncols;
    BitSet<ITYPE> *tile_nacs_pool = new BitSet<ITYPE>[max_residues_per_tile * num_threads];
    BitSet<ITYPE> *tile_nars_pool = new BitSet<ITYPE>[max_residues_per_panel * num_threads];
    BitSet<ITYPE> *curr_tile_nars_pool = new BitSet<ITYPE>[max_residues_per_panel * num_threads];

    // BitSet<ITYPE> *tile_left_nars_pool = new BitSet<ITYPE>[max_residues_per_panel * num_threads];
    // BitSet<ITYPE> *tile_right_nars_pool = new BitSet<ITYPE>[max_residues_per_panel * num_threads];

    ITYPE *cached_res_ptr_pool = new ITYPE[ max_residues_per_panel * num_threads ];
    ITYPE *curr_cached_res_ptr_pool = new ITYPE[ max_residues_per_panel * num_threads ];

    RangeInfo<ITYPE> tile_col_range[num_threads];
    RangeInfo<ITYPE> curr_tile_col_range[num_threads];
    // RangeInfo<ITYPE> tile_row_range[num_threads];
    ITYPE local_num_tiles[num_threads];
    for (ITYPE i = 0; i < num_threads; i++) { local_num_tiles[i] = 0; }

    #pragma omp parallel for num_threads(num_threads)
    for (ITYPE rt = 0; rt < (max_residues_per_tile * num_threads); rt++) { tile_nacs_pool[rt].init(Tj); }

    #pragma omp parallel for num_threads(num_threads)
    for (ITYPE rt = 0; rt < (max_residues_per_panel * num_threads); rt++) {
        tile_nars_pool[rt].init(Ti);
        // tile_left_nars_pool[rt].init(Ti);
        // tile_right_nars_pool[rt].init(Ti);
        curr_tile_nars_pool[rt].init(Ti);
    }

    #ifdef CACHE_CONFLICT_ANALYSIS
        tile_conflict_array.resize(num_panels);
    #endif

    ITYPE DEBUG_PANEL = -1;

    // Tile processing loop -- runs in parallel
    #pragma omp parallel for num_threads(num_threads) schedule(dynamic, 4)
    for ( ITYPE panel = 0; panel < num_panels; panel++ ) { // panels
        auto tid = omp_get_thread_num();
        panel_t &my_panel = worklist[panel];

        #ifdef CACHE_CONFLICT_ANALYSIS
            std::vector<BitSetCounter<T, ITYPE>> &my_tile_conflict_array = tile_conflict_array[panel];
        #endif

        BitSet<ITYPE> *tile_nacs = &(tile_nacs_pool[tid * max_residues_per_tile]);
        BitSet<ITYPE> *tile_nars = &(tile_nars_pool[tid * max_residues_per_panel]);
        BitSet<ITYPE> *curr_tile_nars = &(curr_tile_nars_pool[tid * max_residues_per_panel]);
        // BitSet<ITYPE> *tile_left_nars = &(tile_left_nars_pool[tid * max_residues_per_panel]);
        // BitSet<ITYPE> *tile_right_nars = &(tile_right_nars_pool[tid * max_residues_per_panel]);

        ITYPE *cached_res_ptr = &(cached_res_ptr_pool[tid * max_residues_per_panel]);
        ITYPE *curr_cached_res_ptr = &(curr_cached_res_ptr_pool[tid * max_residues_per_panel]);

        ITYPE panel_row_start = CEIL(my_panel.start_row, this->Ti);
        ITYPE panel_row_end = CEIL(my_panel.end_row, this->Ti);
        // ITYPE num_residues_per_panel = CEIL((panel_row_end - panel_row_start), this->Ti);
        ITYPE num_residues_per_panel = CEIL(my_panel.Ti, this->Ti);
        ITYPE panel_input_effective_volume = cache_size - ((ITYPE) my_panel.output_data_cached);
        panel_input_effective_volume = MAX(panel_input_effective_volume, 0);

        // std::cout << "Panel: " << panel << " panel_row_start: " << panel_row_start << " -- panel_row_end: " << panel_row_end << std::endl;

        // std::cout << "Panel: " << panel << " -- " << panel_effective_active_rows_count << std::endl;

        // start building 2D tiles
        auto &panel_tiles = worklist[panel].tiles;
        ITYPE curr_res_col = 0;

        while ( curr_res_col < this->ncols ) {

            struct tile_t curr_tile;
            curr_tile.col_start = curr_res_col;
            curr_tile.col_end = curr_res_col;
            curr_tile.nnzs = curr_tile.nacs = curr_tile.nars = 0;
            curr_tile.input_volume = curr_tile.output_volume = 0;

            bool grow_tile = true;
            const ITYPE MAX_STRIDE = 1024;
            bool geometric_phase = true;
            ITYPE stride = 1; // stride gets reset for every panel
            ITYPE previous_stride = -1;

            for (ITYPE i = 0; i < num_residues_per_panel; i++) { cached_res_ptr[i] = -1; }

            while (grow_tile && curr_tile.col_end < this->ncols) {
                struct tile_t new_tile = curr_tile;
                new_tile.col_end += stride;

                if ( panel == DEBUG_PANEL ) {
                    print_status("Panel: %d -- Ti: %d -- Tile: %d -- Start: %d -- End: %d -- limit: %d -- new tile nnzs: %d\n", panel, my_panel.Ti, panel_tiles.size(), new_tile.col_start, new_tile.col_end, this->ncols, new_tile.nnzs);
                }

                new_tile.col_end = MIN(new_tile.col_end, this->ncols);

                for (ITYPE ii = panel_row_start; ii < panel_row_end; ii++) {

                    // std::cout << "Growing tile -- " << "Panel: " << panel << " -- " << ii;

                    ITYPE res_row_start = cached_res_ptr[ii % num_residues_per_panel] < 0 ? this->bitset_active_cols_csr->row_ptr[ii] : cached_res_ptr[ii % num_residues_per_panel];
                    ITYPE res_row_end = this->bitset_active_cols_csr->row_ptr[ii + 1];

                    // std::cout << " res_row_start: " << res_row_start << " -- res_row_end: " << res_row_end << std::endl;

                    for (ITYPE act_res_ptr = res_row_start; act_res_ptr < res_row_end; act_res_ptr++) {
                        ITYPE act_res_col = this->bitset_active_cols_csr->cols[act_res_ptr]; // raw column of the non-zero entry

                        // Within the tile that was promised to be built
                        if ( act_res_col >= new_tile.col_start && act_res_col < new_tile.col_end ) {
                            tile_nacs[ act_res_col ] |= this->bitset_active_cols_csr->vals[act_res_ptr];
                            tile_nars[ ii % num_residues_per_panel ] |= this->bitset_active_rows_csr->vals[act_res_ptr];
                            new_tile.nnzs += this->res_raw_sparse->vals[act_res_ptr];

                            if ( panel == DEBUG_PANEL ) {
                                print_status("res_row: %d, res col: %d nnzs added: %d, nacs in residue: %d\n", ii, act_res_col, this->res_raw_sparse->vals[act_res_ptr], this->bitset_active_cols_csr->vals[act_res_ptr].count());
                            }

                            if ( panel == DEBUG_PANEL ) {
                                print_status("Panel: %d -- Tile: %d -- NNZs: %d\n", panel, panel_tiles.size(), new_tile.nnzs);
                            }

                            if ( temporal_input ) {
                                tile_col_range[tid] |= this->active_cols_range->vals[act_res_ptr];
                            }

                            // ASPLOS 2025 - not using this in any computation
                            // if ( temporal_output ) {
                            //     tile_row_range[tid] |= this->active_rows_range->vals[act_res_ptr];
                            // }

                            // have to start from the next non-zero next time
                            cached_res_ptr[ii % num_residues_per_panel] = act_res_ptr + 1;

                        }

                        // ASPLOS 2025 NOTE : Not using the left and right nars for now
                        // else if (act_res_col < new_tile.col_start) {
                        //     tile_left_nars[ii % num_residues_per_panel] |= this->bitset_active_rows_csr->vals[act_res_ptr];
                        // } else if (act_res_col >= new_tile.col_end) {
                        //     tile_right_nars[ii % num_residues_per_panel] |= this->bitset_active_rows_csr->vals[act_res_ptr];
                        // }
                    }
                }

                // ASPLOS 2025 NOTE : Not using the left and right nars for now
                // ITYPE running_active_row_count = 0;
                // ITYPE active_row_killset_count = 0;
                // ITYPE active_row_genset_count = 0;
                // for ( ITYPE rt = 0; rt < num_residues_per_panel; rt++ ) {
                //     active_row_killset_count = (tile_left_nars[rt] & (~(tile_nars[rt] | tile_right_nars[rt]))).count(); // fetched from the left, but not in the curr and the right
                //     active_row_genset_count = (tile_nars[rt] & (~tile_left_nars[rt])).count();  // not fetched from the left, but in the curr
                //     running_active_row_count += ((tile_left_nars[rt] & tile_right_nars[rt]) + tile_nars[rt]).count();
                // }

                // panel_dense_input_cache_volume = cache_size - (running_active_row_count * sizeof(T) * feature);

                if (temporal_input) {
                    new_tile.nacs = tile_col_range[tid].count();
                } else {
                    new_tile.nacs = 0;
                    for (ITYPE rt = new_tile.col_start; rt < new_tile.col_end; rt++) { new_tile.nacs += tile_nacs[rt].count(); }
                }

                new_tile.nars = 0;
                for (ITYPE rt = 0; rt < num_residues_per_panel; rt++) { new_tile.nars += tile_nars[rt].count(); }
                if ( new_tile.nnzs == 0 ) {
                    if ( new_tile.nacs != 0 || new_tile.nars != 0) {
                        if ( new_tile.nacs != 0 ) {
                            print_status("Panel: %d -- Tile: %d -- Empty tile should have 0 nacs -- got: %d\n", panel, panel_tiles.size(), new_tile.nacs);
                        }

                        if ( new_tile.nars != 0 ) {
                            print_status("Panel: %d -- Tile: %d -- Empty tile should have 0 nars -- got: %d\n", panel, panel_tiles.size(), new_tile.nars);
                        }

                        print_error_exit("Please fix this!!!\n");
                    }
                }
                new_tile.input_volume = feature * sizeof(T) * new_tile.nacs;
                new_tile.output_volume = feature * sizeof(T) * new_tile.nars;

                if ( new_tile.nnzs > (new_tile.nars * new_tile.nacs) ) {
                    print_error_exit("New Tile Check failed -- Panel: %d -- Tile: %d -- Tile nnzs: %d -- NACS: %d, NARS: %d -- tile cannot have more non-zeros than dense\n", panel, panel_tiles.size(), new_tile.nnzs, new_tile.nacs, new_tile.nars);
                }


                if (new_tile.input_volume > panel_input_effective_volume) { // tile spills, can't add this most recent residue entry

                    if (geometric_phase) {
                        for ( ITYPE rt = 0; rt < num_residues_per_panel; rt++ ) { tile_nars[rt] = curr_tile_nars[rt]; }
                        for ( ITYPE rt = 0; rt < num_residues_per_panel; rt++ ) { cached_res_ptr[rt] = curr_cached_res_ptr[rt]; }
                        if (temporal_input) { tile_col_range[tid] = curr_tile_col_range[tid]; }

                        if ( stride == 1 ) {
                            geometric_phase = false;
                        } else {
                            stride = MAX(stride / 2, 1);
                            continue;
                        }
                    }

                    if (curr_tile.nnzs == 0) {
                        curr_tile = new_tile;
                    }

                    if ( !temporal_input && ( curr_tile.nnzs > (curr_tile.nacs * curr_tile.nars) ) ) {
                        print_status("tile_col_start: %d, tile_col_end: %d\n", curr_tile.col_start, curr_tile.col_end);
                        print_error_exit("Panel: %d -- Tile: %d -- Tile nnzs: %d -- NACS: %d, NARS: %d -- tile cannot have more non-zeros than dense\n", panel, panel_tiles.size(), curr_tile.nnzs, curr_tile.nacs, curr_tile.nars);
                    }

                    grow_tile = false;

                    break;
                } else {
                    curr_tile = new_tile;
                    for (ITYPE rt = 0; rt < num_residues_per_panel; rt++) { curr_tile_nars[rt] = tile_nars[rt]; }
                    for ( ITYPE rt = 0; rt < num_residues_per_panel; rt++ ) { curr_cached_res_ptr[rt] = cached_res_ptr[rt]; }
                    if (temporal_input) { curr_tile_col_range[tid] = tile_col_range[tid]; }
                }

                #ifdef CACHE_CONFLICT_ANALYSIS
                    ITYPE tiles_added = (init_spill ? num_residues_per_tile : num_residues_per_tile-1);
                    my_tile_conflict_array.push_back( BitSetCounter<T, ITYPE>(num_rows_per_way) );
                    for ( ITYPE rt = 0; rt < tiles_added; rt++ ) {
                        my_tile_conflict_array.back().insert( tile_nacs[rt] );
                    }

                    for ( ITYPE rt = 0; rt < num_residues_per_panel; rt++ ) {
                        my_tile_conflict_array.back().insert( tile_nars[rt] );
                    }
                #endif

                // grow the stride
                if ( geometric_phase ) {
                    stride *= 2;
                } else {
                    stride = 1;
                }
            }   // tile grow loop

            // tile is done -- add it to the panel
            ITYPE num_residues_per_tile = MIN((curr_tile.col_end - curr_tile.col_start + stride), max_residues_per_tile);
            for ( ITYPE rt = 0; rt < max_residues_per_tile; rt++ ) { tile_nacs[rt].reset(); }
            for ( ITYPE rt = 0; rt < num_residues_per_panel; rt++ ) {
                tile_nars[rt].reset();
                // tile_left_nars[rt].reset();
                // tile_right_nars[rt].reset();
            }

            if (temporal_input) { tile_col_range[tid].reset(); }
            // if (temporal_output) { tile_row_range[tid].reset(); }

            curr_res_col = curr_tile.col_end;
            if ( curr_tile.nnzs ) {  // add to the panel only if there is a non-zero entry
                curr_tile.col_start *= this->Tj;
                curr_tile.col_end *= this->Tj;
                panel_tiles.push_back( curr_tile );
            }  else {
                if ( curr_tile.col_end < this->ncols ) {
                    std::cout << "panel: " << panel << " -- curr tile: [" << curr_tile.col_start << ", " << curr_tile.col_end << ")" << " -- " << "NNZs: " << curr_tile.nnzs << " -- " << "NACS: " << curr_tile.nacs << " -- " << "NARS: " << curr_tile.nars << std::endl;
                    print_error_exit( "Empty tile should be at the end of the panel" );
                }
            }
        }
        my_panel.num_tiles = panel_tiles.size();
        local_num_tiles[tid] += panel_tiles.size();
    }

    ITYPE global_num_tiles = 0;
    for ( ITYPE i = 0; i < num_threads; i++ ) {
        global_num_tiles += local_num_tiles[i];
    }

    #ifdef CACHE_CONFLICT_ANALYSIS
        for ( ITYPE panel = 0; panel < num_panels; panel++ ) {
            auto &my_tile_conflict_array = tile_conflict_array[panel];
            for ( ITYPE tile = 0; tile < my_tile_conflict_array.size(); tile++ ) {
                if ( my_tile_conflict_array[tile].max_count() > num_ways ) {
                    std::cout << "Panel: " << panel << " -- " << "Tile: " << tile << " -- " << "max input conflict: " << my_tile_conflict_array[tile].max_count() << std::endl;
                }
            }
        }
    #endif

    print_status("Total tiles created: %ld\n", global_num_tiles);

    // delete[] tile_left_nars_pool;
    // delete[] tile_right_nars_pool;
    delete[] tile_nacs_pool;
    delete[] tile_nars_pool;
    delete[] panel_nars;
    delete[] curr_tile_nars_pool;
    delete[] cached_res_ptr_pool;
    delete[] curr_cached_res_ptr_pool;

    return worklist;
}

#endif // RESDIUE_H

