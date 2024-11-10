#ifndef CSR_H
#define CSR_H

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <set>

#include "config.h"
#include "utils/util.h"

// #define DEBUG_SPARSE_MATRIX

template <typename T = double, typename ITYPE = int64_t>
class CSR
{

public:
    ITYPE nrows; // number of rows
    ITYPE ncols; // number of columns
    ITYPE nnzs;  // number of non-zeroes

    T *vals;
    ITYPE *row_ptr;
    ITYPE *cols;

    // Only used when the CSR matrix is augmented with panel information
    ITYPE num_panels = 0;
    ITYPE *panel_ptr = nullptr;
    ITYPE *panel_Tk = nullptr;

    bool vals_only = false;

    ITYPE get_nrows() const { return nrows; }
    ITYPE get_ncols() const { return ncols; }
    ITYPE get_nnzs() const { return nnzs; }

    CSR() : nrows(0), ncols(0), nnzs(0) {}
    ~CSR();
    CSR(ITYPE nr, ITYPE nc, ITYPE nz, std::pair<ITYPE, ITYPE> *pairs, T *vals, bool inplace = false);
    CSR(ITYPE nr, ITYPE nc, ITYPE nnz, bool vals_only = false);
    CSR(ITYPE nr, ITYPE nc);
    CSR( CSR<T, ITYPE> &other );

    void print_matrix();
    void augment_panel_ptrs(ITYPE num_panels, struct workitem *worklist);
    inline void init_memory(ITYPE nnzs);

    size_t get_size_bytes()
    {
        return sizeof(ITYPE) * (nrows + 1) + sizeof(ITYPE) * (nnzs) + sizeof(T) * (nnzs);
    }
};

template<typename T, typename ITYPE>
CSR<T, ITYPE>::~CSR()
{
    FREE_MEMORY(vals);

    if (!vals_only) {
        FREE_MEMORY(row_ptr);
        FREE_MEMORY(cols);
        FREE_MEMORY(panel_ptr);
        FREE_MEMORY(panel_Tk);
    }
}

// Constructor to just allocate memory and call it a day
template <typename T, typename ITYPE>
CSR<T, ITYPE>::CSR(ITYPE nr, ITYPE nc, ITYPE nz, bool vals_only) : nrows(nr), ncols(nc), nnzs(nz), vals_only(vals_only)
{
    // allocate arrays and call it a day
    this->vals = (T *)std::aligned_alloc(ALLOC_ALIGNMENT, nz * sizeof(T));
    std::memset(this->vals, 0, nz * sizeof(T));

    if (!vals_only) {
        this->cols = (ITYPE *)std::aligned_alloc(64, nz * sizeof(ITYPE));
        this->row_ptr = (ITYPE *)std::aligned_alloc(64, (nr + 1) * sizeof(ITYPE));

        assert(this->vals && this->cols && this->row_ptr && "could not allocate CSR matrix");

        std::memset(this->cols, 0, nz * sizeof(ITYPE));
        std::memset(this->row_ptr, 0, (nr + 1) * sizeof(ITYPE));
    }
}

template <typename T, typename ITYPE>
CSR<T, ITYPE>::CSR(ITYPE nrows, ITYPE ncols) : nrows(nrows), ncols(ncols)
{
    this->row_ptr = (ITYPE *) std::aligned_alloc(ALLOC_ALIGNMENT, sizeof(ITYPE) * (this->nrows + 1));
    std::memset(this->row_ptr, 0, sizeof(ITYPE) * (this->nrows + 1));
}

template <typename T, typename ITYPE>
void CSR<T, ITYPE>::init_memory(ITYPE nnzs)
{
    this->nnzs = nnzs;
    this->cols = (ITYPE *) std::aligned_alloc(ALLOC_ALIGNMENT, sizeof(ITYPE) * nnzs);
    this->vals = (T *) std::aligned_alloc(ALLOC_ALIGNMENT, sizeof(T) * nnzs);
}

template <typename T, typename ITYPE>
CSR<T, ITYPE>::CSR(ITYPE nrows, ITYPE ncols, ITYPE nnzs, std::pair<ITYPE, ITYPE> *pairs, T *v, bool inplace) : nrows(nrows), ncols(ncols), nnzs(nnzs)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    struct v_struct *temp = new struct v_struct[nnzs];

    #pragma omp parallel for
    for (ITYPE i = 0; i < this->nnzs; i++) {
        temp[i].grp = 0;
        temp[i].row = pairs[i].first;
        temp[i].col = pairs[i].second;
        temp[i].val = v[i];
    }

    // sort in row-major order
    std::qsort(temp, nnzs, sizeof(struct v_struct), compare1);

    this->row_ptr = (ITYPE *) ALLOC_MEMORY( (this->nrows + 1) * sizeof(ITYPE) );
    std::memset( this->row_ptr, 0, (this->nrows + 1) * sizeof(ITYPE) );
    this->cols = (ITYPE *) ALLOC_MEMORY( this->nnzs * sizeof(ITYPE) );
    this->vals = (T *) ALLOC_MEMORY( this->nnzs * sizeof(T) );

    for (ITYPE i = 0; i < this->nnzs; i++) {
        this->row_ptr[temp[i].row + 1]++;
        this->cols[i] = temp[i].col;
        this->vals[i] = temp[i].val;
    }

    for (ITYPE i = 1; i < (this->nrows + 1); i++) {
        this->row_ptr[i] += this->row_ptr[i - 1];
    }

    delete[] temp;

#ifdef DEBUG_SPARSE_MATRIX
        print_matrix();
#endif

    auto end_time = std::chrono::high_resolution_clock::now();

    print_status("CSR Construction -- M: %ld, N: %ld, NNZ: %ld -- time: %f\n", this->nrows, this->ncols, this->nnzs, std::chrono::duration<double>(end_time - start_time).count());
}

template <typename T, typename ITYPE>
CSR<T, ITYPE>::CSR( CSR<T, ITYPE> &other ) : nrows(other.nrows), ncols(other.ncols), nnzs(other.nnzs)
{
    this->row_ptr = (ITYPE *) std::aligned_alloc(ALLOC_ALIGNMENT, sizeof(ITYPE) * (this->nrows + 1));
    this->cols = (ITYPE *) std::aligned_alloc(ALLOC_ALIGNMENT, sizeof(ITYPE) * this->nnzs);
    this->vals = (T *) std::aligned_alloc(ALLOC_ALIGNMENT, sizeof(T) * this->nnzs);

    #pragma omp parallel for schedule(static, 8)
    for (ITYPE i = 0; i < (this->nrows + 1); i++) {
        this->row_ptr[i] = other.row_ptr[i];
    }

    #pragma omp parallel for schedule(static, 8)
    for (ITYPE i = 0; i < this->nnzs; i++) {
        this->cols[i] = other.cols[i];
        // this->vals[i] = 0;
        this->vals[i] = other.vals[i];
    }
}

template <typename T, typename ITYPE>
void CSR<T, ITYPE>::augment_panel_ptrs(ITYPE num_panels, struct workitem *worklist)
{
    this->num_panels = num_panels;
    this->panel_ptr = (ITYPE *) std::aligned_alloc(ALLOC_ALIGNMENT, sizeof(ITYPE) * (num_panels + 1));
    this->panel_Tk = (ITYPE *) std::aligned_alloc(ALLOC_ALIGNMENT, sizeof(ITYPE) * (num_panels + 1));
    std::memset(this->panel_ptr, 0, sizeof(ITYPE) * (num_panels + 1));
    for (ITYPE panel = 0; panel < (num_panels + 1); panel++) {
        this->panel_ptr[panel] = worklist[panel].start_row;
        this->panel_Tk[panel] = worklist[panel].Tk;
    }
}

template <typename T, typename ITYPE>
void CSR<T, ITYPE>::print_matrix()
{

    for (ITYPE r = 0; r < nrows; r++)
    {
        ITYPE row_start = row_ptr[r];
        ITYPE row_end = row_ptr[r + 1];
        std::cerr << r << ": ";
        for (ITYPE i = row_start; i < row_end; i++)
        {
            ITYPE c = cols[i];
            std::cerr << c << "(" << vals[i] << ")"
                      << " ";
            // std::cerr << "(" << r << ", " << c << ")";
        }
        std::cerr << std::endl;
    }
}

template <typename T, typename ITYPE>
class SPLIT_CSR
{
public:
    ITYPE partition_size; // Size of each veritcal parition
    ITYPE num_partitions; // Number of vertical splits of a panel
    ITYPE panel_height;   // Number of rows in a row panel

    ITYPE nrows, ncols, nnzs;

    ITYPE *row_ptr; // how do we index into this?
                    // Indexing scheme
                    // partition * nrows + row

    ITYPE *cols; // column value of the non-zero
    T *vals;     // value of the non-zero

    SPLIT_CSR(ITYPE nrows, ITYPE ncols, ITYPE nnzs, std::pair<ITYPE, ITYPE> *locs, T *vals, ITYPE num_partitions);
};

// Let's pass this is deduplicated list
template <typename T, typename ITYPE>
SPLIT_CSR<T, ITYPE>::SPLIT_CSR(ITYPE nrows, ITYPE ncols, ITYPE nnzs, std::pair<ITYPE, ITYPE> *locs, T *vals, ITYPE num_partitions) : nrows(nrows), ncols(ncols), nnzs(nnzs), num_partitions(num_partitions)
{
    this->partition_size = CEIL(ncols, num_partitions);
    std::cout << "Num Partitions: " << num_partitions << std::endl;
    std::cout << "Parition Size: " << partition_size << std::endl;
    this->panel_height = -1;

    struct v_struct *temp = new struct v_struct[nnzs];
    ITYPE real_nnz = 0;
    for (ITYPE i = 0; i < nnzs; i++)
    {
        if (i >= 1 && ((locs[i].first == locs[i - 1].first) && (locs[i].second == locs[i - 1].second)))
        {
            continue;
        }
        temp[real_nnz].row = locs[i].first;
        temp[real_nnz].col = locs[i].second;
        temp[real_nnz].val = vals[i];
        temp[real_nnz].grp = temp[real_nnz].col / partition_size;

        real_nnz++;
    }
    this->nnzs = nnzs = real_nnz;

    std::qsort(temp, nnzs, sizeof(struct v_struct), compare1);

    ITYPE size_row_ptr = (num_partitions * nrows) + 1;
    this->row_ptr = (ITYPE *)std::aligned_alloc(ALLOC_ALIGNMENT, size_row_ptr * sizeof(ITYPE));
    std::memset(this->row_ptr, 0, size_row_ptr * sizeof(ITYPE));
    this->cols = (ITYPE *)std::aligned_alloc(ALLOC_ALIGNMENT, nnzs * sizeof(ITYPE));
    std::memset(this->cols, 0, nnzs * sizeof(ITYPE));
    this->vals = (T *)std::aligned_alloc(ALLOC_ALIGNMENT, nnzs * sizeof(T));
    std::memset(this->vals, 0, nnzs * sizeof(T));

    // Iterate over the matrix and figure out
    for (ITYPE i = 0; i < nnzs; i++)
    {

        ITYPE segment_multiplier = temp[i].grp;
        ITYPE row = temp[i].row;

        this->row_ptr[row + (segment_multiplier * nrows) + 1]++;
    }
    // print_arr<ITYPE>( this->row_ptr, size_row_ptr, "row_ptr", 0, 1 );

    // Add row ptrs to their respective place
    for (ITYPE i = 1; i < size_row_ptr; i++)
    {
        this->row_ptr[i] += this->row_ptr[i - 1];
    }

    ITYPE prev_row = -1;
    ITYPE prev_partition = -1;
    ITYPE num_processed = 0;
    for (ITYPE i = 0; i < nnzs; i++)
    {
        ITYPE curr_row = temp[i].row;
        ITYPE curr_partition = temp[i].grp;

        if (prev_partition != curr_partition || curr_row != prev_row)
        {
            num_processed = 0;
        }

        ITYPE index = this->row_ptr[curr_row + (nrows * curr_partition)];
        this->cols[index + num_processed] = temp[i].col;
        this->vals[index + num_processed] = temp[i].val;
        num_processed++;

        prev_row = curr_row;
        prev_partition = curr_partition;
    }

    // print_arr<ITYPE>( this->row_ptr, size_row_ptr, "row_ptr", 0, 1 );
    // print_arr<ITYPE>( this->cols, nnzs, "cols", 0, 1 );
}

template <typename T, typename ITYPE>
void generate_csr_histogram(CSR<T, ITYPE> &A, ITYPE Ti, ITYPE feature)
{
    ITYPE num_panels = CEIL(A.nrows, Ti);
    std::map<ITYPE, ITYPE> size_histogram;
    std::map<ITYPE, ITYPE> nnzs_histogram;

    // nnzs, nars, nacs, size
    std::map<std::tuple<ITYPE, ITYPE, ITYPE, ITYPE>, ITYPE> tile_histogram;
    for (ITYPE panel = 0; panel < num_panels; panel++) {
        ITYPE start_row = panel * Ti;
        ITYPE end_row = MIN((panel + 1) * Ti, A.nrows);

        ITYPE tile_nnzs = 0;
        std::set<ITYPE> tile_rows;
        std::set<ITYPE> tile_cols;

        for (ITYPE row = start_row; row < end_row; row++) {
            tile_rows.insert(row);
            ITYPE row_start = A.row_ptr[row];
            ITYPE row_end = A.row_ptr[row + 1];
            for (ITYPE ptr = row_start; ptr < row_end; ptr++) {
                tile_nnzs++;
                tile_cols.insert(A.cols[ptr]);
            }
        }

        ITYPE tile_nars = tile_rows.size();
        ITYPE tile_nacs = tile_cols.size();
        ITYPE tile_size = (tile_nacs + tile_nars) * feature * sizeof(T);

        size_histogram[tile_size]++;
        nnzs_histogram[tile_nnzs]++;
        tile_histogram[std::make_tuple(tile_nnzs, tile_nars, tile_nacs, tile_size)]++;
    }

    std::cout << "BEGIN TILE INFO MAP" << std::endl;
    std::cout << "NNZS, NARS, NACS, SIZE, COUNT" << std::endl;
    for (auto &tile : tile_histogram) {
        std::cout << std::get<0>(tile.first) << ", " << std::get<1>(tile.first) << ", " << std::get<2>(tile.first) << ", " << std::get<3>(tile.first) << ", " << tile.second << std::endl;
    }
    std::cout << "END TILE INFO MAP" << std::endl;

    std::cout << "BEGIN TILE SIZE HISTOGRAM" << std::endl;
    std::cout << "SIZE, COUNT" << std::endl;
    for (auto &size : size_histogram) {
        std::cout << size.first << ", " << size.second << std::endl;
    }
    std::cout << "END TILE SIZE HISTOGRAM" << std::endl;

    std::cout << "BEGIN TILE NNZS HISTOGRAM" << std::endl;
    std::cout << "NNZS, COUNT" << std::endl;
    for (auto &nnzs : nnzs_histogram) {
        std::cout << nnzs.first << ", " << nnzs.second << std::endl;
    }
    std::cout << "END TILE NNZS HISTOGRAM" << std::endl;

}

#endif // CSR_H
