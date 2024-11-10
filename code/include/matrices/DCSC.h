#ifndef DCSC_H
#define DCSC_H

#include <cstdint>
#include <cstring>
#include <fstream>
#include <list>
#include <map>
#include <utility>

#include "utils/util.h"

// #define DEBUG_PRINT_MATRIX
#define UPPER_BOUND 128

template<typename T = double, typename ITYPE = int64_t>
class DCSC {

public:
    ITYPE nrows, nrows_orig;
    ITYPE ncols;
    ITYPE nnzs;

public:

    ITYPE *aux; // Stores active rows
    ITYPE *cols, *rows, *col_ptr;
    ITYPE *panel_Tk;
    T* vals;

    ITYPE segment_size; // Number of rows in a row panel (Ti)
    ITYPE num_segments;
    ITYPE dcount_copy;

    ITYPE get_nrows() const { return nrows; }
    ITYPE get_ncols() const { return ncols; }
    ITYPE get_nnzs() const { return nnzs; }

    DCSC() : nrows(0), ncols(0), nnzs(0) {}
    DCSC(ITYPE nrows, ITYPE ncols, ITYPE nnzs, std::pair<ITYPE, ITYPE> *pairs, T *vals, ITYPE segment_size = 64);
    DCSC( DCSC<T, ITYPE> &other );

    // Use this to construct a variable panel height dcsc array
    DCSC(ITYPE nrows, ITYPE ncosl, ITYPE nnzs, std::pair<ITYPE, ITYPE> *locs, T *vals, ITYPE num_panels, struct workitem *worklist);
    DCSC(ITYPE nrows, ITYPE ncosl, ITYPE nnzs, std::pair<ITYPE, ITYPE> *locs, T *vals, ITYPE num_panels, std::list<struct workitem> worklist);
    ~DCSC();

    std::map<std::pair<ITYPE, ITYPE>, ITYPE, comp> print();

    bool is_equal_structure( DCSC<T, ITYPE> &other );
};

template <typename T, typename ITYPE>
bool DCSC<T, ITYPE>::is_equal_structure( DCSC<T, ITYPE> &other )
{
    if ( (this->nnzs != other.nnzs) || (this->nrows != other.nrows) || (this->ncols != other.ncols) ) {
        return false;
    }
    if ( (this->num_segments != other.num_segments) || (this->aux[this->num_segments] != other.aux[other.num_segments]) ) {
        return false;
    }

    for ( ITYPE panel = 0; panel < this->num_segments; panel++ ) {

        if ( (this->aux[panel] != other.aux[panel]) || (this->aux[panel+1] != other.aux[panel+1]) ) {
            return false;
        }

        ITYPE panel_start = this->aux[ panel ];
        ITYPE panel_end = this->aux[ panel + 1];
        for ( ITYPE panel_ptr = panel_start; panel_ptr < panel_end; panel_ptr++ ) {
            if ( (this->col_ptr[panel_ptr] != other.col_ptr[panel_ptr]) || (this->col_ptr[panel_ptr+1] != other.col_ptr[panel_ptr+1]) || (this->cols[panel_ptr] != other.cols[panel_ptr]) ) {
                return false;
            }
            ITYPE col_start = this->col_ptr[panel_ptr];
            ITYPE col_end = this->col_ptr[panel_ptr + 1];
            for ( ITYPE col_ptr = col_start; col_ptr < col_end; col_ptr++ ) {
                if (this->rows[col_ptr] != other.rows[col_ptr]) {
                    return false;
                }
            }
        }
    }

    return true;
}

template<typename T, typename ITYPE>
DCSC<T, ITYPE>::DCSC(ITYPE nrows, ITYPE ncols, ITYPE nnzs, std::pair<ITYPE, ITYPE> *pairs, T *vals, ITYPE segment_size) : nrows(nrows), ncols(ncols), nnzs(nnzs)
{
    this->segment_size = segment_size;
    this->num_segments = CEIL( nrows, segment_size );

    struct v_struct *temp = new struct v_struct[nnzs];

    // #pragma omp parallel for num_threads(8) schedule(static, 8)
    for (ITYPE i = 0; i < this->nnzs; i++) {
        temp[i].row = pairs[i].first;
        temp[i].col = pairs[i].second;
        temp[i].val = vals[i];
        temp[i].grp = 0;
    }

    /*
    ITYPE real_nnz = 0;
    for (ITYPE i = 0; i < nnzs; i++) {
        if ( i >= 1 && ( (pairs[i].first == pairs[i - 1].first) && (pairs[i].second == pairs[i - 1].second) ) ) {
            continue;
        }
        temp[real_nnz].row = pairs[i].first;
        temp[real_nnz].col = pairs[i].second;
        temp[real_nnz].val = vals[i];

        real_nnz++;
    }
    this->nnzs = nnzs = real_nnz;

    */

    // std::cout << "Real nnzs: " << nnzs << std::endl;

    std::qsort(temp, nnzs, sizeof(struct v_struct), compare2);

    // std::sort( temp, temp + nnzs, [](const struct v_struct &a, const struct v_struct &b) -> bool {
    //     if (a.grp > b.grp) { return true; }
    //     if (a.grp < b.grp) { return false; }
    //     if (a.col > b.col) { return true; }
    //     if (a.col < b.col) { return false; }
    //     return a.row > b.row;
    // } );


    // grp -> row panel the element belongs to
    for (ITYPE i = 0; i < nnzs; i++) {
        temp[i].grp = temp[i].row / segment_size;
    }

    this->aux = (ITYPE *) std::aligned_alloc( ALIGNED_ACCESS, sizeof(ITYPE) * (this->num_segments + 1) );
    std::memset( this->aux, 0, sizeof(ITYPE) * (this->num_segments + 1) );

    // std::cout << "Number of segments: " << num_segments << std::endl;

    ITYPE temp_panel = -1;
    ITYPE temp_col = -1;
    for ( ITYPE i = 0; i < nnzs; i++ ) {
        // If not the same panel or not the same column then active column for the temp[i].grp gets updated
        if ( temp[i].grp != temp_panel || temp[i].col != temp_col ) {
            this->aux[ temp[i].grp + 1 ]++;
            temp_panel = temp[i].grp;
            temp_col = temp[i].col;
        }
    }

    assert(aux[0] == 0 && "aux array is not setup correctly");

    for ( ITYPE i = 1; i < (this->num_segments + 1); i++ ) {
        this->aux[i] += aux[i - 1];
    }
    this->aux[0] = 0;

    ITYPE *temp_aux = new ITYPE[num_segments + 1]();
    std::memcpy( temp_aux, this->aux, sizeof(ITYPE) * (this->num_segments + 1) );


    ITYPE num_active_cols = aux[num_segments];
    this->cols = (ITYPE *) std::aligned_alloc( ALIGNED_ACCESS, sizeof(ITYPE) * num_active_cols );
    this->col_ptr = (ITYPE *) std::aligned_alloc( ALIGNED_ACCESS, sizeof(ITYPE) * (num_active_cols + 1) );
    std::memset( this->col_ptr, 0, sizeof(ITYPE) * (num_active_cols + 1) );
    this->rows = (ITYPE *) std::aligned_alloc( ALIGNED_ACCESS, sizeof(ITYPE) * nnzs );
    this->vals = (T *) std::aligned_alloc( ALIGNED_ACCESS, sizeof(T) * nnzs );

    col_ptr[0] = -1;
    temp_panel = -1;
    temp_col = -1;
    for ( ITYPE i = 0; i < nnzs; i++ ) {
        if ( temp[i].grp != temp_panel || temp[i].col != temp_col ) {
            cols[ temp_aux[ temp[i].grp ] ] = temp[i].col;
            col_ptr[ temp_aux[ temp[i].grp ] ]++;
            temp_aux[ temp[i].grp ]++;
            temp_panel = temp[i].grp;
            temp_col = temp[i].col;
        } else {
            col_ptr[ temp_aux[ temp[i].grp ] ]++;
        }
    }

    // setup the col_ndx array pointers
    for ( ITYPE i = 1; i < (num_active_cols + 1); i++ ) {
        col_ptr[ i ] += col_ptr[i - 1];
    }
    col_ptr[ num_active_cols ]++;

    ITYPE *temp_col_ptr = new ITYPE[ num_active_cols + 1 ];
    std::memcpy( temp_col_ptr, col_ptr, sizeof(ITYPE) * (num_active_cols + 1) );
    std::memcpy( temp_aux, aux, sizeof(ITYPE) * (num_segments + 1) );
    for ( ITYPE i = 0; i < nnzs; i++ ) {
        ITYPE ndx = temp_aux[ temp[i].grp ];
        ITYPE row_ndx = temp_col_ptr[ ndx ];
        this->rows[ row_ndx ] = temp[i].row;
        this->vals[ row_ndx ] = temp[i].val;

        (temp_col_ptr[ndx])++;

        if ( temp_col_ptr[ ndx ] == temp_col_ptr[ ndx + 1 ] ) {
            (temp_aux[ temp[i].grp ])++;
        }
    }

    // free temporary memory
    delete[] temp;
    delete[] temp_col_ptr;
    delete[] temp_aux;

    // #ifdef DEBUG_PRINT_MATRIX
    //     print_arr<ITYPE, ITYPE>(aux,  num_segments + 1, "aux");
    //     print_arr<ITYPE, ITYPE>(col_ndx, dcount, "col_ndx");
    //     print_arr<ITYPE, ITYPE>(cols, dcount, "cols");
    //     print_arr<ITYPE, ITYPE>(rows, nnzs, "rows");
    //     print_arr<T, ITYPE>(vals, nnzs, "vals");
    // #endif
}

template <typename T, typename ITYPE>
DCSC<T, ITYPE>::DCSC( DCSC<T, ITYPE> &other ) : nrows(other.nrows), ncols(other.ncols), nnzs(other.nnzs), num_segments(other.num_segments)
{
    this->aux = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof( ITYPE ) * (this->num_segments + 1) );
    ITYPE num_col_ptrs = other.aux[ num_segments ];

    std::cout << "Other.num_col_ptrs: " << other.aux[ num_segments ] << std::endl;

    this->col_ptr = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE) * (num_col_ptrs + 1) );
    this->cols = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE) * (num_col_ptrs) );
    this->rows = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE) * this->nnzs );
    this->vals = (T *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(T) * this->nnzs );

    // #pragma omp parallel for num_threads(8) schedule(static, 8)
    for ( ITYPE i = 0; i < (num_segments + 1); i++ ) {
        this->aux[i] = other.aux[i];
    }

    // #pragma omp parallel for num_threads(8) schedule(static, 8)
    for ( ITYPE i = 0; i < num_col_ptrs; i++ ) {
        this->col_ptr[i] = other.col_ptr[i];
        this->cols[i] = other.cols[i];
    }
    this->col_ptr[num_col_ptrs] = other.col_ptr[num_col_ptrs];

    // #pragma omp parallel for num_threads(8) schedule(static, 8)
    for ( ITYPE i = 0; i < this->nnzs; i++ ) {
        this->rows[i] = other.rows[i];
        this->vals[i] = 0;
    }
}

// Variable panel height dcsc array
template<typename T, typename ITYPE>
DCSC<T, ITYPE>::DCSC(ITYPE nrows, ITYPE ncols, ITYPE nnzs, std::pair<ITYPE, ITYPE> *pairs, T *vals, ITYPE num_panels, struct workitem *worklist) : nrows(nrows), ncols(ncols), nnzs(nnzs)
// DCSC<T, ITYPE>::DCSC(ITYPE nrows, ITYPE ncols, ITYPE nnzs, std::pair<ITYPE, ITYPE> *pairs, T *vals, ITYPE num_panels, std::list<struct workitem> worklist) : nrows(nrows), ncols(ncols), nnzs(nnzs)
{
    this->num_segments = num_panels;

    struct v_struct *temp = new struct v_struct[nnzs];
    
    // #pragma omp parallel for num_threads(8) schedule(static, 8)
    for (ITYPE i = 0; i < this->nnzs; i++) {
        temp[i].row = pairs[i].first;
        temp[i].col = pairs[i].second;
        temp[i].val = vals[i];
        temp[i].grp = 0;
    }

    /*
    ITYPE real_nnz = 0;
    for (ITYPE i = 0; i < nnzs; i++) {
        if ( i >= 1 && ( (pairs[i].first == pairs[i - 1].first) && (pairs[i].second == pairs[i - 1].second) ) ) {
            continue;
        }
        temp[real_nnz].row = pairs[i].first;
        temp[real_nnz].col = pairs[i].second;
        temp[real_nnz].val = vals[i];

        real_nnz++;
    }
    this->nnzs = nnzs = real_nnz;

    */

    // std::cout << "Real nnzs: " << nnzs << std::endl;

    this->panel_Tk = (ITYPE *) std::aligned_alloc(ALLOC_ALIGNMENT, sizeof(ITYPE) * this->num_segments);
    for (ITYPE i = 0; i < num_panels; i++) {
        this->panel_Tk[i] = worklist[i].Tk;
    }

    std::qsort(temp, nnzs, sizeof(struct v_struct), compare2);

    // grp -> row panel the element belongs to
    for (ITYPE i = 0; i < nnzs; i++) {

        // check the worklist to figure out the non-zero mapping
        for (ITYPE pid = 0; pid < num_panels; pid++) {
            if ( (temp[i].row >= worklist[pid].start_row) && (temp[i].row < worklist[pid].end_row) ) {
                temp[i].grp = pid;
                break;
            }
        }
    }

    this->aux = (ITYPE *) std::aligned_alloc( ALIGNED_ACCESS, sizeof(ITYPE) * (this->num_segments + 1) );
    std::memset( this->aux, 0, sizeof(ITYPE) * (this->num_segments + 1) );

    // std::cout << "Number of segments: " << num_segments << std::endl;

    ITYPE temp_panel = -1;
    ITYPE temp_col = -1;
    for ( ITYPE i = 0; i < nnzs; i++ ) {
        // If not the same panel or not the same column then active column for the temp[i].grp gets updated
        if ( temp[i].grp != temp_panel || temp[i].col != temp_col ) {
            this->aux[ temp[i].grp + 1 ]++;
            temp_panel = temp[i].grp;
            temp_col = temp[i].col;
        }
    }

    assert(aux[0] == 0 && "aux array is not setup correctly");

    for ( ITYPE i = 1; i < (this->num_segments + 1); i++ ) {
        this->aux[i] += aux[i - 1];
    }
    this->aux[0] = 0;

    ITYPE *temp_aux = new ITYPE[num_segments + 1]();
    std::memcpy( temp_aux, this->aux, sizeof(ITYPE) * (this->num_segments + 1) );


    ITYPE num_active_cols = aux[num_segments];
    this->cols = (ITYPE *) std::aligned_alloc( ALIGNED_ACCESS, sizeof(ITYPE) * num_active_cols );
    this->col_ptr = (ITYPE *) std::aligned_alloc( ALIGNED_ACCESS, sizeof(ITYPE) * (num_active_cols + 1) );
    std::memset( this->col_ptr, 0, sizeof(ITYPE) * (num_active_cols + 1) );
    this->rows = (ITYPE *) std::aligned_alloc( ALIGNED_ACCESS, sizeof(ITYPE) * nnzs );
    this->vals = (T *) std::aligned_alloc( ALIGNED_ACCESS, sizeof(T) * nnzs );

    col_ptr[0] = -1;
    temp_panel = -1;
    temp_col = -1;
    for ( ITYPE i = 0; i < nnzs; i++ ) {
        if ( temp[i].grp != temp_panel || temp[i].col != temp_col ) {
            cols[ temp_aux[ temp[i].grp ] ] = temp[i].col;
            col_ptr[ temp_aux[ temp[i].grp ] ]++;
            temp_aux[ temp[i].grp ]++;
            temp_panel = temp[i].grp;
            temp_col = temp[i].col;
        } else {
            col_ptr[ temp_aux[ temp[i].grp ] ]++;
        }
    }

    // setup the col_ndx array pointers
    for ( ITYPE i = 1; i < (num_active_cols + 1); i++ ) {
        col_ptr[ i ] += col_ptr[i - 1];
    }
    col_ptr[ num_active_cols ]++;

    ITYPE *temp_col_ptr = new ITYPE[ num_active_cols + 1 ];
    std::memcpy( temp_col_ptr, col_ptr, sizeof(ITYPE) * (num_active_cols + 1) );
    std::memcpy( temp_aux, aux, sizeof(ITYPE) * (num_segments + 1) );
    for ( ITYPE i = 0; i < nnzs; i++ ) {
        ITYPE ndx = temp_aux[ temp[i].grp ];
        ITYPE row_ndx = temp_col_ptr[ ndx ];
        this->rows[ row_ndx ] = temp[i].row;
        this->vals[ row_ndx ] = temp[i].val;

        (temp_col_ptr[ndx])++;

        if ( temp_col_ptr[ ndx ] == temp_col_ptr[ ndx + 1 ] ) {
            (temp_aux[ temp[i].grp ])++;
        }
    }

    // free temporary memory
    delete[] temp;
    delete[] temp_col_ptr;
    delete[] temp_aux;
}

template<typename T, typename ITYPE>
DCSC<T, ITYPE>::~DCSC()
{
    if (cols) { std::free(cols); }
    if (col_ptr) { std::free(col_ptr); }
    if (aux) { std::free(aux); }
    if (vals) { std::free(vals); }
    if (rows) { std::free(rows); }
}

template<typename T, typename ITYPE>
std::map<std::pair<ITYPE, ITYPE>, ITYPE, comp> DCSC<T, ITYPE>::print()
{

    std::map<std::pair<ITYPE, ITYPE>, ITYPE, comp> non_zero_pairs;
    std::fstream nnz_file("dscr.out", std::fstream::out);

    // Lets print this matrix and verify correctness
    // Collect all the non-zero key value pairs in a map and output them
    for (ITYPE row_panel = 0; row_panel < this->num_segments; row_panel++) {
        if (aux[row_panel] == aux[row_panel+1]) { continue; }

        // Iterate over the csr matrix
        for (ITYPE j = aux[row_panel]; j < aux[row_panel + 1]; j++) {
            ITYPE col = cols[j]; // Non-zero row
            ITYPE num_rows = col_ptr[j + 1] - col_ptr[j];

            for (ITYPE i = 0; i < num_rows; i++) {
                ITYPE row = rows[ col_ptr[j] + i ];

                auto point = std::pair<ITYPE, ITYPE>(row, col);
                if (non_zero_pairs.find(point) == non_zero_pairs.end()) {
                    non_zero_pairs[point] = 0;
                }
                non_zero_pairs[point]++;
            }
        }
    }

    // print nnz access count
    for (auto iter : non_zero_pairs) {
        nnz_file << "(" << iter.first.first << ", " << iter.first.second << ") -- " << iter.second << std::endl;
    }

    nnz_file.close();

    return non_zero_pairs;
}



#endif // DCSC_H
