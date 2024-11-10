#ifndef MATRICES_ATM_H
#define MATRICES_ATM_H

#include <cstdlib>
#include <map>
#include <set>
#include <vector>

#include "common.h"
#include "config.h"
#include "utils/util.h"

// #define SPECIAL_THRESHOLD 2048
// Adaptive Tiled Matrix

template<typename T, typename ITYPE>
static ITYPE find_panel(ITYPE row, std::vector<struct panel_t> &panels);

template<typename T = double, typename ITYPE = int64_t>
class ATM {

    private:
        void mark_special_rows(std::vector<panel_t> &panels);

    public:
        // const ITYPE SPECIAL_THRESHOLD = 512;    // NUM_THREADS * CACHE_BLOCK_SIZE / sizeof(T)
        bool run_special = false;
        ITYPE special_count = 0;
        ITYPE nrows, ncols, nnzs;
        ITYPE num_panels;

        ITYPE *panel_Ti;        // height of each panel
        ITYPE *panel_offset;    // panel offset into the tile_row_ptr array

        ITYPE *row_ptr;         // CSR row-ptr can be used for traditional iteration
        ITYPE *panel_ptr;       // number of tiles in each panel offset
        ITYPE *panel_start;     // starting row of each panel

        ITYPE *tile_row_ptr;    // row ptrs for each tile
        ITYPE *cols;            // non-zero cols
        T *vals;                // non-zero values


        #if defined(RUN_ASPT_SPECIAL) || defined(ASPT_SPECIAL_SIMD_PARALLEL)
            ITYPE *special_row_ndx;
            ITYPE *special_ptr;
            ITYPE *special_row_panel;
        #endif


        ATM(ITYPE nrows, ITYPE ncols, ITYPE nnzs, std::pair<ITYPE, ITYPE> *pairs, T *vals, ITYPE num_panels, struct panel_t *panels);
        ATM(ITYPE nrows, ITYPE ncols, ITYPE nnzs, std::pair<ITYPE, ITYPE> *pairs, T *vals, ITYPE num_panels, std::vector<struct panel_t> &panels);
        ATM(ITYPE nrows, ITYPE ncols, ITYPE nnzs, std::pair<ITYPE, ITYPE> *pairs, T *vals, std::vector<struct panel_t> &panels);
        ~ATM();
};

template<typename T, typename ITYPE>
ATM<T, ITYPE>::~ATM()
{
    if ( panel_ptr ) { std::free(panel_ptr); }
    if ( panel_offset ) { std::free(panel_offset); }
    if ( panel_start ) { std::free(panel_start); }
    if ( panel_Ti ) { std::free(panel_Ti); }
    if ( row_ptr ) { std::free(row_ptr); }
    if ( cols ) { std::free(cols); }
    if ( vals ) { std::free(vals); }
    if ( tile_row_ptr ) { std::free(tile_row_ptr); }
}

template<typename T, typename ITYPE>
void ATM<T, ITYPE>::mark_special_rows(std::vector<panel_t> &panels)
{
    #if defined(RUN_ASPT_SPECIAL) || defined(ASPT_SPECIAL_SIMD_PARALLEL)
        // print_status("Marking special rows\n");
        ITYPE num_special_rows = 0;
        this->special_count = 0;
        for ( ITYPE i = 0; i < nrows; i++ ) {
            if ( row_ptr[i + 1] - row_ptr[i] >= SPECIAL_THRESHOLD ) { // found a special row
                this->special_count += FLOOR( (row_ptr[i + 1] - row_ptr[i]), SPECIAL_THRESHOLD );
                // print_status("Special Row: %ld\n", i);
                num_special_rows++;

                // std::cout << "Special row: " << i << " nnzs: " << (row_ptr[i + 1] - row_ptr[i]) << std::endl;
            }
        }

        print_status("Special Row Count: %ld\n", num_special_rows);
        this->special_row_ndx = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE) * this->special_count );
        this->special_ptr = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE) * this->special_count );
        this->special_row_panel = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE) * this->special_count );

        ITYPE ptr = 0;
        for ( ITYPE i = 0; i < nrows; i++ ) {
            ITYPE row_nnzs = row_ptr[i + 1] - row_ptr[i];
            if ( row_nnzs >= SPECIAL_THRESHOLD ) {
                ITYPE special_row_panel_ndx = find_panel<T, ITYPE>(i, panels);

                auto &special_row_panel_reference = panels[special_row_panel_ndx];

                // print the special row panel before clearing it
                // std::cout << "Special row: " << i << ", " << "Special Row Panel: " << special_row_panel_ndx << std::endl;
                // for ( auto &tile : special_row_panel_reference.tiles ) {
                //     std::cout << "Tile: " << tile.col_start << " " << tile.col_end << " " << tile.nnzs << std::endl;
                // }

                // go over all the tile_row_ptrs of the special row panel and make their value equal to the next rows start

                ITYPE row_num_specials = FLOOR( row_nnzs, SPECIAL_THRESHOLD );
                #if defined(RUN_ASPT_SPECIAL_DENSE_OPT)
                    for ( ITYPE tile = 0; tile < special_row_panel_reference.num_tiles ; tile++ ) {
                        ITYPE ptr = this->panel_offset[special_row_panel_ndx] + (i - this->panel_start[special_row_panel_ndx]) * special_row_panel_reference.num_tiles + tile;
                        // tile_row_ptr[ptr] = tile_row_ptr[ptr + 1];
                        if (tile == 0) {
                            tile_row_ptr[ptr + 1] = tile_row_ptr[ptr] + row_num_specials * SPECIAL_THRESHOLD;
                        } else if (tile_row_ptr[ptr] < tile_row_ptr[ptr - tile + 1]) {
                            tile_row_ptr[ptr] = tile_row_ptr[ptr - tile + 1];
                        }
                    }
                #else
                    special_row_panel_reference.tiles.clear();
                    special_row_panel_reference.tiles.push_back( {0, this->ncols, 0, 0, 0, 0, 0} );
                    special_row_panel_reference.num_tiles = 1;
                #endif
                for ( ITYPE j = 0; j < row_num_specials; j++ ) {
                    this->special_row_ndx[ptr] = i;
                    this->special_row_panel[ptr] =  special_row_panel_ndx;
                    this->special_ptr[ptr] = (j * SPECIAL_THRESHOLD);
                    ptr++;
                }
            }
        }

        if (this->special_count) { this->run_special = true; }
        print_status("Can Execute: %ld Segments of %ld Size in Special\n", this->special_count, SPECIAL_THRESHOLD);
    #endif
}

template<typename T, typename ITYPE>
static ITYPE find_panel_atm(ITYPE row, ITYPE num_panels, struct panel_t *panels)
{
    ITYPE panel = 0;
    for ( ITYPE i = 0; i < num_panels; i++ ) {
        if ( row < panels[i].end_row ) {
            panel = i;
            break;
        }
    }
    return panel;
}


template<typename T, typename ITYPE>
ATM<T, ITYPE>::ATM(ITYPE nrows, ITYPE ncols, ITYPE nnzs, std::pair<ITYPE, ITYPE> *pairs, T *vals, ITYPE num_panels, struct panel_t *panels) : nrows(nrows), ncols(ncols), nnzs(nnzs), num_panels(num_panels)
{
    this->panel_ptr = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE) * (num_panels + 1) );
    this->row_ptr = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE) * (nrows + 1) );
    this->cols = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE) * nnzs );
    this->vals = (T *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(T) * nnzs );

    // required for indexing into the panel_ptr
    this->panel_offset = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE) * (num_panels + 1) );
    this->panel_start = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE) * (num_panels) );
    this->panel_Ti = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE) * num_panels );


    struct v_struct *temp = new struct v_struct[nnzs];

    #pragma omp parallel for num_threads(8)
    for (ITYPE i = 0; i < this->nnzs; i++) {
        temp[i].row = pairs[i].first;
        temp[i].col = pairs[i].second;
        temp[i].val = vals[i];
        temp[i].grp = find_panel_atm<T, ITYPE>( temp[i].row, num_panels, panels );
        temp[i].col_grp = temp[i].col / panels[ temp[i].grp ].Tj;
    }

    std::qsort( temp, nnzs, sizeof(struct v_struct), compare1 );    // sorted in row-major order

    #pragma omp parallel for num_threads(8)
    for ( ITYPE i = 0; i < (nrows+1); i++ ) {
        row_ptr[i] = 0;
    }

    #pragma omp parallel for num_threads(8)
    for ( ITYPE i = 0; i < nnzs; i++ ) {
        row_ptr[ temp[i].row + 1 ]++;
        cols[i] = temp[i].col;
        this->vals[i] = temp[i].val;
    }

    for ( ITYPE i = 1; i < (nrows + 1); i++ ) {
        row_ptr[i] += row_ptr[i - 1];
    }
    row_ptr[0] = 0;

    // Setup tile row ptrs
    std::map<ITYPE, std::set<ITYPE>> panel_active_tiles;
    for ( ITYPE i = 0; i < nnzs; i++ ) {
        if ( panel_active_tiles.find( temp[i].grp ) == panel_active_tiles.end() ) {
            panel_active_tiles[ temp[i].grp ] = std::set<ITYPE>();
        }

        if ( panel_active_tiles[ temp[i].grp ].find( temp[i].col_grp ) == panel_active_tiles[ temp[i].grp ].end() ) {
            panel_active_tiles[ temp[i].grp ].insert( temp[i].col_grp );
        }
    }

    for ( ITYPE i = 0; i < ( num_panels + 1 ); i++ ) {
        panel_ptr[i] = 0;
        panel_offset[i] = 0;
    }

    ITYPE tile_ptr_array_size = 0;

    for ( ITYPE i = 0; i < num_panels; i++ ) {
        auto p = panel_active_tiles.find(i);
        if ( p != panel_active_tiles.end() ) {
            panel_ptr[i + 1] = p->second.size();
            panel_offset[i + 1] = p->second.size() * panels[i].Ti;
            tile_ptr_array_size += p->second.size() * panels[i].Ti;
        }
        panel_start[i] = panels[i].start_row;
        panel_Ti[i] = panels[i].Ti;
    }

    for ( ITYPE i = 1; i < ( num_panels + 1 ); i++ ) {
        panel_ptr[i] += panel_ptr[i - 1];
        panel_offset[i] += panel_offset[i - 1];
    }

    tile_row_ptr = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE) * (tile_ptr_array_size) );

    #pragma omp parallel for num_threads(8)
    for ( ITYPE i = 0; i < ( tile_ptr_array_size + 1); i++ ) {
        tile_row_ptr[i] = 0;
    }

    for ( ITYPE i = 0; i < nrows; i++ ) {
        auto row_start = row_ptr[i];
        auto row_end = row_ptr[i + 1];

        ITYPE tile_id = -1;
        ITYPE curr_tile = -1;
        ITYPE curr_row_grp = find_panel_atm<T, ITYPE>( i, num_panels, panels );

        std::set<ITYPE> &panel_active_tiles_set = panel_active_tiles[ curr_row_grp ];
        auto panel_num_tiles = panel_ptr[curr_row_grp + 1] - panel_ptr[curr_row_grp];
        auto panel_Ti = panels[ curr_row_grp ].Ti;
        auto panel_Tj = panels[ curr_row_grp ].Tj;

        for ( ITYPE ptr = row_start; ptr < row_end; ptr++ ) {
            ITYPE curr_col_grp = cols[ptr] / panel_Tj;
            if ( curr_tile == -1 || curr_col_grp != curr_tile ) {
                curr_tile = curr_col_grp;
                tile_id = 0;

                for ( auto &tile : panel_active_tiles_set ) {
                    if ( tile < curr_tile ) { tile_id ++; }
                }
            }
            // ITYPE update_ptr = panel_offset[ curr_row_grp ] + ( (i % panel_Ti) * panel_num_tiles ) + tile_id;
            ITYPE update_ptr = panel_offset[ curr_row_grp ] + ( (i - panel_start[curr_row_grp]) * panel_num_tiles ) + tile_id;
            // std::cout << "Adding element: (" << i << ", " << cols[ptr] << ")" << " panel: " << curr_row_grp << " tile: " << curr_col_grp << " ptr: " << update_ptr << std::endl;
            tile_row_ptr[ update_ptr + 1]++;
        }
    }

    tile_row_ptr[0] = 0;
    for ( ITYPE i = 1; i < (tile_ptr_array_size + 1); i++ ) {
        tile_row_ptr[i] += tile_row_ptr[i - 1];
    }

        /*
        print_arr<ITYPE>( row_ptr, nrows + 1, "row_ptr" );
        print_arr<ITYPE>( cols, nnzs, "cols: " );
        print_arr<ITYPE>( panel_ptr, num_panels + 1, "panel_ptr" );
        print_arr<ITYPE>( panel_start, num_panels, "panel_start" );
        print_arr<ITYPE>( panel_offset, num_panels, "panel_offset" );
        print_arr<ITYPE>( panel_Ti, num_panels, "panel_Ti" );
        print_arr<ITYPE>( tile_row_ptr, tile_ptr_array_size, "tile_row_ptr" );

        */
}



template<typename T, typename ITYPE>
static ITYPE find_panel(ITYPE row, std::vector<struct panel_t> &panels)
{
    ITYPE panel = 0;
    ITYPE num_panels = panels.size();
    if ( num_panels < BIN_SEARCH_THRESHOLD ) {
        for ( ITYPE i = 0; i < num_panels; i++ ) {
            if ( row >= panels[i].start_row && row < panels[i].end_row ) {
                panel = i;
                break;
            }
        }
    } else {
        ITYPE lo = 0;
        ITYPE hi = num_panels - 1;
        while ( lo <= hi ) {
            panel = (lo + hi) / 2;
            if ( row >= panels[panel].start_row && row < panels[panel].end_row ) {
                break;
            }

            if ( row < panels[panel].start_row ) {
                hi = panel - 1;
            } else {
                lo = panel + 1;
            }
        }
    }
    return panel;
}

template<typename T, typename ITYPE>
static ITYPE find_tile(ITYPE col, struct panel_t &panel)
{
    ITYPE tile = 0;
    for ( ITYPE i = 0; i < panel.tiles.size(); i++ ) {
        if ( col >= panel.tiles[i].col_start && col < panel.tiles[i].col_end ) {
            tile = i;
            break;
        }
    }
    return tile;
}

template<typename T, typename ITYPE>
ATM<T, ITYPE>::ATM(ITYPE nrows, ITYPE ncols, ITYPE nnzs, std::pair<ITYPE, ITYPE> *pairs, T *vals, ITYPE num_panels, std::vector<struct panel_t> &panels) : nrows(nrows), ncols(ncols), nnzs(nnzs), num_panels(num_panels)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    this->panel_ptr = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE) * (num_panels + 1) );
    this->row_ptr = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE) * (nrows + 1) );
    this->cols = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE) * nnzs );
    this->vals = (T *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(T) * nnzs );

    // required for indexing into the panel_ptr
    this->panel_offset = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE) * (num_panels + 1) );
    this->panel_start = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE) * (num_panels) );
    this->panel_Ti = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE) * num_panels );

    struct v_struct *temp = new struct v_struct[nnzs];

    // #pragma omp parallel for num_threads(8)
    for (ITYPE i = 0; i < this->nnzs; i++) {
        temp[i].row = pairs[i].first;
        temp[i].col = pairs[i].second;
        temp[i].val = vals[i];
        temp[i].grp = find_panel<T, ITYPE>( temp[i].row, panels );
        temp[i].col_grp = find_tile<T, ITYPE>( temp[i].col, panels[temp[i].grp] );
    }

    std::sort( temp, temp + nnzs, [](const struct v_struct &a, const struct v_struct &b) {
        if (a.row != b.row) { return a.row < b.row;}
        return a.col < b.col;
    });

    #pragma omp parallel for num_threads(8)
    for ( ITYPE i = 0; i < (nrows+1); i++ ) {
        row_ptr[i] = 0;
    }

    for ( ITYPE i = 0; i < nnzs; i++ ) {
        this->row_ptr[ temp[i].row + 1 ]++;
        this->cols[i] = temp[i].col;
        this->vals[i] = temp[i].val;
    }

    for ( ITYPE i = 1; i < (nrows + 1); i++ ) {
        row_ptr[i] += row_ptr[i - 1];
    }
    row_ptr[0] = 0;

    // Setup tile row ptrs
    std::map<ITYPE, std::set<ITYPE>> panel_active_tiles;
    for ( ITYPE i = 0; i < nnzs; i++ ) {
        if ( panel_active_tiles.find( temp[i].grp ) == panel_active_tiles.end() ) {
            panel_active_tiles[ temp[i].grp ] = std::set<ITYPE>();
        }

        if ( panel_active_tiles[ temp[i].grp ].find( temp[i].col_grp ) == panel_active_tiles[ temp[i].grp ].end() ) {
            panel_active_tiles[ temp[i].grp ].insert( temp[i].col_grp );
        }
    }

    for ( ITYPE i = 0; i < ( num_panels + 1 ); i++ ) {
        panel_ptr[i] = 0;
        panel_offset[i] = 0;
    }

    ITYPE tile_ptr_array_size = 0;

    for ( ITYPE i = 0; i < num_panels; i++ ) {
        auto p = panel_active_tiles.find(i);
        if ( p != panel_active_tiles.end() ) {
            panel_ptr[i + 1] = p->second.size();
            panel_offset[i + 1] = p->second.size() * panels[i].Ti;
            tile_ptr_array_size += p->second.size() * panels[i].Ti;
        }
        panel_start[i] = panels[i].start_row;
        panel_Ti[i] = panels[i].Ti;
    }

    for ( ITYPE i = 1; i < ( num_panels + 1 ); i++ ) {
        panel_ptr[i] += panel_ptr[i - 1];
        panel_offset[i] += panel_offset[i - 1];
    }

    tile_row_ptr = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE) * (tile_ptr_array_size + 1) );

    #pragma omp parallel for num_threads(8)
    for ( ITYPE i = 0; i < ( tile_ptr_array_size + 1); i++ ) {
        tile_row_ptr[i] = 0;
    }

    // TOOD: Check if we can use the row-major temp array to avoid the repeated find_panel and find_tile lookups
    for ( ITYPE i = 0; i < nrows; i++ ) {
        auto row_start = row_ptr[i];
        auto row_end = row_ptr[i + 1];

        ITYPE tile_id = -1;
        ITYPE curr_tile = -1;
        ITYPE curr_row_grp = find_panel<T, ITYPE>( i, panels );

        std::set<ITYPE> &panel_active_tiles_set = panel_active_tiles[ curr_row_grp ];
        auto panel_num_tiles = panel_ptr[curr_row_grp + 1] - panel_ptr[curr_row_grp];
        auto panel_Ti = panels[ curr_row_grp ].Ti;
        // auto panel_Tj = panels[ curr_row_grp ].Tj;

        for ( ITYPE ptr = row_start; ptr < row_end; ptr++ ) {
            // ITYPE curr_col_grp = cols[ptr] / panel_Tj;

            // find the curr col grp
            // ITYPE curr_col_grp = -1;
            // for ( ITYPE col_grp = 0; col_grp < panels[curr_row_grp].var_Tj.size(); col_grp++ ) {
            //     if ( cols[ptr] >= panels[curr_row_grp].var_Tj[col_grp].first && cols[ptr] < panels[curr_row_grp].var_Tj[col_grp].second ) {
            //         curr_col_grp = col_grp;
            //         break;
            //     }
            // }

            ITYPE curr_col_grp = find_tile<T, ITYPE>( cols[ptr], panels[curr_row_grp] );

            if ( curr_tile == -1 || curr_col_grp != curr_tile ) {
                curr_tile = curr_col_grp;
                tile_id = 0;

                // for ( auto &tile : panel_active_tiles_set ) {
                //     if ( tile < curr_tile ) { tile_id ++; }
                // }
            }
            // ITYPE update_ptr = panel_offset[ curr_row_grp ] + ( (i % panel_Ti) * panel_num_tiles ) + tile_id;
            // ITYPE update_ptr = panel_offset[ curr_row_grp ] + ( (i - panel_start[curr_row_grp]) * panel_num_tiles ) + tile_id;

            ITYPE update_ptr = panel_offset[ curr_row_grp ] + ( (i - panel_start[curr_row_grp]) * panel_num_tiles ) + curr_tile;


            // std::cout << "Adding element: (" << i << ", " << cols[ptr] << ")" << " panel: " << curr_row_grp << " tile: " << curr_col_grp << " ptr: " << update_ptr << std::endl;
            tile_row_ptr[ update_ptr + 1]++;
        }
    }

    tile_row_ptr[0] = 0;
    for ( ITYPE i = 1; i < (tile_ptr_array_size + 1); i++ ) {
        tile_row_ptr[i] += tile_row_ptr[i - 1];
    }

    /*
        print_arr<ITYPE>( row_ptr, nrows + 1, "row_ptr" );
        print_arr<ITYPE>( cols, nnzs, "cols: " );
        print_arr<ITYPE>( panel_ptr, num_panels + 1, "panel_ptr" );
        print_arr<ITYPE>( panel_start, num_panels, "panel_start" );
        print_arr<ITYPE>( panel_offset, num_panels, "panel_offset" );
        print_arr<ITYPE>( panel_Ti, num_panels, "panel_Ti" );
        print_arr<ITYPE>( tile_row_ptr, tile_ptr_array_size, "tile_row_ptr" );
    */
    delete[] temp;
    auto end_time = std::chrono::high_resolution_clock::now();

    print_status("ATM Construction Time: %fs\n", std::chrono::duration<double>(end_time - start_time).count());
}

template<typename T, typename ITYPE>
ATM<T, ITYPE>::ATM(ITYPE nrows, ITYPE ncols, ITYPE nnzs, std::pair<ITYPE, ITYPE> *pairs, T *vals, std::vector<struct panel_t> &panels) : nrows(nrows), ncols(ncols), nnzs(nnzs), num_panels(panels.size())
{
    auto start_time = std::chrono::high_resolution_clock::now();

    this->panel_ptr = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE) * (num_panels + 1) );
    this->row_ptr = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE) * (nrows + 1) );
    this->cols = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE) * nnzs );
    this->vals = (T *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(T) * nnzs );

    // required for indexing into the panel_ptr
    this->panel_offset = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE) * (num_panels + 1) ); // panel offset into the tile_row_ptr array
    this->panel_start = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE) * (num_panels) );      // starting row of each panel
    this->panel_Ti = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE) * num_panels );           // height of each panel

    struct v_struct *temp = new struct v_struct[nnzs];

    // #pragma omp parallel for num_threads(8)
    for (ITYPE i = 0; i < this->nnzs; i++) {
        temp[i].row = pairs[i].first;
        temp[i].col = pairs[i].second;
        temp[i].val = vals[i];
        temp[i].grp = find_panel<T, ITYPE>( temp[i].row, panels );
        temp[i].col_grp = find_tile<T, ITYPE>( temp[i].col, panels[temp[i].grp] );
    }

    std::sort( temp, temp + nnzs, [](const struct v_struct &a, const struct v_struct &b) {
        if (a.row != b.row) { return a.row < b.row;}
        return a.col < b.col;
    });

    #pragma omp parallel for num_threads(8)
    for ( ITYPE i = 0; i < (nrows+1); i++ ) { row_ptr[i] = 0; }

    for ( ITYPE i = 0; i < nnzs; i++ ) {
        this->row_ptr[ temp[i].row + 1 ]++;
        this->cols[i] = temp[i].col;
        this->vals[i] = temp[i].val;
    }

    for ( ITYPE i = 1; i < (nrows + 1); i++ ) {
        row_ptr[i] += row_ptr[i - 1];
    }
    row_ptr[0] = 0;

    #if (defined(RUN_ASPT_SPECIAL) || defined(ASPT_SPECIAL_SIMD_PARALLEL)) && !defined(RUN_ASPT_SPECIAL_DENSE_OPT)
        mark_special_rows(panels);
    #endif


    // recompute the tile grps since things have now changed and panels with dense rows have become CSR i.e. one tile only
    for (ITYPE i = 0; i < this->nnzs; i++) {
        temp[i].col_grp = find_tile<T, ITYPE>( temp[i].col, panels[temp[i].grp] );
    }

    // setup panel ptr and tile row ptr arrays
    for ( ITYPE i = 0; i < ( num_panels + 1 ); i++ ) {
        panel_ptr[i] = 0;
        panel_offset[i] = 0;
    }

    ITYPE tile_ptr_array_size = 0;
    for (ITYPE i = 0; i < num_panels; i++ ) {
        tile_ptr_array_size += panels[i].Ti * panels[i].tiles.size();
        panel_Ti[i] = panels[i].Ti;
        panel_start[i] = panels[i].start_row;
        panel_ptr[i + 1] = panels[i].tiles.size();
        panel_offset[i + 1] = panels[i].tiles.size() * panels[i].Ti;
    }

    for ( ITYPE i = 1; i < ( num_panels + 1 ); i++ ) {
        panel_ptr[i] += panel_ptr[i - 1];
        panel_offset[i] += panel_offset[i - 1];
    }

    // Setup tile row ptrs
    this->tile_row_ptr = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE) * (tile_ptr_array_size + 1) );

    #pragma omp parallel for num_threads(8)
    for ( ITYPE i = 0; i < ( tile_ptr_array_size + 1); i++ ) { tile_row_ptr[i] = 0; }

    for ( ITYPE i = 0; i < nrows; i++ ) {
        auto row_start = row_ptr[i];
        auto row_end = row_ptr[i + 1];

        ITYPE tile_id = -1;
        ITYPE curr_tile = -1;
        ITYPE curr_row_grp = find_panel<T, ITYPE>( i, panels );

        auto panel_num_tiles = panel_ptr[curr_row_grp + 1] - panel_ptr[curr_row_grp];
        // auto panel_Ti = panel_Ti[curr_row_grp];

        for ( ITYPE ptr = row_start; ptr < row_end; ptr++ ) {
            ITYPE curr_col_grp = find_tile<T, ITYPE>( cols[ptr], panels[curr_row_grp] );

            if ( curr_tile == -1 || curr_col_grp != curr_tile ) {
                curr_tile = curr_col_grp;
            }
            ITYPE update_ptr = panel_offset[ curr_row_grp ] + ( (i - panel_start[curr_row_grp]) * panel_num_tiles ) + curr_tile;
            tile_row_ptr[ update_ptr + 1]++;
        }
    }

    tile_row_ptr[0] = 0;
    for ( ITYPE i = 1; i < (tile_ptr_array_size + 1); i++ ) {
        tile_row_ptr[i] += tile_row_ptr[i - 1];
    }

    #if (defined(RUN_ASPT_SPECIAL) || defined(ASPT_SPECIAL_SIMD_PARALLEL)) && defined(RUN_ASPT_SPECIAL_DENSE_OPT)
        mark_special_rows(panels);
    #endif

    /*
        print_arr<ITYPE>( row_ptr, nrows + 1, "row_ptr" );
        print_arr<ITYPE>( cols, nnzs, "cols: " );
        print_arr<ITYPE>( panel_ptr, num_panels + 1, "panel_ptr" );
        print_arr<ITYPE>( panel_start, num_panels, "panel_start" );
        print_arr<ITYPE>( panel_offset, num_panels, "panel_offset" );
        print_arr<ITYPE>( panel_Ti, num_panels, "panel_Ti" );
        print_arr<ITYPE>( tile_row_ptr, tile_ptr_array_size, "tile_row_ptr" );
    */
    delete[] temp;

    auto end_time = std::chrono::high_resolution_clock::now();

    // #if defined(RUN_ASPT_SPECIAL) || defined(ASPT_SPECIAL_SIMD_PARALLEL)
    //     mark_special_rows(panels);
    // #endif

    print_status("ATM Construction Time: %fs\n", std::chrono::duration<double>(end_time - start_time).count());
}


#endif // MATRICES_ATM_H
