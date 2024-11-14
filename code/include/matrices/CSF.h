#ifndef CSF_H
#define CSF_H

#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#define CSF_MAX_DIMENSIONS 4

// Adapted from the CSF-4 tensor generated by the TACO compiler

template <typename T, typename ITYPE>
class CSF
{
    public:
    ITYPE nrows;
    ITYPE ncols;
    ITYPE nnzs;

    ITYPE order;          // tensor order (number of modes)
    ITYPE *dimensions;    // tensor dimensions
    ITYPE csize;          // component size
    ITYPE *mode_ordering; // mode storage ordering
    ITYPE ***indices;    // tensor index data (per mode)
    T *vals;             // tensor values
    uint8_t *fill_value; // tensor fill value -- Not used
    ITYPE vals_size;     // values array size

    public:

    // Constructor for the CSF-4 representation
    CSF(ITYPE nrows, ITYPE ncols, ITYPE nnzs, ITYPE num_panels, ITYPE *A_COO1_crd, ITYPE *A_COO2_crd, ITYPE *A_COO3_crd, ITYPE *A_COO4_crd, T *A_COO_vals) : nrows(nrows), ncols(ncols), nnzs(nnzs), vals_size(nnzs)
    {
        this->order = 4;
        this->dimensions = (ITYPE *)malloc(sizeof(ITYPE) * this->order);
        this->indices = (ITYPE ***)malloc(this->order * sizeof(ITYPE **));
        this->dimensions[0] = nrows;
        this->dimensions[1] = ncols;

        for (ITYPE i = 0; i < this->order; i++) {
            this->indices[i] = (ITYPE **)malloc(2 * sizeof(ITYPE *));
        }

        auto *A = this;

        ITYPE *A1_pos = (ITYPE *)(A->indices[0][0]);
        ITYPE *A1_crd = (ITYPE *)(A->indices[0][1]);
        ITYPE *A2_pos = (ITYPE *)(A->indices[1][0]);
        ITYPE *A2_crd = (ITYPE *)(A->indices[1][1]);
        ITYPE *A3_pos = (ITYPE *)(A->indices[2][0]);
        ITYPE *A3_crd = (ITYPE *)(A->indices[2][1]);
        ITYPE *A4_pos = (ITYPE *)(A->indices[3][0]);
        ITYPE *A4_crd = (ITYPE *)(A->indices[3][1]);
        T *A_vals = (T *)(A->vals);

        A1_pos = (ITYPE *)malloc(sizeof(ITYPE) * 2);
        A1_pos[0] = 0;
        ITYPE A1_crd_size = 1048576;
        A1_crd = (ITYPE *)malloc(sizeof(ITYPE) * A1_crd_size);
        ITYPE ioA = 0;
        ITYPE A2_pos_size = 1048576;
        A2_pos = (ITYPE *)malloc(sizeof(ITYPE) * A2_pos_size);
        A2_pos[0] = 0;
        ITYPE A2_crd_size = 1048576;
        A2_crd = (ITYPE *)malloc(sizeof(ITYPE) * A2_crd_size);
        ITYPE joA = 0;
        ITYPE A3_pos_size = 1048576;
        A3_pos = (ITYPE *)malloc(sizeof(ITYPE) * A3_pos_size);
        A3_pos[0] = 0;
        ITYPE A3_crd_size = 1048576;
        A3_crd = (ITYPE *)malloc(sizeof(ITYPE) * A3_crd_size);
        ITYPE iiA = 0;
        ITYPE A4_pos_size = 1048576;
        A4_pos = (ITYPE *)malloc(sizeof(ITYPE) * A4_pos_size);
        A4_pos[0] = 0;
        ITYPE A4_crd_size = 1048576;
        A4_crd = (ITYPE *)malloc(sizeof(ITYPE) * A4_crd_size);
        ITYPE jiA = 0;
        ITYPE A_capacity = 1048576;
        A_vals = (T *)malloc(sizeof(T) * A_capacity);

        ITYPE ioA_COO = 0;
        ITYPE pA_COO1_end = nnzs;


        ITYPE row_panel = 0;
        while (ioA_COO < pA_COO1_end)
        {
            // std::cout << "Row panel: " << ioA_COO << std::endl;
            ITYPE io = A_COO1_crd[ioA_COO];
            ITYPE A_COO1_segend = ioA_COO + 1;
            while (A_COO1_segend < pA_COO1_end && A_COO1_crd[A_COO1_segend] == io)
            {
                A_COO1_segend++;
            }
            row_panel++;

            ITYPE pA2_begin = joA;
            if (A2_pos_size <= ioA + 1)
            {
                A2_pos = (ITYPE *)realloc(A2_pos, sizeof(ITYPE) * (A2_pos_size * 2));
                A2_pos_size *= 2;
            }

            ITYPE joA_COO = ioA_COO;

            while (joA_COO < A_COO1_segend)
            {
                ITYPE jo = A_COO2_crd[joA_COO];
                ITYPE A_COO2_segend = joA_COO + 1;
                while (A_COO2_segend < A_COO1_segend && A_COO2_crd[A_COO2_segend] == jo)
                {
                    A_COO2_segend++;
                }
                ITYPE pA3_begin = iiA;
                if (A3_pos_size <= joA + 1)
                {
                    A3_pos = (ITYPE *)realloc(A3_pos, sizeof(ITYPE) * (A3_pos_size * 2));
                    A3_pos_size *= 2;
                }

                ITYPE iiA_COO = joA_COO;

                while (iiA_COO < A_COO2_segend)
                {
                    ITYPE ii = A_COO3_crd[iiA_COO];
                    ITYPE A_COO3_segend = iiA_COO + 1;
                    while (A_COO3_segend < A_COO2_segend && A_COO3_crd[A_COO3_segend] == ii)
                    {
                        A_COO3_segend++;
                    }
                    ITYPE pA4_begin = jiA;
                    if (A4_pos_size <= iiA + 1)
                    {
                        A4_pos = (ITYPE *)realloc(A4_pos, sizeof(ITYPE) * (A4_pos_size * 2));
                        A4_pos_size *= 2;
                    }

                    ITYPE jiA_COO = iiA_COO;

                    while (jiA_COO < A_COO3_segend)
                    {
                        ITYPE ji = A_COO4_crd[jiA_COO];
                        T A_COO_val = A_COO_vals[jiA_COO];
                        jiA_COO++;
                        while (jiA_COO < A_COO3_segend && A_COO4_crd[jiA_COO] == ji)
                        {
                            A_COO_val += A_COO_vals[jiA_COO];
                            jiA_COO++;
                        }
                        if (A_capacity <= jiA)
                        {
                            A_vals = (T *)realloc(A_vals, sizeof(T) * (A_capacity * 2));
                            A_capacity *= 2;
                        }
                        A_vals[jiA] = A_COO_val;
                        if (A4_crd_size <= jiA)
                        {
                            A4_crd = (ITYPE *)realloc(A4_crd, sizeof(ITYPE) * (A4_crd_size * 2));
                            A4_crd_size *= 2;
                        }
                        A4_crd[jiA] = ji;
                        jiA++;
                    }

                    A4_pos[iiA + 1] = jiA;
                    if (pA4_begin < jiA)
                    {
                        if (A3_crd_size <= iiA)
                        {
                            A3_crd = (ITYPE *)realloc(A3_crd, sizeof(ITYPE) * (A3_crd_size * 2));
                            A3_crd_size *= 2;
                        }
                        A3_crd[iiA] = ii;
                        iiA++;
                    }
                    iiA_COO = A_COO3_segend;
                }

                A3_pos[joA + 1] = iiA;
                if (pA3_begin < iiA)
                {
                    if (A2_crd_size <= joA)
                    {
                        A2_crd = (ITYPE *)realloc(A2_crd, sizeof(ITYPE) * (A2_crd_size * 2));
                        A2_crd_size *= 2;
                    }
                    A2_crd[joA] = jo;
                    joA++;
                }
                joA_COO = A_COO2_segend;
            }

            A2_pos[ioA + 1] = joA;
            if (pA2_begin < joA)
            {
                if (A1_crd_size <= ioA)
                {
                    A1_crd = (ITYPE *)realloc(A1_crd, sizeof(ITYPE) * (A1_crd_size * 2));
                    A1_crd_size *= 2;
                }
                A1_crd[ioA] = io;
                ioA++;
            }
            ioA_COO = A_COO1_segend;
        }

        A1_pos[1] = ioA;

        A->indices[0][0] = (ITYPE *)(A1_pos);
        A->indices[0][1] = (ITYPE *)(A1_crd);
        A->indices[1][0] = (ITYPE *)(A2_pos);
        A->indices[1][1] = (ITYPE *)(A2_crd);
        A->indices[2][0] = (ITYPE *)(A3_pos);
        A->indices[2][1] = (ITYPE *)(A3_crd);
        A->indices[3][0] = (ITYPE *)(A4_pos);
        A->indices[3][1] = (ITYPE *)(A4_crd);
        A->vals = (T *)A_vals;
    }

}; // class CSF

template <typename T, typename ITYPE>
static bool compare_tiled_csf_coordinates(std::tuple<ITYPE, ITYPE, ITYPE, ITYPE, T> &a, std::tuple<ITYPE, ITYPE, ITYPE, ITYPE, T> &b)
{
    if (std::get<0>(a) < std::get<0>(b))
        return true;
    else if (std::get<1>(a) < std::get<1>(b))
        return true;
    else if (std::get<2>(a) < std::get<2>(b))
        return true;
    else if (std::get<3>(a) < std::get<3>(b))
        return true;
    else
        return false;
}

template <typename T, typename ITYPE>
ITYPE generate_coo_representation(ITYPE Ti, ITYPE Tj, ITYPE nnzs, std::pair<ITYPE, ITYPE> *locs, T *vals, ITYPE **A_coo1, ITYPE **A_coo2, ITYPE **A_coo3, ITYPE **A_coo4)
{
    std::set<ITYPE> panels;
    std::map<ITYPE, std::set<ITYPE>> panel_fibers;

    *A_coo1 = (ITYPE *)malloc(sizeof(ITYPE) * nnzs);
    *A_coo2 = (ITYPE *)malloc(sizeof(ITYPE) * nnzs);
    *A_coo3 = (ITYPE *)malloc(sizeof(ITYPE) * nnzs);
    *A_coo4 = (ITYPE *)malloc(sizeof(ITYPE) * nnzs);

    std::tuple<ITYPE, ITYPE, ITYPE, ITYPE, T> *tiled_coo = (std::tuple<ITYPE, ITYPE, ITYPE, ITYPE, T> *)malloc(sizeof(std::tuple<ITYPE, ITYPE, ITYPE, ITYPE, T>) * nnzs);
    for (ITYPE i = 0; i < nnzs; i++)
    {
        ITYPE row = locs[i].first;
        ITYPE col = locs[i].second;

        ITYPE tile_row = row / Ti;
        ITYPE tile_col = col / Tj;

        if (panels.find(tile_row) == panels.end()) {
            panels.insert(tile_row);
        }

        if (panel_fibers.find(tile_row) == panel_fibers.end()) {
            panel_fibers.insert({tile_row, std::set<ITYPE>()});
        }
        panel_fibers[tile_row].insert(tile_col);

        tiled_coo[i] = std::make_tuple(tile_row, tile_col, row, col, vals[i]);
    }

    // sort the tiled_coo representation
    std::sort(tiled_coo, tiled_coo + nnzs);

    // copy the sorted values to the COO representation
    for (ITYPE i = 0; i < nnzs; i++)
    {
        (*A_coo1)[i] = std::get<0>(tiled_coo[i]);
        (*A_coo2)[i] = std::get<1>(tiled_coo[i]);
        (*A_coo3)[i] = std::get<2>(tiled_coo[i]);
        (*A_coo4)[i] = std::get<3>(tiled_coo[i]);
        vals[i] = std::get<4>(tiled_coo[i]);

        // print the COO representation
        // std::cout << (*A_coo1)[i] << " " << (*A_coo2)[i] << " " << (*A_coo3)[i] << " " << (*A_coo4)[i] << " " << vals[i] << std::endl;
    }

    free(tiled_coo);

    ITYPE num_fibers = 0;
    for (auto &panel : panel_fibers) {
        num_fibers += panel.second.size();
    }

    std::cout << "Total fibers created: " << num_fibers << std::endl;

    return num_fibers;
}

template <typename T, typename ITYPE>
ITYPE generate_fixed_nnzs_coo_representation(ITYPE nnzs, std::pair<ITYPE, ITYPE> *locs, T *vals, ITYPE target_tile_nnzs, ITYPE **A_coo1, ITYPE **A_coo2, ITYPE **A_coo3, ITYPE **A_coo4)
{
    *A_coo1 = (ITYPE *)malloc(sizeof(ITYPE) * nnzs);
    *A_coo2 = (ITYPE *)malloc(sizeof(ITYPE) * nnzs);
    *A_coo3 = (ITYPE *)malloc(sizeof(ITYPE) * nnzs);
    *A_coo4 = (ITYPE *)malloc(sizeof(ITYPE) * nnzs);

    std::tuple<ITYPE, ITYPE, ITYPE, ITYPE, TYPE> *tiled_coo = (std::tuple<ITYPE, ITYPE, ITYPE, ITYPE, TYPE> *)malloc(sizeof(std::tuple<ITYPE, ITYPE, ITYPE, ITYPE, TYPE>) * nnzs);

    std::map<ITYPE, std::vector<std::tuple<ITYPE, ITYPE, TYPE>>> tile_map;  // map for fibers

    for (ITYPE i = 0; i < nnzs; i++)
    {
        ITYPE row = locs[i].first;
        ITYPE col = locs[i].second;

        tiled_coo[i] = std::make_tuple(0, 0, row, col, vals[i]);
    }
    std::sort(tiled_coo, tiled_coo + nnzs);
    // Figure out which tile each non-zero belongs so that all tiles have the same number of non-zeros

    ITYPE tile_id = 0;
    ITYPE tile_nnzs = 0;
    ITYPE curr_row = -1;

    for (ITYPE i = 0; i < nnzs; i++) {
        ITYPE row = std::get<2>(tiled_coo[i]);
        ITYPE col = std::get<3>(tiled_coo[i]);

        if (curr_row == -1) { curr_row = row; } // initialization condition

        if (curr_row != row) { // if curr row splits, then increment the tile_id
            ITYPE j;
            for (j = i; j < nnzs; j++) {
                ITYPE check_row = std::get<2>(tiled_coo[j]);
                if (check_row != row) {
                    break;
                }
            }

            if ( (j - i + tile_nnzs) > target_tile_nnzs) {
                tile_id++;
                tile_nnzs = 0;
            }
        }

        curr_row = row;

        if (tile_nnzs == target_tile_nnzs) {
            tile_id++;
            tile_nnzs = 0;
        }

        if (tile_map.find(tile_id) == tile_map.end()) {
            tile_map.insert({tile_id, std::vector<std::tuple<ITYPE, ITYPE, TYPE>>()});
        }
        tile_map[tile_id].push_back({row, col, std::get<4>(tiled_coo[i])});

        tile_nnzs++;
    }

    // ensure that split rows map to the same fiber group?
    std::vector< std::vector<std::vector<std::tuple<ITYPE, ITYPE, TYPE>>> > tile_groups;
    ITYPE curr_fiber_bundle = 0;
    for (auto &tile : tile_map) {
        if (tile_groups.size() == 0) {
            tile_groups.push_back(std::vector<std::vector<std::tuple<ITYPE, ITYPE, TYPE>>>());
        }

        if (tile_groups[curr_fiber_bundle].size() == 0) {
            tile_groups[curr_fiber_bundle].push_back(tile.second);
        } else {
            ITYPE last_row = std::get<0>(tile_groups[curr_fiber_bundle].back().back());
            ITYPE first_row = std::get<0>(tile.second.front());
            if (last_row == first_row) {
                tile_groups[curr_fiber_bundle].push_back(tile.second);
            } else {
                curr_fiber_bundle++;
                tile_groups.push_back(std::vector<std::vector<std::tuple<ITYPE, ITYPE, TYPE>>>());
                tile_groups[curr_fiber_bundle].push_back(tile.second);
            }
        }
    }

    // copy the sorted values to the COO representation
    ITYPE outer_idx = 0;
    ITYPE inner_idx = 0;
    ITYPE nnz = 0;
    for ( auto &tile_group : tile_groups ) {
        inner_idx = 0;
        for ( auto &tile : tile_group ) {
            for ( auto &fiber : tile ) {
                (*A_coo1)[nnz] = outer_idx;
                (*A_coo2)[nnz] = inner_idx;
                (*A_coo3)[nnz] = std::get<0>(fiber);
                (*A_coo4)[nnz] = std::get<1>(fiber);
                vals[nnz] = std::get<2>(fiber);
                nnz++;
            }
            inner_idx++;
        }
        outer_idx++;
    }

    free(tiled_coo);

    std::cout << "Total tiles created: " << tile_map.size() << std::endl;

    return tile_id;
}

template <typename T, typename ITYPE>
static ITYPE find_panel_csf(ITYPE row, std::vector<struct panel_t> &panels)
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

template <typename T, typename ITYPE>
static ITYPE find_tile_csf(ITYPE col, struct panel_t &panel)
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

// Generate rassm coo representation using the input from the tile builder
template <typename T, typename ITYPE>
ITYPE generate_coo_representation_rassm(ITYPE nnzs, std::pair<ITYPE, ITYPE> *locs, T *vals, std::vector<struct panel_t> &panels, ITYPE **A_coo1, ITYPE **A_coo2, ITYPE **A_coo3, ITYPE **A_coo4)
{
    // std::set<ITYPE> panels;
    std::map<ITYPE, std::set<ITYPE>> panel_fibers;

    *A_coo1 = (ITYPE *)malloc(sizeof(ITYPE) * nnzs);
    *A_coo2 = (ITYPE *)malloc(sizeof(ITYPE) * nnzs);
    *A_coo3 = (ITYPE *)malloc(sizeof(ITYPE) * nnzs);
    *A_coo4 = (ITYPE *)malloc(sizeof(ITYPE) * nnzs);

    std::tuple<ITYPE, ITYPE, ITYPE, ITYPE, T> *tiled_coo = (std::tuple<ITYPE, ITYPE, ITYPE, ITYPE, T> *)malloc(sizeof(std::tuple<ITYPE, ITYPE, ITYPE, ITYPE, T>) * nnzs);
    for (ITYPE i = 0; i < nnzs; i++)
    {
        ITYPE row = locs[i].first;
        ITYPE col = locs[i].second;

        ITYPE tile_row = find_panel_csf<T, ITYPE>(row, panels);
        ITYPE tile_col = find_tile_csf<T, ITYPE>(col, panels[tile_row]);

        if (panel_fibers.find(tile_row) == panel_fibers.end()) {
            panel_fibers.insert({tile_row, std::set<ITYPE>()});
        }
        panel_fibers[tile_row].insert(tile_col);

        tiled_coo[i] = std::make_tuple(tile_row, tile_col, row, col, vals[i]);
    }

    // sort the tiled_coo representation
    std::sort(tiled_coo, tiled_coo + nnzs);

    // copy the sorted values to the COO representation
    for (ITYPE i = 0; i < nnzs; i++)
    {
        (*A_coo1)[i] = std::get<0>(tiled_coo[i]);
        (*A_coo2)[i] = std::get<1>(tiled_coo[i]);
        (*A_coo3)[i] = std::get<2>(tiled_coo[i]);
        (*A_coo4)[i] = std::get<3>(tiled_coo[i]);
        vals[i] = std::get<4>(tiled_coo[i]);

        // print the COO representation
        // std::cout << (*A_coo1)[i] << " " << (*A_coo2)[i] << " " << (*A_coo3)[i] << " " << (*A_coo4)[i] << " " << vals[i] << std::endl;
    }

    free(tiled_coo);

    ITYPE num_fibers = 0;
    for (auto &panel : panel_fibers) {
        num_fibers += panel.second.size();
    }

    std::cout << "Total fibers created: " << num_fibers << std::endl;

    return num_fibers;
}

template <typename T, typename ITYPE>
void print_tile_histogram(CSF<T, ITYPE> &csf, ITYPE feature)
{
    ITYPE *A1_pos = (ITYPE *)(csf.indices[0][0]);
    ITYPE *A1_crd = (ITYPE *)(csf.indices[0][1]);
    ITYPE *A2_pos = (ITYPE *)(csf.indices[1][0]);
    ITYPE *A2_crd = (ITYPE *)(csf.indices[1][1]);
    ITYPE *A3_pos = (ITYPE *)(csf.indices[2][0]);
    ITYPE *A3_crd = (ITYPE *)(csf.indices[2][1]);
    ITYPE *A4_pos = (ITYPE *)(csf.indices[3][0]);
    ITYPE *A4_crd = (ITYPE *)(csf.indices[3][1]);
    T *A_vals = (T *)(csf.vals);

    std::map<ITYPE, ITYPE> tile_nnzs_histogram;
    std::map<ITYPE, ITYPE> tile_size_histogram;
    std::map<ITYPE, std::tuple<ITYPE, ITYPE, ITYPE, ITYPE>> tile_info_map;

    ITYPE tile_id = 0;
    for (ITYPE ioA = A1_pos[0]; ioA < A1_pos[1]; ioA++) { // fiber group
        ITYPE io = A1_crd[ioA]; // tile row
        for (ITYPE joA = A2_pos[ioA]; joA < A2_pos[(ioA + 1)]; joA++) { // fiber
            ITYPE jo = A2_crd[joA]; // tile column

            std::set<ITYPE> tile_nacs_set;
            std::set<ITYPE> tile_nars_set;
            ITYPE tile_nnzs = 0;
            for (ITYPE iiA = A3_pos[joA]; iiA < A3_pos[(joA + 1)]; iiA++) { // active rows of the tile
                ITYPE ii = A3_crd[iiA]; // row
                tile_nars_set.insert(ii);
                for (ITYPE jiA = A4_pos[iiA]; jiA < A4_pos[(iiA + 1)]; jiA++) { // active columns of the row
                    ITYPE ji = A4_crd[jiA]; // column
                    tile_nacs_set.insert(ji);
                    tile_nnzs++;
                }
            }
            ITYPE tile_size = (tile_nacs_set.size() + tile_nars_set.size()) * feature * sizeof(T);

            // populate tile size map
            if (tile_size_histogram.find(tile_size) == tile_size_histogram.end()) {
                tile_size_histogram.insert({tile_size, 0});
            }
            tile_size_histogram[tile_size]++;

            // populate tile non-zeroes map
            if (tile_nnzs_histogram.find(tile_nnzs) == tile_nnzs_histogram.end()) {
                tile_nnzs_histogram.insert({tile_nnzs, 0});
            }
            tile_nnzs_histogram[tile_nnzs]++;

            // populate tile info map
            tile_info_map.insert({tile_id++, {tile_nnzs, tile_nacs_set.size(), tile_nars_set.size(), tile_size}});
        }
    }

    std::cout << "BEGIN TILE NNZS HISTOGRAM" << std::endl;
    std::cout << "Tile NNZs, Count" << std::endl;
    for (auto &tile : tile_nnzs_histogram) {
        std::cout << tile.first << ", " << tile.second << std::endl;
    }
    std::cout << "END TILE NNZS HISTOGRAM" << std::endl;

    std::cout << "BEGIN TILE SIZE HISTOGRAM" << std::endl;
    std::cout << "Tile Size, Count" << std::endl;
    for (auto &tile : tile_size_histogram) {
        std::cout << tile.first << ", " << tile.second << std::endl;
    }
    std::cout << "END TILE SIZE HISTOGRAM" << std::endl;

    std::cout << "BEGIN TILE INFO MAP" << std::endl;
    std::cout << "Tile ID, Tile NNZs, Tile NACs, Tile NARs, Tile Size" << std::endl;
    for (auto &tile : tile_info_map) {
        std::cout << tile.first << ", " << std::get<0>(tile.second) << ", " << std::get<1>(tile.second) << ", " << std::get<2>(tile.second) << ", " << std::get<3>(tile.second) << std::endl;
    }
    std::cout << "END TILE INFO MAP" << std::endl;
}

#endif // CSF_H
