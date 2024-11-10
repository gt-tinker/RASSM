#ifndef CSF_H
#define CSF_H

#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

#define CSF_MAX_DIMENSIONS 4

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
    // taco_mode_t*    mode_types;    // mode storage types
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
        // this->dimensions[2] = 512;
        // this->dimensions[3] = 512;

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

    // sort the tiled_coo representation
    // std::sort(tiled_coo, tiled_coo + nnzs, [](std::tuple<ITYPE, ITYPE, ITYPE, ITYPE> &a, std::tuple<ITYPE, ITYPE, ITYPE, ITYPE> &b) {
    //     if (std::get<2>(a) < std::get<2>(b))
    //         return true;
    //     else if (std::get<3>(a) < std::get<3>(b))
    //         return true;
    //     else
    //         return false;
    // });

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

// // Generated by the Tensor Algebra Compiler (tensor-compiler.org)
// // taco "C(io,ii,k)=A(io,jo,ii,ji)*B(jo,ji,k)" -f=C:ddd:0,1,2 -f=A:ssss:0,1,2,3 -f=B:ddd:0,1,2 -write-source=taco_kernel.c -write-compute=taco_compute.c -write-assembly=taco_assembly.c
// #ifndef TACO_C_HEADERS
// #define TACO_C_HEADERS
// #include <stdio.h>
// #include <stdlib.h>
// #include <stdint.h>
// #include <stdbool.h>
// #include <math.h>
// #include <complex.h>
// #include <string.h>
// #if _OPENMP
// #include <omp.h>
// #endif
// #define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))
// #define TACO_MAX(_a,_b) ((_a) > (_b) ? (_a) : (_b))
// #define TACO_DEREF(_a) (((___context___*)(*__ctx__))->_a)
// #ifndef TACO_TENSOR_T_DEFINED
// #define TACO_TENSOR_T_DEFINED
// typedef enum { taco_mode_dense, taco_mode_sparse } taco_mode_t;
// typedef struct {
//   int32_t      order;         // tensor order (number of modes)
//   int32_t*     dimensions;    // tensor dimensions
//   int32_t      csize;         // component size
//   int32_t*     mode_ordering; // mode storage ordering
//   taco_mode_t* mode_types;    // mode storage types
//   uint8_t***   indices;       // tensor index data (per mode)
//   uint8_t*     vals;          // tensor values
//   uint8_t*     fill_value;    // tensor fill value
//   int32_t      vals_size;     // values array size
// } taco_tensor_t;
// #endif
// #if !_OPENMP
// int omp_get_thread_num() { return 0; }
// int omp_get_max_threads() { return 1; }
// #endif
// int cmp(const void *a, const void *b) {
//   return *((const int*)a) - *((const int*)b);
// }
// int taco_gallop(int *array, int arrayStart, int arrayEnd, int target) {
//   if (array[arrayStart] >= target || arrayStart >= arrayEnd) {
//     return arrayStart;
//   }
//   int step = 1;
//   int curr = arrayStart;
//   while (curr + step < arrayEnd && array[curr + step] < target) {
//     curr += step;
//     step = step * 2;
//   }

//   step = step / 2;
//   while (step > 0) {
//     if (curr + step < arrayEnd && array[curr + step] < target) {
//       curr += step;
//     }
//     step = step / 2;
//   }
//   return curr+1;
// }
// int taco_binarySearchAfter(int *array, int arrayStart, int arrayEnd, int target) {
//   if (array[arrayStart] >= target) {
//     return arrayStart;
//   }
//   int lowerBound = arrayStart; // always < target
//   int upperBound = arrayEnd; // always >= target
//   while (upperBound - lowerBound > 1) {
//     int mid = (upperBound + lowerBound) / 2;
//     int midValue = array[mid];
//     if (midValue < target) {
//       lowerBound = mid;
//     }
//     else if (midValue > target) {
//       upperBound = mid;
//     }
//     else {
//       return mid;
//     }
//   }
//   return upperBound;
// }
// int taco_binarySearchBefore(int *array, int arrayStart, int arrayEnd, int target) {
//   if (array[arrayEnd] <= target) {
//     return arrayEnd;
//   }
//   int lowerBound = arrayStart; // always <= target
//   int upperBound = arrayEnd; // always > target
//   while (upperBound - lowerBound > 1) {
//     int mid = (upperBound + lowerBound) / 2;
//     int midValue = array[mid];
//     if (midValue < target) {
//       lowerBound = mid;
//     }
//     else if (midValue > target) {
//       upperBound = mid;
//     }
//     else {
//       return mid;
//     }
//   }
//   return lowerBound;
// }
// taco_tensor_t* init_taco_tensor_t(int32_t order, int32_t csize,
//                                   int32_t* dimensions, int32_t* mode_ordering,
//                                   taco_mode_t* mode_types) {
//   taco_tensor_t* t = (taco_tensor_t *) malloc(sizeof(taco_tensor_t));
//   t->order         = order;
//   t->dimensions    = (int32_t *) malloc(order * sizeof(int32_t));
//   t->mode_ordering = (int32_t *) malloc(order * sizeof(int32_t));
//   t->mode_types    = (taco_mode_t *) malloc(order * sizeof(taco_mode_t));
//   t->indices       = (uint8_t ***) malloc(order * sizeof(uint8_t***));
//   t->csize         = csize;
//   for (int32_t i = 0; i < order; i++) {
//     t->dimensions[i]    = dimensions[i];
//     t->mode_ordering[i] = mode_ordering[i];
//     t->mode_types[i]    = mode_types[i];
//     switch (t->mode_types[i]) {
//       case taco_mode_dense:
//         t->indices[i] = (uint8_t **) malloc(1 * sizeof(uint8_t **));
//         break;
//       case taco_mode_sparse:
//         t->indices[i] = (uint8_t **) malloc(2 * sizeof(uint8_t **));
//         break;
//     }
//   }
//   return t;
// }
// void deinit_taco_tensor_t(taco_tensor_t* t) {
//   for (int i = 0; i < t->order; i++) {
//     free(t->indices[i]);
//   }
//   free(t->indices);
//   free(t->dimensions);
//   free(t->mode_ordering);
//   free(t->mode_types);
//   free(t);
// }
// #endif

// int compute(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B) {
//   int C1_dimension = (int)(C->dimensions[0]);
//   int C2_dimension = (int)(C->dimensions[1]);
//   int C3_dimension = (int)(C->dimensions[2]);
//   double*  C_vals = (double*)(C->vals);
//   int*  A1_pos = (int*)(A->indices[0][0]);
//   int*  A1_crd = (int*)(A->indices[0][1]);
//   int*  A2_pos = (int*)(A->indices[1][0]);
//   int*  A2_crd = (int*)(A->indices[1][1]);
//   int*  A3_pos = (int*)(A->indices[2][0]);
//   int*  A3_crd = (int*)(A->indices[2][1]);
//   int*  A4_pos = (int*)(A->indices[3][0]);
//   int*  A4_crd = (int*)(A->indices[3][1]);
//   double*  A_vals = (double*)(A->vals);
//   int B1_dimension = (int)(B->dimensions[0]);
//   int B2_dimension = (int)(B->dimensions[1]);
//   int B3_dimension = (int)(B->dimensions[2]);
//   double*  B_vals = (double*)(B->vals);

//   #pragma omp parallel for schedule(static)
//   for (int32_t pC = 0; pC < ((C1_dimension * C2_dimension) * C3_dimension); pC++) {
//     C_vals[pC] = 0.0;
//   }

//   #pragma omp parallel for schedule(runtime)
//   for (int32_t ioA = A1_pos[0]; ioA < A1_pos[1]; ioA++) {
//     int32_t io = A1_crd[ioA];
//     for (int32_t joA = A2_pos[ioA]; joA < A2_pos[(ioA + 1)]; joA++) {
//       int32_t jo = A2_crd[joA];
//       for (int32_t iiA = A3_pos[joA]; iiA < A3_pos[(joA + 1)]; iiA++) {
//         int32_t ii = A3_crd[iiA];
//         int32_t iiC = io * C2_dimension + ii;
//         for (int32_t jiA = A4_pos[iiA]; jiA < A4_pos[(iiA + 1)]; jiA++) {
//           int32_t ji = A4_crd[jiA];
//           int32_t jiB = jo * B2_dimension + ji;
//           for (int32_t k = 0; k < B3_dimension; k++) {
//             int32_t kC = iiC * C3_dimension + k;
//             int32_t kB = jiB * B3_dimension + k;
//             C_vals[kC] = C_vals[kC] + A_vals[jiA] * B_vals[kB];
//           }
//         }
//       }
//     }
//   }
//   return 0;
// }

// int assemble(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B) {
//   int C1_dimension = (int)(C->dimensions[0]);
//   int C2_dimension = (int)(C->dimensions[1]);
//   int C3_dimension = (int)(C->dimensions[2]);
//   double*  C_vals = (double*)(C->vals);

//   C_vals = (double*)malloc(sizeof(double) * ((C1_dimension * C2_dimension) * C3_dimension));

//   C->vals = (uint8_t*)C_vals;
//   return 0;
// }

// int evaluate(taco_tensor_t *C, taco_tensor_t *A, taco_tensor_t *B) {
//   int C1_dimension = (int)(C->dimensions[0]);
//   int C2_dimension = (int)(C->dimensions[1]);
//   int C3_dimension = (int)(C->dimensions[2]);
//   double*  C_vals = (double*)(C->vals);
//   int*  A1_pos = (int*)(A->indices[0][0]);
//   int*  A1_crd = (int*)(A->indices[0][1]);
//   int*  A2_pos = (int*)(A->indices[1][0]);
//   int*  A2_crd = (int*)(A->indices[1][1]);
//   int*  A3_pos = (int*)(A->indices[2][0]);
//   int*  A3_crd = (int*)(A->indices[2][1]);
//   int*  A4_pos = (int*)(A->indices[3][0]);
//   int*  A4_crd = (int*)(A->indices[3][1]);
//   double*  A_vals = (double*)(A->vals);
//   int B1_dimension = (int)(B->dimensions[0]);
//   int B2_dimension = (int)(B->dimensions[1]);
//   int B3_dimension = (int)(B->dimensions[2]);
//   double*  B_vals = (double*)(B->vals);

//   int32_t C_capacity = (C1_dimension * C2_dimension) * C3_dimension;
//   C_vals = (double*)malloc(sizeof(double) * C_capacity);

//   #pragma omp parallel for schedule(static)
//   for (int32_t pC = 0; pC < C_capacity; pC++) {
//     C_vals[pC] = 0.0;
//   }

//   #pragma omp parallel for schedule(runtime)
//   for (int32_t ioA = A1_pos[0]; ioA < A1_pos[1]; ioA++) {
//     int32_t io = A1_crd[ioA];
//     for (int32_t joA = A2_pos[ioA]; joA < A2_pos[(ioA + 1)]; joA++) {
//       int32_t jo = A2_crd[joA];
//       for (int32_t iiA = A3_pos[joA]; iiA < A3_pos[(joA + 1)]; iiA++) {
//         int32_t ii = A3_crd[iiA];
//         int32_t iiC = io * C2_dimension + ii;
//         for (int32_t jiA = A4_pos[iiA]; jiA < A4_pos[(iiA + 1)]; jiA++) {
//           int32_t ji = A4_crd[jiA];
//           int32_t jiB = jo * B2_dimension + ji;
//           for (int32_t k = 0; k < B3_dimension; k++) {
//             int32_t kC = iiC * C3_dimension + k;
//             int32_t kB = jiB * B3_dimension + k;
//             C_vals[kC] = C_vals[kC] + A_vals[jiA] * B_vals[kB];
//           }
//         }
//       }
//     }
//   }

//   C->vals = (uint8_t*)C_vals;
//   return 0;
// }

// /*
//  * The `pack` functions convert coordinate and value arrays in COO format,
//  * with nonzeros sorted lexicographically by their coordinates, to the
//  * specified input format.
//  *
//  * The `unpack` function converts the specified output format to coordinate
//  * and value arrays in COO format.
//  *
//  * For both, the `_COO_pos` arrays contain two elements, where the first is 0
//  * and the second is the number of nonzeros in the tensor.
//  */

// int pack_A(taco_tensor_t *A, int* A_COO1_pos, int* A_COO1_crd, int* A_COO2_crd, int* A_COO3_crd, int* A_COO4_crd, double* A_COO_vals) {
//   int* restrict A1_pos = (int*)(A->indices[0][0]);
//   int* restrict A1_crd = (int*)(A->indices[0][1]);
//   int* restrict A2_pos = (int*)(A->indices[1][0]);
//   int* restrict A2_crd = (int*)(A->indices[1][1]);
//   int* restrict A3_pos = (int*)(A->indices[2][0]);
//   int* restrict A3_crd = (int*)(A->indices[2][1]);
//   int* restrict A4_pos = (int*)(A->indices[3][0]);
//   int* restrict A4_crd = (int*)(A->indices[3][1]);
//   double* restrict A_vals = (double*)(A->vals);

//   A1_pos = (int32_t*)malloc(sizeof(int32_t) * 2);
//   A1_pos[0] = 0;
//   int32_t A1_crd_size = 1048576;
//   A1_crd = (int32_t*)malloc(sizeof(int32_t) * A1_crd_size);
//   int32_t ioA = 0;
//   int32_t A2_pos_size = 1048576;
//   A2_pos = (int32_t*)malloc(sizeof(int32_t) * A2_pos_size);
//   A2_pos[0] = 0;
//   int32_t A2_crd_size = 1048576;
//   A2_crd = (int32_t*)malloc(sizeof(int32_t) * A2_crd_size);
//   int32_t joA = 0;
//   int32_t A3_pos_size = 1048576;
//   A3_pos = (int32_t*)malloc(sizeof(int32_t) * A3_pos_size);
//   A3_pos[0] = 0;
//   int32_t A3_crd_size = 1048576;
//   A3_crd = (int32_t*)malloc(sizeof(int32_t) * A3_crd_size);
//   int32_t iiA = 0;
//   int32_t A4_pos_size = 1048576;
//   A4_pos = (int32_t*)malloc(sizeof(int32_t) * A4_pos_size);
//   A4_pos[0] = 0;
//   int32_t A4_crd_size = 1048576;
//   A4_crd = (int32_t*)malloc(sizeof(int32_t) * A4_crd_size);
//   int32_t jiA = 0;
//   int32_t A_capacity = 1048576;
//   A_vals = (double*)malloc(sizeof(double) * A_capacity);

//   int32_t ioA_COO = A_COO1_pos[0];
//   int32_t pA_COO1_end = A_COO1_pos[1];

//   while (ioA_COO < pA_COO1_end) {
//     int32_t io = A_COO1_crd[ioA_COO];
//     int32_t A_COO1_segend = ioA_COO + 1;
//     while (A_COO1_segend < pA_COO1_end && A_COO1_crd[A_COO1_segend] == io) {
//       A_COO1_segend++;
//     }
//     int32_t pA2_begin = joA;
//     if (A2_pos_size <= ioA + 1) {
//       A2_pos = (int32_t*)realloc(A2_pos, sizeof(int32_t) * (A2_pos_size * 2));
//       A2_pos_size *= 2;
//     }

//     int32_t joA_COO = ioA_COO;

//     while (joA_COO < A_COO1_segend) {
//       int32_t jo = A_COO2_crd[joA_COO];
//       int32_t A_COO2_segend = joA_COO + 1;
//       while (A_COO2_segend < A_COO1_segend && A_COO2_crd[A_COO2_segend] == jo) {
//         A_COO2_segend++;
//       }
//       int32_t pA3_begin = iiA;
//       if (A3_pos_size <= joA + 1) {
//         A3_pos = (int32_t*)realloc(A3_pos, sizeof(int32_t) * (A3_pos_size * 2));
//         A3_pos_size *= 2;
//       }

//       int32_t iiA_COO = joA_COO;

//       while (iiA_COO < A_COO2_segend) {
//         int32_t ii = A_COO3_crd[iiA_COO];
//         int32_t A_COO3_segend = iiA_COO + 1;
//         while (A_COO3_segend < A_COO2_segend && A_COO3_crd[A_COO3_segend] == ii) {
//           A_COO3_segend++;
//         }
//         int32_t pA4_begin = jiA;
//         if (A4_pos_size <= iiA + 1) {
//           A4_pos = (int32_t*)realloc(A4_pos, sizeof(int32_t) * (A4_pos_size * 2));
//           A4_pos_size *= 2;
//         }

//         int32_t jiA_COO = iiA_COO;

//         while (jiA_COO < A_COO3_segend) {
//           int32_t ji = A_COO4_crd[jiA_COO];
//           double A_COO_val = A_COO_vals[jiA_COO];
//           jiA_COO++;
//           while (jiA_COO < A_COO3_segend && A_COO4_crd[jiA_COO] == ji) {
//             A_COO_val += A_COO_vals[jiA_COO];
//             jiA_COO++;
//           }
//           if (A_capacity <= jiA) {
//             A_vals = (double*)realloc(A_vals, sizeof(double) * (A_capacity * 2));
//             A_capacity *= 2;
//           }
//           A_vals[jiA] = A_COO_val;
//           if (A4_crd_size <= jiA) {
//             A4_crd = (int32_t*)realloc(A4_crd, sizeof(int32_t) * (A4_crd_size * 2));
//             A4_crd_size *= 2;
//           }
//           A4_crd[jiA] = ji;
//           jiA++;
//         }

//         A4_pos[iiA + 1] = jiA;
//         if (pA4_begin < jiA) {
//           if (A3_crd_size <= iiA) {
//             A3_crd = (int32_t*)realloc(A3_crd, sizeof(int32_t) * (A3_crd_size * 2));
//             A3_crd_size *= 2;
//           }
//           A3_crd[iiA] = ii;
//           iiA++;
//         }
//         iiA_COO = A_COO3_segend;
//       }

//       A3_pos[joA + 1] = iiA;
//       if (pA3_begin < iiA) {
//         if (A2_crd_size <= joA) {
//           A2_crd = (int32_t*)realloc(A2_crd, sizeof(int32_t) * (A2_crd_size * 2));
//           A2_crd_size *= 2;
//         }
//         A2_crd[joA] = jo;
//         joA++;
//       }
//       joA_COO = A_COO2_segend;
//     }

//     A2_pos[ioA + 1] = joA;
//     if (pA2_begin < joA) {
//       if (A1_crd_size <= ioA) {
//         A1_crd = (int32_t*)realloc(A1_crd, sizeof(int32_t) * (A1_crd_size * 2));
//         A1_crd_size *= 2;
//       }
//       A1_crd[ioA] = io;
//       ioA++;
//     }
//     ioA_COO = A_COO1_segend;
//   }

//   A1_pos[1] = ioA;

//   A->indices[0][0] = (uint8_t*)(A1_pos);
//   A->indices[0][1] = (uint8_t*)(A1_crd);
//   A->indices[1][0] = (uint8_t*)(A2_pos);
//   A->indices[1][1] = (uint8_t*)(A2_crd);
//   A->indices[2][0] = (uint8_t*)(A3_pos);
//   A->indices[2][1] = (uint8_t*)(A3_crd);
//   A->indices[3][0] = (uint8_t*)(A4_pos);
//   A->indices[3][1] = (uint8_t*)(A4_crd);
//   A->vals = (uint8_t*)A_vals;
//   return 0;
// }

// int pack_B(taco_tensor_t *B, int* B_COO1_pos, int* B_COO1_crd, int* B_COO2_crd, int* B_COO3_crd, double* B_COO_vals) {
//   int B1_dimension = (int)(B->dimensions[0]);
//   int B2_dimension = (int)(B->dimensions[1]);
//   int B3_dimension = (int)(B->dimensions[2]);
//   double* restrict B_vals = (double*)(B->vals);

//   int32_t B_capacity = (B1_dimension * B2_dimension) * B3_dimension;
//   B_vals = (double*)malloc(sizeof(double) * B_capacity);

//   #pragma omp parallel for schedule(static)
//   for (int32_t pB = 0; pB < B_capacity; pB++) {
//     B_vals[pB] = 0.0;
//   }

//   int32_t joB_COO = B_COO1_pos[0];
//   int32_t pB_COO1_end = B_COO1_pos[1];

//   while (joB_COO < pB_COO1_end) {
//     int32_t jo = B_COO1_crd[joB_COO];
//     int32_t B_COO1_segend = joB_COO + 1;
//     while (B_COO1_segend < pB_COO1_end && B_COO1_crd[B_COO1_segend] == jo) {
//       B_COO1_segend++;
//     }
//     int32_t jiB_COO = joB_COO;

//     while (jiB_COO < B_COO1_segend) {
//       int32_t ji = B_COO2_crd[jiB_COO];
//       int32_t B_COO2_segend = jiB_COO + 1;
//       while (B_COO2_segend < B_COO1_segend && B_COO2_crd[B_COO2_segend] == ji) {
//         B_COO2_segend++;
//       }
//       int32_t jiB = jo * B2_dimension + ji;
//       int32_t kB_COO = jiB_COO;

//       while (kB_COO < B_COO2_segend) {
//         int32_t k = B_COO3_crd[kB_COO];
//         double B_COO_val = B_COO_vals[kB_COO];
//         kB_COO++;
//         while (kB_COO < B_COO2_segend && B_COO3_crd[kB_COO] == k) {
//           B_COO_val += B_COO_vals[kB_COO];
//           kB_COO++;
//         }
//         int32_t kB = jiB * B3_dimension + k;
//         B_vals[kB] = B_COO_val;
//       }
//       jiB_COO = B_COO2_segend;
//     }
//     joB_COO = B_COO1_segend;
//   }

//   B->vals = (uint8_t*)B_vals;
//   return 0;
// }

// int unpack(int** C_COO1_pos_ptr, int** C_COO1_crd_ptr, int** C_COO2_crd_ptr, int** C_COO3_crd_ptr, double** C_COO_vals_ptr, taco_tensor_t *C) {
//   int* C_COO1_pos;
//   int* C_COO1_crd;
//   int* C_COO2_crd;
//   int* C_COO3_crd;
//   double* C_COO_vals;
//   int C1_dimension = (int)(C->dimensions[0]);
//   int C2_dimension = (int)(C->dimensions[1]);
//   int C3_dimension = (int)(C->dimensions[2]);
//   double* restrict C_vals = (double*)(C->vals);

//   C_COO1_pos = (int32_t*)malloc(sizeof(int32_t) * 2);
//   C_COO1_pos[0] = 0;
//   int32_t C_COO1_crd_size = 1048576;
//   C_COO1_crd = (int32_t*)malloc(sizeof(int32_t) * C_COO1_crd_size);
//   int32_t C_COO2_crd_size = 1048576;
//   C_COO2_crd = (int32_t*)malloc(sizeof(int32_t) * C_COO2_crd_size);
//   int32_t C_COO3_crd_size = 1048576;
//   C_COO3_crd = (int32_t*)malloc(sizeof(int32_t) * C_COO3_crd_size);
//   int32_t kC_COO = 0;
//   int32_t C_COO_capacity = 1048576;
//   C_COO_vals = (double*)malloc(sizeof(double) * C_COO_capacity);

//   for (int32_t io = 0; io < C1_dimension; io++) {
//     for (int32_t ii = 0; ii < C2_dimension; ii++) {
//       int32_t iiC = io * C2_dimension + ii;
//       for (int32_t k = 0; k < C3_dimension; k++) {
//         if (C_COO_capacity <= kC_COO) {
//           C_COO_vals = (double*)realloc(C_COO_vals, sizeof(double) * (C_COO_capacity * 2));
//           C_COO_capacity *= 2;
//         }
//         int32_t kC = iiC * C3_dimension + k;
//         C_COO_vals[kC_COO] = C_vals[kC];
//         if (C_COO3_crd_size <= kC_COO) {
//           int32_t C_COO3_crd_new_size = TACO_MAX(C_COO3_crd_size * 2,(kC_COO + 1));
//           C_COO3_crd = (int32_t*)realloc(C_COO3_crd, sizeof(int32_t) * C_COO3_crd_new_size);
//           C_COO3_crd_size = C_COO3_crd_new_size;
//         }
//         C_COO3_crd[kC_COO] = k;
//         if (C_COO2_crd_size <= kC_COO) {
//           int32_t C_COO2_crd_new_size = TACO_MAX(C_COO2_crd_size * 2,(kC_COO + 1));
//           C_COO2_crd = (int32_t*)realloc(C_COO2_crd, sizeof(int32_t) * C_COO2_crd_new_size);
//           C_COO2_crd_size = C_COO2_crd_new_size;
//         }
//         C_COO2_crd[kC_COO] = ii;
//         if (C_COO1_crd_size <= kC_COO) {
//           C_COO1_crd = (int32_t*)realloc(C_COO1_crd, sizeof(int32_t) * (C_COO1_crd_size * 2));
//           C_COO1_crd_size *= 2;
//         }
//         C_COO1_crd[kC_COO] = io;
//         kC_COO++;
//       }
//     }
//   }

//   C_COO1_pos[1] = kC_COO;

//   *C_COO1_pos_ptr = C_COO1_pos;
//   *C_COO1_crd_ptr = C_COO1_crd;
//   *C_COO2_crd_ptr = C_COO2_crd;
//   *C_COO3_crd_ptr = C_COO3_crd;
//   *C_COO_vals_ptr = C_COO_vals;
//   return 0;
// }

#endif // CSF_H
