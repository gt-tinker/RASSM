#ifndef COMBINATION_H
#define COMBINATION_H

#include "common.h"
#include "Residue.h"
#include "utils/util.h"

template <typename T, typename ITYPE>
ITYPE combine_generated_tiles(std::vector<panel_t> &adaptive_2d_tiles, ITYPE target_cache_size )
{
    ITYPE num_tiles = 0;
    ITYPE num_tiles_combined = 0;
    for ( auto &panel : adaptive_2d_tiles ) {
        num_tiles += panel.tiles.size();
        if ( panel.tiles.size() < 2 ) { // nothing to combine here tile is a row panel
            continue;
        }

        std::vector<struct tile_t> combined_tiles;

        for ( ITYPE i = 0; i < panel.tiles.size(); i++ ) {
            auto &tile = panel.tiles[i];
            // if ( (tile.input_volume + tile.output_volume) <= (target_cache_size/2) ) {
            if ( tile.nnzs < panel.Ti / 4 ) {
            // if ( tile.nacs < Ti / 4 ) {

                if ( combined_tiles.size() == 0 ) { // first tile
                    // check if we can merge with the next tile
                    if ( i == panel.tiles.size() - 1 ) {    // can't merge with the next tile, so push it
                        combined_tiles.push_back( tile );
                    } else {
                        struct tile_t &next_tile = panel.tiles[i+1];
                        double next_tile_active_density = ((double) next_tile.nnzs) / (((double) next_tile.nacs) * ((double) next_tile.nars));
                        double tile_active_density = ((double) tile.nnzs) / (((double) tile.nacs) * ((double) tile.nars));

                        next_tile.col_start = tile.col_start;
                        next_tile.nnzs += tile.nnzs;
                        next_tile.nacs += tile.nacs;
                    }

                } else { // check if we can merge
                    struct tile_t &last_tile = combined_tiles.back();
                    if ( i == panel.tiles.size() - 1 ) {    // can't merge with the next tile, so merge with previous
                        last_tile.col_end = tile.col_end;
                        last_tile.nnzs += tile.nnzs;
                        last_tile.nacs += tile.nacs;
                    } else {    // next tile exists, check if we can merge with it
                        struct tile_t &next_tile = panel.tiles[i+1];

                        double last_tile_active_density = ((double) last_tile.nnzs) / (((double) last_tile.nacs) * ((double) last_tile.nars));
                        double next_tile_active_density = ((double) next_tile.nnzs) / (((double) next_tile.nacs) * ((double) next_tile.nars));

                        if ( next_tile_active_density < last_tile_active_density ) {    // merge with the next tile
                            next_tile.col_start = tile.col_start;
                            next_tile.nnzs += tile.nnzs;
                            next_tile.nacs += tile.nacs;
                        } else {    // merge with the previous tile
                            last_tile.col_end = tile.col_end;
                            last_tile.nnzs += tile.nnzs;
                            last_tile.nacs += tile.nacs;
                        }
                    }
                }
            } else {
                combined_tiles.push_back( tile ); // just push it anyways, it's big enough
            }
        }
        num_tiles_combined += combined_tiles.size();

        panel.tiles.clear();
        for ( auto &tile : combined_tiles ) {
            panel.tiles.push_back( tile );
        }
        panel.num_tiles = panel.tiles.size();
    }

    print_status("Num tiles generates: %d\n", num_tiles);
    print_status("Num tiles after combination: %d\n", num_tiles_combined);

    return (num_tiles - num_tiles_combined);
}

template <typename T, typename ITYPE>
void sort_generated_tiles(std::vector<panel_t> &adaptive_2d_tiles)
{
    for ( auto &panel : adaptive_2d_tiles ) {
        std::sort( panel.tiles.begin(), panel.tiles.end(), [](const struct tile_t &a, const struct tile_t &b) {
            if (a.nars != b.nars ) return a.nars > b.nars;
            return a.nnzs > b.nnzs;
        });
    }
}

template <typename T, typename ITYPE>
void identify_dense_panels(std::vector<panel_t> &adaptive_2d_tiles, ITYPE target_cache_size)
{
    ITYPE dense_count = 0;
    for ( auto &panel : adaptive_2d_tiles ) {
        if (!panel.nars || !panel.nacs) { continue; }
        ITYPE nnzs_per_nar = panel.nnzs / panel.nars;

        ITYPE nnzs_per_nac = panel.nnzs / panel.nacs;

        if (nnzs_per_nar > panel.Ti) {
            panel.is_dense = true;
            dense_count++;
        }
    }

    print_status("Dense panels: %d\n", dense_count);
}

#endif // COMBINATION_H
