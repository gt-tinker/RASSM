
#include "common.h"
#include <cstdlib>
#include <cstring>
#include <fstream>

void panel_t::serialize(std::ofstream &ofs) {
    ofs.write((char*)&type, sizeof(type));
    ofs.write((char*)&num_residues, sizeof(num_residues));
    ofs.write((char*)&num_tiles, sizeof(num_tiles));
    ofs.write((char*)&start_row, sizeof(start_row));
    ofs.write((char*)&end_row, sizeof(end_row));
    ofs.write((char*)&Tk, sizeof(Tk));
    ofs.write((char*)&Ti, sizeof(Ti));
    ofs.write((char*)&Tj, sizeof(Tj));
    ofs.write((char*)&id, sizeof(id));
    ofs.write((char*)&nnzs, sizeof(nnzs));
    ofs.write((char*)&nacs, sizeof(nacs));
    ofs.write((char*)&rnacs, sizeof(rnacs));
    ofs.write((char*)&nars, sizeof(nars));
    ofs.write((char*)&rnars, sizeof(rnars));
    ofs.write((char*)&OI, sizeof(OI));
    ofs.write((char*)&nnzs_per_col_seg, sizeof(nnzs_per_col_seg));
    ofs.write((char*)&nnzs_per_row_seg, sizeof(nnzs_per_row_seg));
    ofs.write((char*)&score, sizeof(score));
    ofs.write((char*)&data_moved, sizeof(data_moved));
    ofs.write((char*)&flop_count, sizeof(flop_count));
    ofs.write((char*)&output_data_cached, sizeof(output_data_cached));
    ofs.write((char*)&tile_size, sizeof(tile_size));
    ofs.write((char*)&sparse_cached, sizeof(sparse_cached));
    ofs.write((char*)&is_dense, sizeof(is_dense));

    for (auto &tile : this->tiles) {
        ofs.write((char*)&tile.col_start, sizeof(tile.col_start));
        ofs.write((char*)&tile.col_end, sizeof(tile.col_end));
        ofs.write((char*)&tile.nnzs, sizeof(tile.nnzs));
        ofs.write((char*)&tile.nacs, sizeof(tile.nacs));
        ofs.write((char*)&tile.nars, sizeof(tile.nars));
        ofs.write((char*)&tile.input_volume, sizeof(tile.input_volume));
        ofs.write((char*)&tile.output_volume, sizeof(tile.output_volume));
    }

    // for (auto &var : var_Tj) {
    //     ofs.write((char*)&var.first, sizeof(var.first));
    //     ofs.write((char*)&var.second, sizeof(var.second));
    // }
}

void panel_t::deserialize(std::ifstream &ifs) {
    ifs.read((char*)&type, sizeof(type));
    ifs.read((char*)&num_residues, sizeof(num_residues));
    ifs.read((char*)&num_tiles, sizeof(num_tiles));
    ifs.read((char*)&start_row, sizeof(start_row));
    ifs.read((char*)&end_row, sizeof(end_row));
    ifs.read((char*)&Tk, sizeof(Tk));
    ifs.read((char*)&Ti, sizeof(Ti));
    ifs.read((char*)&Tj, sizeof(Tj));
    ifs.read((char*)&id, sizeof(id));
    ifs.read((char*)&nnzs, sizeof(nnzs));
    ifs.read((char*)&nacs, sizeof(nacs));
    ifs.read((char*)&rnacs, sizeof(rnacs));
    ifs.read((char*)&nars, sizeof(nars));
    ifs.read((char*)&rnars, sizeof(rnars));
    ifs.read((char*)&OI, sizeof(OI));
    ifs.read((char*)&nnzs_per_col_seg, sizeof(nnzs_per_col_seg));
    ifs.read((char*)&nnzs_per_row_seg, sizeof(nnzs_per_row_seg));
    ifs.read((char*)&score, sizeof(score));
    ifs.read((char*)&data_moved, sizeof(data_moved));
    ifs.read((char*)&flop_count, sizeof(flop_count));
    ifs.read((char*)&output_data_cached, sizeof(output_data_cached));
    ifs.read((char*)&tile_size, sizeof(tile_size));
    ifs.read((char*)&sparse_cached, sizeof(sparse_cached));
    ifs.read((char*)&is_dense, sizeof(is_dense));

    tiles.resize(num_tiles);
    for (auto &tile : tiles) {
        ifs.read((char*)&tile.col_start, sizeof(tile.col_start));
        ifs.read((char*)&tile.col_end, sizeof(tile.col_end));
        ifs.read((char*)&tile.nnzs, sizeof(tile.nnzs));
        ifs.read((char*)&tile.nacs, sizeof(tile.nacs));
        ifs.read((char*)&tile.nars, sizeof(tile.nars));
        ifs.read((char*)&tile.input_volume, sizeof(tile.input_volume));
        ifs.read((char*)&tile.output_volume, sizeof(tile.output_volume));
    }

    // var_Tj.resize(num_tiles);
    // for (auto &var : var_Tj) {
    //     ifs.read((char*)&var.first, sizeof(var.first));
    //     ifs.read((char*)&var.second, sizeof(var.second));
    // }
}

bool panel_t::compare(const panel_t &rhs) {
    if (type != rhs.type) return false;
    if (num_residues != rhs.num_residues) return false;
    if (num_tiles != rhs.num_tiles) return false;
    if (start_row != rhs.start_row) return false;
    if (end_row != rhs.end_row) return false;
    if (Tk != rhs.Tk) return false;
    if (Ti != rhs.Ti) return false;
    if (Tj != rhs.Tj) return false;
    if (id != rhs.id) return false;
    if (nnzs != rhs.nnzs) return false;
    if (nacs != rhs.nacs) return false;
    if (rnacs != rhs.rnacs) return false;
    if (nars != rhs.nars) return false;
    if (rnars != rhs.rnars) return false;
    if (OI != rhs.OI) return false;
    if (nnzs_per_col_seg != rhs.nnzs_per_col_seg) return false;
    if (nnzs_per_row_seg != rhs.nnzs_per_row_seg) return false;
    if (score != rhs.score) return false;
    if (data_moved != rhs.data_moved) return false;
    if (flop_count != rhs.flop_count) return false;
    if (output_data_cached != rhs.output_data_cached) return false;
    if (tile_size != rhs.tile_size) return false;
    if (sparse_cached != rhs.sparse_cached) return false;
    if (is_dense != rhs.is_dense) return false;

    for (size_t i = 0; i < num_tiles; i++) {
        if (tiles[i].col_start != rhs.tiles[i].col_start) return false;
        if (tiles[i].col_end != rhs.tiles[i].col_end) return false;
        if (tiles[i].nnzs != rhs.tiles[i].nnzs) return false;
        if (tiles[i].nacs != rhs.tiles[i].nacs) return false;
        if (tiles[i].nars != rhs.tiles[i].nars) return false;
        if (tiles[i].input_volume != rhs.tiles[i].input_volume) return false;
        if (tiles[i].output_volume != rhs.tiles[i].output_volume) return false;
    }

    return true;
}

void serialize_panel_vector(std::ofstream &ofs, std::vector<panel_t> &panels) {
    size_t size = panels.size();
    ofs.write((char*)&size, sizeof(size));
    for (auto &panel : panels) {
        panel.num_tiles = panel.tiles.size();
        std::cout << "Panel has: " << panel.num_tiles << " tiles" << std::endl;
        panel.serialize(ofs);
    }
}

void deserialize_panel_vector(std::ifstream &ifs, std::vector<panel_t> &panels) {
    size_t size;
    ifs.read((char*)&size, sizeof(size));
    panels.resize(size);
    for (auto &panel : panels) {
        panel.deserialize(ifs);
    }
}


