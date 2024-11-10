#ifndef COMMON_H
#define COMMON_H

#include <algorithm>
#include <fstream>
#include <unordered_map>
#include <vector>

#include "BitSet.h"

template <typename ITYPE>
struct OccupancyInfo {
    ITYPE nnzs;
};

template<typename ITYPE>
struct RangeInfo {
    std::unordered_map<ITYPE, std::pair<ITYPE, ITYPE>> range;
    std::vector<std::pair<ITYPE, ITYPE>> interval_points;
    ITYPE overlapping_count;

    void reset()
    {
        this->range.clear();
    }

    // for adding a new range record
    void addRange(ITYPE key, ITYPE loc)
    {
        auto this_it = this->range.find( key );
        if ( this_it == this->range.end() ) {
            this->range.insert( {key, {loc, loc}} );
        } else {
            if ( loc > this_it->second.second ) {
                this_it->second.second = loc;
            }
        }
    }

    ITYPE count()
    {
        ITYPE max_overlapping_intervals = 0;
        // std::vector<std::pair<ITYPE, bool>> interval_points;
        interval_points.reserve( this->range.size() * 2 );

        for ( auto this_it : this->range ) {
            interval_points.push_back( {this_it.second.first, false} );
            interval_points.push_back( {this_it.second.second, true} );
        }
        std::sort( interval_points.begin(), interval_points.end(), [](auto &left, auto &right) {
            if (left.first != right.first) { return left.first < right.first; }
            else { return left.second < right.second; }
        });
        ITYPE active_intervals = 0;
        for (auto it : interval_points) {
            if (!it.second) {
                active_intervals++;
            } else {
                active_intervals--;
            }
            max_overlapping_intervals = MAX(max_overlapping_intervals, active_intervals);
        }

        // TODO: check if this is really needed
        if (max_overlapping_intervals == 0 && this->range.size() > 0) {
            max_overlapping_intervals++;
        }
        this->overlapping_count = max_overlapping_intervals;
        interval_points.clear();
        return max_overlapping_intervals;
    }

    // for combining range info of active columns
    RangeInfo& operator |= (RangeInfo &rhs)
    {
        for ( auto &rhs_it : rhs.range ) {
            auto this_it = this->range.find(rhs_it.first);
            if ( this_it == this->range.end() ) {
                this->range.insert( {rhs_it.first, {rhs_it.second.first, rhs_it.second.second}} ); // not sure if this works, but we will try
            } else {
                if ( rhs_it.second.second > this_it->second.second ) {
                    this_it->second.second = rhs_it.second.second;
                }
            }
        }

        return *this;
    }

    RangeInfo& operator &= (RangeInfo &rhs)
    {
        for ( auto &rhs_it : rhs.range ) {
            auto this_it = this->range.find(rhs_it.first);
            if ( this_it != this->range.end() ) {
                if ( rhs_it.second.second > this_it->second.second ) {
                    this_it->second.second = rhs_it.second.second;
                }
            }
        }
        for ( auto it = this->range.begin(); it != this->range.end(); ) {
            auto rhs_it = rhs.find(it->first);
            if ( rhs_it == rhs.end() ) {
                it = this->range.erase(it);
            } else {
                it++;
            }
        }
    }

    // write an assign method overload
    RangeInfo& operator = (RangeInfo &rhs)
    {
        this->range = rhs.range;
        return *this;
    }
};


template <typename T, typename ITYPE>
struct BitSetCounter {
    std::vector<ITYPE> buckets;
    ITYPE num_bits;

    BitSetCounter(ITYPE num_bits) : num_bits(num_bits) {
        buckets = std::vector<ITYPE>(num_bits);
    }

    // init method that takes num_bits as argument
    void init(ITYPE num_bits) {
        this->num_bits = num_bits;
        buckets = std::vector<ITYPE>(num_bits);
    }

    void insert( BitSet<ITYPE> &bitset ) {
        for ( ITYPE bit = 0; bit < bitset.num_entries; bit++ ) {
            this->buckets[ bit % num_bits ] += bitset.get(bit);
        }
    }

    // for tracking conflicts from range based information, add the column id to the bucket
    // TODO: check if this is accurage, might need to change the mod factor
    void insert( RangeInfo<ITYPE> &range_info ) {
        for ( auto &range : range_info.range ) {
            this->buckets[ range.first % num_bits ]++;
        }
    }

    ITYPE max_count() {
        ITYPE max = buckets[0];
        for ( ITYPE bit = 0; bit < num_bits; bit++ ) {
            if (buckets[bit] > max) {
                max = buckets[bit];
            }
        }
        return max;
    }

    ITYPE count_more_than( ITYPE threshold ) {
        ITYPE count = 0;
        for ( ITYPE bit = 0; bit < num_bits; bit++ ) {
            if (buckets[bit] > threshold) {
                count++;
            }
        }
        return count;
    }
};

struct workitem {
    // int start_row;
    // int end_row;

    ITYPE start_row;
    ITYPE end_row;
    ITYPE Tk;
    ITYPE panel_id;

    ITYPE nnzs;
    ITYPE nacs;
    ITYPE nars;

    bool type;      // True -- implies CSR traversal, False -- implies CSC traversal
    double score;


    ITYPE num_partitions;

    ITYPE rnacs;
    ITYPE rnars;
    // workitem() {  }

    // workitem(const struct workitem &A) {
    //     start_row = A.start_row;
    //     end_row = A.end_row;
    //     Tk = A.Tk;
    //     panel_id = A.panel_id;
    // }
};

enum panel_type_t {
    P_CSR = 0,
    P_CSC,
    P_SPLIT,

    P_2D,
};

enum score_type_t {
    SCORE_OI = 0,
    SCORE_TILE_SIZE,
    SCORE_NNZ_COMP,
    SCORE_CRIT_LOAD,
    SCORE_PERCEPTRON_OI,
};

struct tile_t
{
    ITYPE col_start = 0;
    ITYPE col_end = 0;
    ITYPE nnzs = 0;
    ITYPE nacs = 0;
    ITYPE nars = 0;

    ITYPE input_volume;
    ITYPE output_volume;
};

// template <typename ITYPE>
struct panel_t
{
    enum panel_type_t type;
    ITYPE num_residues = 0;
    ITYPE num_tiles = 0;    // Number of tiles in the panel
    ITYPE start_row = 0;
    ITYPE end_row = 0;

    ITYPE Tk = 0;
    ITYPE Ti = 0;
    ITYPE Tj = 0;   // Used when 2D tiling is used
    ITYPE id = 0;

    std::vector<std::pair<ITYPE, ITYPE>> var_Tj; // variable Tj values {start, end of each tile}
    std::vector<tile_t> tiles;  // tiles in the panel -- used for 2D tiling

    ITYPE nnzs = 0;
    ITYPE nacs = 0;
    ITYPE rnacs = 0;
    ITYPE nars = 0;
    ITYPE rnars = 0;

    double OI = 0;
    double nnzs_per_col_seg = 0;
    double nnzs_per_row_seg = 0;
    double score = 0;

    size_t data_moved = 0;
    size_t flop_count = 0;
    size_t output_data_cached = 0;

    ITYPE tile_size = 0;    // tile size in bytes
    bool sparse_cached = false;
    bool is_dense = false;

    // #ifdef CACHE_CONFLICT_ANALYSIS
    // BitSetCounter<TYPE, ITYPE> cache_conflict;
    // #endif
    void serialize(std::ofstream &ofs);
    void deserialize(std::ifstream &ifs);
    bool compare(const panel_t &rhs);
};

void serialize_panel_vector(std::ofstream &ofs, std::vector<panel_t> &panels);
void deserialize_panel_vector(std::ifstream &ifs, std::vector<panel_t> &panels);

#endif // COMMON_H
