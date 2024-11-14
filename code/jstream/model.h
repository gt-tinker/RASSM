// This code has been adapted from the original code of the jstream paper from SC'20
// https://inria.hal.science/hal-03117491/file/main.pdf

#ifndef MODEL_H
#define MODEL_H

#include <cmath>
#include <math.h>
#include <cstdint>

extern uint64_t** intervalsp;
extern uint64_t** notsmallerp;
extern uint64_t** notsmallerweightedp;

typedef unsigned __int128 uint128_t;
int init_model(int nr, int nc, int ne, struct v_struct *gold_temp_v);
int pick_tile(int &Ti, int &Tk , int Nk , int cache, int nr, int nc, int ne);
int pick_tile_sddmm(int &Ti, int &Tk , int Nk , int cache, int nr, int nc, int ne);
uint64_t active_cols(int Ti, int nr, int nc, int ne);

#endif