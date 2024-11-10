
#include "config.h"
#include "global.h"

#include <utility>

double alpha_val = 5.0;
double beta_val = 14.0;
bool global_debug = false;

ITYPE global_nnzs, global_nrows, global_ncols;
std::pair<ITYPE, ITYPE> *global_locs = nullptr;
TYPE *global_vals = nullptr;

ITYPE CACHE_NUM_WAYS;
