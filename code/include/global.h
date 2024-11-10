#ifndef GLOBAL_H
#define GLOBAL_H

#include "config.h"
#include "matrices/CSR.h"

#include <utility>

extern std::pair<ITYPE, ITYPE> *global_locs;
extern ITYPE global_nnzs, global_nrows, global_ncols;
extern TYPE *global_vals;
extern CSR<TYPE, ITYPE> *global_csr;

#endif // GLOBAL_H
