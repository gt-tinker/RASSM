// This code has been adapted from the Adaptive Sparse Tiling Paper (ASpT) from PPoPP'19
// No modifications have been made to the primary logic or matrix reordering. Only the boilerplate code to match rassm's implementation has been tweaked.
// https://dl.acm.org/doi/pdf/10.1145/3293883.3295712


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
// #include <xmmintrin.h>

#include <immintrin.h>
#include "mkl.h"
#include <time.h>
#include <omp.h>
#include <sys/time.h>
#include <string.h>
#include<math.h>
#include<iostream>
#include <fstream>

#include <cstdlib>
#include <chrono>
#include <map>
#include <set>

#include "utils/Statistics.h"

using namespace std;

using ITYPE = int32_t;

using T = double;

double time_in_mill_now();
double time_in_mill_now() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  double time_in_mill =
    (tv.tv_sec) * 1000.0 + (tv.tv_usec) / 1000.0;
  return time_in_mill;
}

#define ERR fprintf(stderr, "ERR\n");

#define CLOCK_FREQUENCY ((double) (2.6E9))
#define ALLOC_ALIGNMENT 64

// #define OMP_SCHEDULE static
#define OMP_SCHEDULE dynamic

// #define MIN(a,b) (((a)<(b))?(a):(b))
// #define MAX(a,b) (((a)>(b))?(a):(b))
// #define CEIL(a,b) (((a)+(b)-1)/(b))
#define FTYPE double

#define MFACTOR (32)
#define LOG_MFACTOR (5)
#define BSIZE (1024/1)
#define BF (BSIZE/32)
#define INIT_GRP (10000000)
#define INIT_LIST (-1)


// The threshold should control the threshold of nnz's in a dense column
// lets play with parameter and see where we get
#define THRESHOLD (16*1)
// #define THRESHOLD (12 * 1)


// #define BH (128*1)
// #define LOG_BH (7)
// #define BW (128*1)


// #define MIN_OCC (128*3/4)
// #define MIN_OCC (BW*3/4)
// #define MIN_OCC (BW*1/2)
// #define MIN_OCC (-1)


#define SBSIZE (128)
#define SBF (SBSIZE / 32)
#define DBSIZE (1024)
#define DBF (DBSIZE / 32)
#define SPBSIZE (256)
#define SPBF (SPBSIZE / 32)
#define STHRESHOLD (1024/2*1)
#define SSTRIDE (STHRESHOLD / SPBF)
// #define NTHREAD (20)

#define MAX_NTHREAD 128
#define MAX_FEATURE 1024

// #define WARMUP_DIVIDER 10

// #define LARGE_ALLOC

/*

// #define RUN_PANEL_STATS
#ifdef RUN_PANEL_STATS
    #define NTHREAD (1)
#else
    #define NTHREAD (4)
#endif

*/

// #define SC_SIZE (2048)
#define SC_SIZE (32768)

//#define SIM_VALUE

// struct v_struct {
// 	ITYPE row, col;
// 	FTYPE val;
// 	ITYPE grp;
// };

double vari, avg;
double avg0[MAX_NTHREAD];
struct v_struct *temp_v, *gold_temp_v;
ITYPE sc, nr, nc, ne, gold_ne, npanel, mne, mne_nr;
ITYPE nr0;

// Added for the respmm paper
ITYPE layers = 1;	// number of layers of the input and output
ITYPE ITER = 100;
ITYPE Tk;

bool dont_run_special = false;

ITYPE BH, BW, LOG_BH, MIN_OCC;
char *in_file;

ITYPE *csr_v;
ITYPE *csr_e, *csr_e0;
FTYPE *csr_ev, *csr_ev0;
//int *mcsr_v;
ITYPE *mcsr_e; // can be short type
ITYPE *mcsr_cnt;
ITYPE *mcsr_list;
ITYPE *mcsr_chk;

ITYPE *baddr, *saddr;
ITYPE num_dense;

ITYPE *special;
ITYPE *special2;
ITYPE special_p;
char scr_pad[MAX_NTHREAD][SC_SIZE];
double p_elapsed;

bool running_sddmm = false;

int NTHREAD = 64;

int compare0(const void *a, const void *b)
{
	if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row > 0) return 1;
	if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row < 0) return -1;
	return ((struct v_struct *)a)->col - ((struct v_struct *)b)->col;
}

int compare1(const void *a, const void *b)
{
	if ((((struct v_struct *)a)->row)/BH - (((struct v_struct *)b)->row)/BH > 0) return 1;
	if ((((struct v_struct *)a)->row)/BH - (((struct v_struct *)b)->row)/BH < 0) return -1;
	if (((struct v_struct *)a)->col - ((struct v_struct *)b)->col > 0) return 1;
	if (((struct v_struct *)a)->col - ((struct v_struct *)b)->col < 0) return -1;
	return ((struct v_struct *)a)->row - ((struct v_struct *)b)->row;
}

int compare2(const void *a, const void *b)
{
	if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row > 0) return 1;
	if (((struct v_struct *)a)->row - ((struct v_struct *)b)->row < 0) return -1;
	if (((struct v_struct *)a)->grp - ((struct v_struct *)b)->grp > 0) return 1;
	if (((struct v_struct *)a)->grp - ((struct v_struct *)b)->grp < 0) return -1;
	return ((struct v_struct *)a)->col - ((struct v_struct *)b)->col;
}


void ready(int argc, char **argv)
{
	FILE *fp;
	ITYPE *loc;
	char buf[300];
	int nflag, sflag;
	int pre_count=0, tmp_ne;
	int i;

    if (argc < 2) {
        printf("Usage: ./SpMM <Matrix File> <Feature> <layers> <Iter>\n");
        exit(EXIT_FAILURE);
    }

	in_file = argv[1]; // record the input filename

	// fprintf(stdout, "TTAAGG,%s,", argv[1]);
	printf("Matrix: %s\n", basename(argv[1]));

	////sc = atoi(argv[2]);
	sc=128;
	if (argc > 2) {
		sc = (ITYPE) atoi(argv[2]);
        Tk = sc;
	}

	if (argc > 3) {
		layers = (ITYPE) atoi(argv[3]);
	}

	if (argc > 4) {
		ITER = (ITYPE) atoi(argv[4]);
	}

    if (argc > 5) {
        NTHREAD = (int) atoi(argv[5]);
    }

    if (argc > 6) {
        Tk = (ITYPE) atoi(argv[6]);
    }

	if (argc > 7) {
		BH = (ITYPE) atoi(argv[7]);
		BW = BH;
		LOG_BH = (ITYPE) log2(BH);
	} else {
		BH = 128;
		BW = BH;
		LOG_BH = 7;
	}
	MIN_OCC = ((BW * 3)/4);

	if (argc > 8) {
		if (strcmp(argv[8], "sddmm") == 0) {
			running_sddmm = true;
		} else {
			running_sddmm = false;
		}
		// running_sddmm = (bool) atoi(argv[8]);
	}

	printf("feature: %d\n", sc);
	printf("layers: %d\n", layers);
	printf("Ti: %d\n", BH);
	printf("Tj: %d\n", BW);
    printf("MIN_OCC: %d\n", MIN_OCC);
    printf("Tk: %d\n", Tk);
	printf("ITER: %d\n", ITER);
	printf("running_sddmm: %d\n", running_sddmm);

    std::cout << "NTHREADS: " << NTHREAD << std::endl;
    omp_set_num_threads(NTHREAD);

	fp = fopen(argv[1], "r");
	fgets(buf, 300, fp);
	if(strstr(buf, "symmetric") != NULL || strstr(buf, "Hermitian") != NULL) sflag = 1; // symmetric
	else sflag = 0;
	if(strstr(buf, "pattern") != NULL) nflag = 0; // non-value
	else if(strstr(buf, "complex") != NULL) nflag = -1;
	else nflag = 1;

	std::cout << "has values: " << nflag << std::endl;

#ifdef SYM
	sflag = 1;
#endif

	while(1) {
			pre_count++;
			fgets(buf, 300, fp);
			if(strstr(buf, "%") == NULL) break;
	}
	fclose(fp);

	fp = fopen(argv[1], "r");
	for(i=0;i<pre_count;i++)
			fgets(buf, 300, fp);

	fscanf(fp, "%d %d %d", &nr, &nc, &ne);
	nr0 = nr;
	ne *= (sflag+1);
	nr = CEIL(nr,BH)*BH;
	npanel = CEIL(nr,BH);


	temp_v = (struct v_struct *)malloc(sizeof(struct v_struct)*(ne+1));
	gold_temp_v = (struct v_struct *)malloc(sizeof(struct v_struct)*(ne+1));

	for(i=0;i<ne;i++) {
			fscanf(fp, "%d %d", &temp_v[i].row, &temp_v[i].col);
	temp_v[i].grp = INIT_GRP;
			temp_v[i].row--; temp_v[i].col--;

			if(temp_v[i].row < 0 || temp_v[i].row >= nr || temp_v[i].col < 0 || temp_v[i].col >= nc) {
					fprintf(stdout, "A vertex id is out of range %d %d\n", temp_v[i].row, temp_v[i].col);
					exit(0);
			}
			if(nflag == 0) temp_v[i].val = (FTYPE)(rand()%1048576)/1048576;
			else if(nflag == 1) {
					FTYPE ftemp;
					fscanf(fp, " %lf ", &ftemp);
					temp_v[i].val = ftemp;
                    // std::cout << "Location: " << temp_v[i].row << ", " << temp_v[i].col << " -- " << ftemp << std::endl;
			} else { // complex
					FTYPE ftemp1, ftemp2;
					fscanf(fp, " %lf %lf ", &ftemp1, &ftemp2);
					temp_v[i].val = ftemp1;
			}
#ifdef SIM_VALUE
temp_v[i].val = 1.0f;
#endif
			if(sflag == 1) {
					i++;
					temp_v[i].row = temp_v[i-1].col;
					temp_v[i].col = temp_v[i-1].row;
					temp_v[i].val = temp_v[i-1].val;
			temp_v[i].grp = INIT_GRP;
		}
	}
	qsort(temp_v, ne, sizeof(struct v_struct), compare0);

	loc = (ITYPE *)malloc(sizeof(ITYPE)*(ne+1));

	memset(loc, 0, sizeof(ITYPE)*(ne+1));
	loc[0]=1;
	for(i=1;i<ne;i++) {
			if(temp_v[i].row == temp_v[i-1].row && temp_v[i].col == temp_v[i-1].col)
					loc[i] = 0;
			else loc[i] = 1;
	}
	for(i=1;i<=ne;i++)
			loc[i] += loc[i-1];
	for(i=ne; i>=1; i--)
			loc[i] = loc[i-1];
	loc[0] = 0;

	for(i=0;i<ne;i++) {
			temp_v[loc[i]].row = temp_v[i].row;
			temp_v[loc[i]].col = temp_v[i].col;
			temp_v[loc[i]].val = temp_v[i].val;
			temp_v[loc[i]].grp = temp_v[i].grp;
	}
	ne = loc[ne];
	temp_v[ne].row = nr;
	gold_ne = ne;
	for(i=0;i<=ne;i++) {
			gold_temp_v[i].row = temp_v[i].row;
			gold_temp_v[i].col = temp_v[i].col;
			gold_temp_v[i].val = temp_v[i].val;
			gold_temp_v[i].grp = temp_v[i].grp;
	}
	free(loc);


    std::cout << "M: " << nr0 << " N: " << nc << " NNZ: " << ne << std::endl;

	// csr_v = (ITYPE *)malloc(sizeof(ITYPE)*(nr+1));
	// csr_e0 = (ITYPE *)malloc(sizeof(ITYPE)*ne);
	// csr_ev0 = (FTYPE *)malloc(sizeof(FTYPE)*ne);

    csr_v = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE)*(nr+1) );
	csr_e0 = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE)*ne );
	csr_ev0 = (FTYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(FTYPE)*ne );


	memset(csr_v, 0, sizeof(ITYPE)*(nr+1));

	for(i=0;i<ne;i++) {
			csr_e0[i] = temp_v[i].col;
			csr_ev0[i] = temp_v[i].val;
			csr_v[1+temp_v[i].row] = i+1;
	}

	for(i=1;i<nr;i++) {
			if(csr_v[i] == 0) csr_v[i] = csr_v[i-1];
	}
	csr_v[nr] = ne;

    // csr_e = (ITYPE *) malloc (sizeof(ITYPE)*ne);
	// csr_ev = (FTYPE *)malloc(sizeof(FTYPE)*ne);


	csr_e = (ITYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(ITYPE)*ne );
	csr_ev = (FTYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeof(FTYPE)*ne );

	fprintf(stdout,"%d,%d,%d\n",nr0,nc,ne);
	// printf("M: %ld\nN: %ld\nNNZ: %ld\n", nr0, nc, ne);
}

void gen()
{
	special = (ITYPE *)malloc(sizeof(ITYPE)*ne);
	special2 = (ITYPE *)malloc(sizeof(ITYPE)*ne);
	memset(special, 0, sizeof(ITYPE)*ne);
	memset(special2, 0, sizeof(ITYPE)*ne);


	mcsr_cnt = (ITYPE *)malloc(sizeof(ITYPE)*(npanel+1));
	mcsr_chk = (ITYPE *)malloc(sizeof(ITYPE)*(npanel+1));
	mcsr_e = (ITYPE *)malloc(sizeof(ITYPE)*ne); // reduced later
	memset(mcsr_cnt, 0, sizeof(ITYPE)*(npanel+1));
	memset(mcsr_chk, 0, sizeof(ITYPE)*(npanel+1));
	memset(mcsr_e, 0, sizeof(ITYPE)*ne);

	ITYPE bv_size = CEIL(nc, 32);
	unsigned int **bv = (unsigned int **)malloc(sizeof(unsigned int *)*MAX_NTHREAD);
	for(int i=0;i<MAX_NTHREAD;i++)
		bv[i] = (unsigned int *)malloc(sizeof(unsigned int)*bv_size);
	ITYPE **csr_e1 = (ITYPE **)malloc(sizeof(ITYPE *)*2);
	ITYPE **coo = (ITYPE **)malloc(sizeof(ITYPE *)*2);
	for(int i=0;i<2;i++) {
		csr_e1[i] = (ITYPE *)malloc(sizeof(ITYPE)*ne);
		coo[i] = (ITYPE *)malloc(sizeof(ITYPE)*ne);
	}

	struct timeval tt1, tt2, tt3, tt4;
	struct timeval starttime0;

	struct timeval starttime, endtime;
	gettimeofday(&starttime0, NULL);

	// filtering(WILL)
	//memcpy(csr_e1[0], csr_e0, sizeof(int)*ne);
    #pragma omp parallel for num_threads(NTHREAD) schedule(OMP_SCHEDULE, 1)
	for(int row_panel=0; row_panel<nr/BH; row_panel++) {
		for(int i=row_panel*BH; i<(row_panel+1)*BH; i++) {
			for(int j=csr_v[i]; j<csr_v[i+1]; j++) {
				csr_e1[0][j] = csr_e0[j];
			}
		}

	}

	gettimeofday(&starttime, NULL);

    #pragma omp parallel for num_threads(NTHREAD) schedule(OMP_SCHEDULE, 1)
	for(int row_panel=0; row_panel<nr/BH; row_panel++) {
		int tid = omp_get_thread_num();
		int i, j, t_sum=0;

		// coo generate and mcsr_chk
		memset(scr_pad[tid], 0, sizeof(char)*SC_SIZE);
		for(i=row_panel*BH; i<(row_panel+1)*BH; i++) {
			for(j=csr_v[i]; j<csr_v[i+1]; j++) {
				coo[0][j] = (i&(BH-1));
				int k = (csr_e0[j]&(SC_SIZE-1));
				if(scr_pad[tid][k] < THRESHOLD) {
					if(scr_pad[tid][k] == THRESHOLD - 1) t_sum++;
					scr_pad[tid][k]++;
				}
			}
		}

		if (t_sum < MIN_OCC) {
			mcsr_chk[row_panel] = 1;
			mcsr_cnt[row_panel+1] = 1;
			continue;
		}

		// sorting(merge sort)
		int flag = 0;
		for(int stride = 1; stride <= BH/2; stride *= 2, flag=1-flag) {
			for(int pivot = row_panel*BH; pivot < (row_panel+1)*BH; pivot += stride*2) {
				int l1, l2;
				for(i = l1 = csr_v[pivot], l2 = csr_v[pivot+stride]; l1 < csr_v[pivot+stride] && l2 < csr_v[pivot+stride*2]; i++) {
					if(csr_e1[flag][l1] <= csr_e1[flag][l2]) {
						coo[1-flag][i] = coo[flag][l1];
						csr_e1[1-flag][i] = csr_e1[flag][l1++];
					}
					else {
						coo[1-flag][i] = coo[flag][l2];
						csr_e1[1-flag][i] = csr_e1[flag][l2++];
					}
				}
				while(l1 < csr_v[pivot+stride]) {
					coo[1-flag][i] = coo[flag][l1];
					csr_e1[1-flag][i++] = csr_e1[flag][l1++];
				}
				while(l2 < csr_v[pivot+stride*2]) {
					coo[1-flag][i] = coo[flag][l2];
					csr_e1[1-flag][i++] = csr_e1[flag][l2++];
				}
			}
		}

		ITYPE weight=1;

		ITYPE cq=0, cr=0;

		// dense bit extract (and mcsr_e making)
		for(i=csr_v[row_panel*BH]+1; i<csr_v[(row_panel+1)*BH]; i++) {
			if(csr_e1[flag][i-1] == csr_e1[flag][i]) weight++;
			else {
				if(weight >= THRESHOLD) {
					cr++;
				} 				//if(cr == BW) { cq++; cr=0;}
				weight = 1;
			}
		}
		//int reminder = (csr_e1[flag][i-1]&31);
		if(weight >= THRESHOLD) {
			cr++;
		} 		//if(cr == BW) { cq++; cr=0; }
		// TODO = occ control
		mcsr_cnt[row_panel+1] = CEIL(cr,BW)+1;

	}

	////gettimeofday(&tt1, NULL);
	// prefix-sum
	for(int i=1; i<=npanel;i++)
		mcsr_cnt[i] += mcsr_cnt[i-1];
	//mcsr_e[0] = 0;
	mcsr_e[BH * mcsr_cnt[npanel]] = ne;

	////gettimeofday(&tt2, NULL);

    #pragma omp parallel for num_threads(NTHREAD) schedule(OMP_SCHEDULE, 1)
	for(int row_panel=0; row_panel<nr/BH; row_panel++) {
		int tid = omp_get_thread_num();
	    if(mcsr_chk[row_panel] == 0) {
			int i, j;
			int flag = 0;
			ITYPE cq=0, cr=0;
			for(int stride = 1; stride <= BH/2; stride*=2, flag=1-flag);
			int base = (mcsr_cnt[row_panel]*BH);
			int mfactor = mcsr_cnt[row_panel+1] - mcsr_cnt[row_panel];
			ITYPE weight=1;

			// mcsr_e making
			for(i=csr_v[row_panel*BH]+1; i<csr_v[(row_panel+1)*BH]; i++) {
				if(csr_e1[flag][i-1] == csr_e1[flag][i]) weight++;
				else {
					int reminder = (csr_e1[flag][i-1]&31);
					if(weight >= THRESHOLD) {
						cr++;
						bv[tid][csr_e1[flag][i-1]>>5] |= (1<<reminder);
						for(j=i-weight; j<=i-1; j++) {
							mcsr_e[base + coo[flag][j] * mfactor + cq + 1]++;
						}
					} else {
						//bv[tid][csr_e1[flag][i-1]>>5] &= (~0 - (1<<reminder));
						bv[tid][csr_e1[flag][i-1]>>5] &= (0xFFFFFFFF - (1<<reminder));
					}
					if(cr == BW) { cq++; cr=0;}
					weight = 1;
				}
			}

			//fprintf(stderr, "inter : %d\n", i);

			ITYPE reminder = (csr_e1[flag][i-1]&31);
			if(weight >= THRESHOLD) {
				cr++;
				bv[tid][csr_e1[flag][i-1]>>5] |= (1<<reminder);
				for(j=i-weight; j<=i-1; j++) {
					mcsr_e[base + coo[flag][j] * mfactor + cq + 1]++;
				}
			} else {
				bv[tid][csr_e1[flag][i-1]>>5] &= (0xFFFFFFFF - (1<<reminder));
			}
			// reordering
			int delta = mcsr_cnt[row_panel+1] - mcsr_cnt[row_panel];
			int base0 = mcsr_cnt[row_panel]*BH;
			for(i=row_panel*BH; i<(row_panel+1)*BH; i++) {
				int base = base0+(i-row_panel*BH)*delta;
				int dpnt = mcsr_e[base] = csr_v[i];
				for(j=1;j<delta;j++) {
					mcsr_e[base+j] += mcsr_e[base+j-1];
				}
				int spnt=mcsr_e[mcsr_cnt[row_panel]*BH + (mcsr_cnt[row_panel+1] - mcsr_cnt[row_panel])*(i - row_panel*BH + 1) - 1];

				avg0[tid] += csr_v[i+1] - spnt;
				for(j=csr_v[i]; j<csr_v[i+1]; j++) {
					int k = csr_e0[j];
					if((bv[tid][k>>5]&(1<<(k&31)))) {
						csr_e[dpnt] = csr_e0[j];
						csr_ev[dpnt++] = csr_ev0[j];
					} else {
						csr_e[spnt] = csr_e0[j];
						csr_ev[spnt++] = csr_ev0[j];
					}
				}
			}
	   } else {
		int base0 = mcsr_cnt[row_panel]*BH;
		memcpy(&mcsr_e[base0], &csr_v[row_panel*BH], sizeof(ITYPE)*BH);
		avg0[tid] += csr_v[(row_panel+1)*BH] - csr_v[row_panel*BH];
		int bidx = csr_v[row_panel*BH];
		int bseg = csr_v[(row_panel+1)*BH] - bidx;
		memcpy(&csr_e[bidx], &csr_e0[bidx], sizeof(ITYPE)*bseg);
		memcpy(&csr_ev[bidx], &csr_ev0[bidx], sizeof(FTYPE)*bseg);

	   }
	}


	for(int i=0;i<NTHREAD;i++)
		avg += avg0[i];
	avg /= (double)nr;

	////gettimeofday(&tt3, NULL);

	for(int i=0;i<nr;i++) {
		ITYPE idx = (mcsr_cnt[i>>LOG_BH])*BH + (mcsr_cnt[(i>>LOG_BH)+1] - mcsr_cnt[i>>LOG_BH])*((i&(BH-1))+1);
		ITYPE diff = csr_v[i+1] - mcsr_e[idx-1];

		// if ( i < 10) {
			// std::cout << "idx: " << idx << std::endl;
			// std::cout << "Row has: " << csr_v[i+1] - csr_v[i] << " values " << std::endl;
			// std::cout << "diff: " << diff << std::endl;
		// }

		double r = ((double)diff - avg);
		vari += r * r;

		if(diff >= STHRESHOLD) {
			int pp = (diff) / STHRESHOLD;
			for(int j=0; j<pp; j++) {
				special[special_p] = i;
				special2[special_p] = j * STHRESHOLD;
				special_p++;

				// std::cout << "Adding special: " << i << ", " << j * STHRESHOLD << std::endl;
			}
		}



	}
	vari /= (double)nr;

	gettimeofday(&endtime, NULL);

    //     double elapsed0 = ((starttime.tv_sec-starttime0.tv_sec)*1000000 + starttime.tv_usec-starttime0.tv_usec)/1000000.0;
    //     //double elapsed1 = ((tt1.tv_sec-starttime.tv_sec)*1000000 + tt1.tv_usec-starttime.tv_usec)/1000000.0;
    //     //double elapsed2 = ((tt2.tv_sec-tt1.tv_sec)*1000000 + tt2.tv_usec-tt1.tv_usec)/1000000.0;
    //     //double elapsed3 = ((tt3.tv_sec-tt2.tv_sec)*1000000 + tt3.tv_usec-tt2.tv_usec)/1000000.0;
    //     //double elapsed4 = ((endtime.tv_sec-tt3.tv_sec)*1000000 + endtime.tv_usec-tt3.tv_usec)/1000000.0;
	// //fprintf(stdout, "(%f %f %f %f %f)", elapsed0*1000, elapsed1*1000, elapsed2*1000, elapsed3*1000, elapsed4*1000);
    //     p_elapsed = ((endtime.tv_sec-starttime.tv_sec)*1000000 + endtime.tv_usec-starttime.tv_usec)/1000000.0;
	// fprintf(stdout, "%f,%f,", elapsed0*1000, p_elapsed*1000);

	for(int i=0;i<NTHREAD;i++)
		free(bv[i]);
	for(int i=0;i<2;i++) {
		free(csr_e1[i]);
		free(coo[i]);
	}
	free(bv); free(csr_e1); free(coo);
}


// Intel intrinsic for reading the timestamp counter
// static inline unsigned long long readTSC() {
//     // _mm_lfence();  // optionally wait for earlier insns to retire before reading the clock
//     _mm_mfence();
//     unsigned long long val = _rdtsc();
//     // return __rdtsc();
//     // return __rdtscp();
//     // _mm_lfence();  // optionally block later instructions until rdtsc retires

//     _mm_mfence();
//     return val;
// }


#define RUN_SPARSE
#define RUN_DENSE
#define RUN_SPECIAL

// #define ASPT_TK

long long mprocess_kernel( FTYPE *vin, FTYPE *vout, long long *per_core_timing = nullptr, long long *per_panel_timing = nullptr )
{

	__assume_aligned(csr_v, 64);
	__assume_aligned(csr_e, 64);
	__assume_aligned(csr_ev, 64);
	//__assume_aligned(mcsr_cnt, 64);
	//__assume_aligned(mcsr_e, 64);
	//__assume_aligned(mcsr_list, 64);
	__assume_aligned(vin, 64);
	__assume_aligned(vout, 64);


    auto start_cycle = readTSC();
	// if(vari < 5000*1/1*1) {
	if(vari < 5000*1/1*1 || dont_run_special) {

		// Kernel loop
		#pragma ivdep
		#pragma vector aligned
		#pragma vector temporal
		#pragma omp parallel for num_threads(NTHREAD) schedule(OMP_SCHEDULE, 1)
		for(ITYPE row_panel=0; row_panel<nr/BH; row_panel ++) {

            //dense
            ITYPE stride = 0;

        #ifndef RUN_DENSE
            stride = mcsr_cnt[row_panel + 1] - mcsr_cnt[row_panel] - 1;
        #endif

        #ifdef RUN_DENSE
			for(stride = 0; stride < mcsr_cnt[row_panel+1]-mcsr_cnt[row_panel]-1; stride++) {

				for(ITYPE i=row_panel*BH; i<(row_panel+1)*BH; i++) {
						ITYPE dummy = mcsr_cnt[row_panel]*BH + (i&(BH-1))*(mcsr_cnt[row_panel+1] - mcsr_cnt[row_panel]) + stride;
					ITYPE loc1 = mcsr_e[dummy], loc2 = mcsr_e[dummy+1];

					ITYPE interm = loc1 + (((loc2 - loc1)>>3)<<3);
					ITYPE j;
					for(j=loc1; j<interm; j+=8) {
						#pragma ivdep
						#pragma vector nontemporal (csr_ev)
						#pragma prefetch vin:_MM_HINT_T1
						for(ITYPE k=0; k<sc; k++) {
							vout[i*sc+k] = vout[i*sc+k] + csr_ev[j] * vin[csr_e[j]*sc + k]
							+ csr_ev[j+1] * vin[csr_e[j+1]*sc + k]
							+ csr_ev[j+2] * vin[csr_e[j+2]*sc + k]
							+ csr_ev[j+3] * vin[csr_e[j+3]*sc + k]
							+ csr_ev[j+4] * vin[csr_e[j+4]*sc + k]
							+ csr_ev[j+5] * vin[csr_e[j+5]*sc + k]
							+ csr_ev[j+6] * vin[csr_e[j+6]*sc + k]
							+ csr_ev[j+7] * vin[csr_e[j+7]*sc + k];
						}
					}
					for(; j<loc2; j++) {
						#pragma ivdep
						#pragma vector nontemporal (csr_ev)
						#pragma prefetch vout:_MM_HINT_T1
						for(ITYPE k=0; k<sc; k++) {
							vout[i*sc + k] += csr_ev[j] * vin[csr_e[j]*sc + k];
						}
					}
				}
			}
        #endif // RUN_DENSE

        #ifdef RUN_SPARSE
			//sparse
			ITYPE ilim = MIN( (row_panel+1)*BH, nr0 );
			// for(ITYPE i=row_panel*BH; i<(row_panel+1)*BH; i++) {
			for(ITYPE i=row_panel*BH; i<ilim; i++) {

				ITYPE dummy = mcsr_cnt[row_panel]*BH + (i&(BH-1))*(mcsr_cnt[row_panel+1] - mcsr_cnt[row_panel]) + stride;
				ITYPE loc1 = mcsr_e[dummy], loc2 = mcsr_e[dummy+1];

				ITYPE interm = loc1 + (((loc2 - loc1)>>3)<<3);
				ITYPE j;
				for(j=loc1; j<interm; j+=8) {
					#pragma ivdep
					#pragma vector nontemporal (csr_ev)
					#pragma prefetch vin:_MM_HINT_T1
					for(ITYPE k=0; k<sc; k++) {
						vout[i*sc+k] = vout[i*sc+k] + csr_ev[j] * vin[csr_e[j]*sc + k]
						+ csr_ev[j+1] * vin[csr_e[j+1]*sc + k]
						+ csr_ev[j+2] * vin[csr_e[j+2]*sc + k]
						+ csr_ev[j+3] * vin[csr_e[j+3]*sc + k]
						+ csr_ev[j+4] * vin[csr_e[j+4]*sc + k]
						+ csr_ev[j+5] * vin[csr_e[j+5]*sc + k]
						+ csr_ev[j+6] * vin[csr_e[j+6]*sc + k]
						+ csr_ev[j+7] * vin[csr_e[j+7]*sc + k];
					}
				}
				for(; j<loc2; j++) {
					#pragma ivdep
					#pragma vector nontemporal (csr_ev)
					#pragma prefetch vout:_MM_HINT_T1
					for(ITYPE k=0; k<sc; k++) {
						vout[i*sc + k] += csr_ev[j] * vin[csr_e[j]*sc + k];
					}
				}
			}

        #endif // RUN_SPARSE

        }	// end row panel loop

	// printf("Iter: %d, duration: %f\n", loop, duration);
	} else { // big var

		#pragma ivdep
		#pragma vector aligned
		#pragma omp parallel for num_threads(NTHREAD) schedule(OMP_SCHEDULE, 1)
		for(ITYPE row_panel=0; row_panel<nr/BH; row_panel ++) {
			//dense

            ITYPE stride = 0;
        #ifndef RUN_DENSE
            stride = mcsr_cnt[row_panel + 1] - mcsr_cnt[row_panel] - 1;
        #endif

        #ifdef RUN_DENSE
			for(stride = 0; stride < mcsr_cnt[row_panel+1]-mcsr_cnt[row_panel]-1; stride++) {

				for(ITYPE i=row_panel*BH; i<(row_panel+1)*BH; i++) {
						ITYPE dummy = mcsr_cnt[row_panel]*BH + (i&(BH-1))*(mcsr_cnt[row_panel+1] - mcsr_cnt[row_panel]) + stride;
					ITYPE loc1 = mcsr_e[dummy], loc2 = mcsr_e[dummy+1];

					ITYPE interm = loc1 + (((loc2 - loc1)>>3)<<3);
					ITYPE j;
					for(j=loc1; j<interm; j+=8) {
						#pragma ivdep
						#pragma vector nontemporal (csr_ev)
						#pragma prefetch vin:_MM_HINT_T1
						for(ITYPE k=0; k<sc; k++) {
							vout[i*sc+k] = vout[i*sc+k] + csr_ev[j] * vin[csr_e[j]*sc + k]
							+ csr_ev[j+1] * vin[csr_e[j+1]*sc + k]
							+ csr_ev[j+2] * vin[csr_e[j+2]*sc + k]
							+ csr_ev[j+3] * vin[csr_e[j+3]*sc + k]
							+ csr_ev[j+4] * vin[csr_e[j+4]*sc + k]
							+ csr_ev[j+5] * vin[csr_e[j+5]*sc + k]
							+ csr_ev[j+6] * vin[csr_e[j+6]*sc + k]
							+ csr_ev[j+7] * vin[csr_e[j+7]*sc + k];
						}
					}
					for(; j<loc2; j++) {
						#pragma ivdep
						#pragma vector nontemporal (csr_ev)
						#pragma prefetch vout:_MM_HINT_T1
						for(ITYPE k=0; k<sc; k++) {
							vout[i*sc + k] += csr_ev[j] * vin[csr_e[j]*sc + k];
						}
					}
				}

			}


        #endif // RUN_DENSE

        #ifdef RUN_SPARSE
            //sparse
			for(ITYPE i=row_panel*BH; i<(row_panel+1)*BH; i++) {

				ITYPE dummy = mcsr_cnt[row_panel]*BH + (i&(BH-1))*(mcsr_cnt[row_panel+1] - mcsr_cnt[row_panel]) + stride;
				ITYPE loc1 = mcsr_e[dummy], loc2 = mcsr_e[dummy+1];

				loc1 += ((loc2 - loc1)/STHRESHOLD)*STHRESHOLD;

				ITYPE interm = loc1 + (((loc2 - loc1)>>3)<<3);
				ITYPE j;
				for(j=loc1; j<interm; j+=8) {
					#pragma ivdep
					#pragma vector nontemporal (csr_ev)
					#pragma prefetch vin:_MM_HINT_T1
					for(ITYPE k=0; k<sc; k++) {
						vout[i*sc+k] = vout[i*sc+k] + csr_ev[j] * vin[csr_e[j]*sc + k]
						+ csr_ev[j+1] * vin[csr_e[j+1]*sc + k]
						+ csr_ev[j+2] * vin[csr_e[j+2]*sc + k]
						+ csr_ev[j+3] * vin[csr_e[j+3]*sc + k]
						+ csr_ev[j+4] * vin[csr_e[j+4]*sc + k]
						+ csr_ev[j+5] * vin[csr_e[j+5]*sc + k]
						+ csr_ev[j+6] * vin[csr_e[j+6]*sc + k]
						+ csr_ev[j+7] * vin[csr_e[j+7]*sc + k];
					}
				}
				for(; j<loc2; j++) {
					#pragma ivdep
					#pragma vector nontemporal (csr_ev)
					#pragma prefetch vout:_MM_HINT_T1
					for(ITYPE k=0; k<sc; k++) {
						vout[i*sc + k] += csr_ev[j] * vin[csr_e[j]*sc + k];
					}
				}
			}

        #endif // RUN_SPARSE

		} // end row panel loop


    #ifdef RUN_SPECIAL

		#pragma ivdep
		#pragma vector aligned
		#pragma omp parallel for num_threads(NTHREAD) schedule(OMP_SCHEDULE, 1)
		for(ITYPE row_panel=0; row_panel<special_p;row_panel ++) {
			ITYPE i=special[row_panel];

			ITYPE dummy = mcsr_cnt[i>>LOG_BH]*BH + ((i&(BH-1))+1)*(mcsr_cnt[(i>>LOG_BH)+1] - mcsr_cnt[i>>LOG_BH]);

			ITYPE loc1 = mcsr_e[dummy-1] + special2[row_panel];
			ITYPE loc2 = loc1 + STHRESHOLD;

			//int interm = loc1 + (((loc2 - loc1)>>3)<<3);
			ITYPE j;
			//assume to 128
			FTYPE temp_r[MAX_FEATURE]={0,};
			//for(int e=0;e<128;e++) {
			//	temp_r[e] = 0.0f;
			//}

			for(j=loc1; j<loc2; j+=8) {
				#pragma ivdep
				#pragma vector nontemporal (csr_ev)
				#pragma prefetch vin:_MM_HINT_T1
				for(ITYPE k=0; k<sc; k++) {
					temp_r[k] = temp_r[k] + csr_ev[j] * vin[csr_e[j]*sc + k]
					+ csr_ev[j+1] * vin[csr_e[j+1]*sc + k]
					+ csr_ev[j+2] * vin[csr_e[j+2]*sc + k]
					+ csr_ev[j+3] * vin[csr_e[j+3]*sc + k]
					+ csr_ev[j+4] * vin[csr_e[j+4]*sc + k]
					+ csr_ev[j+5] * vin[csr_e[j+5]*sc + k]
					+ csr_ev[j+6] * vin[csr_e[j+6]*sc + k]
					+ csr_ev[j+7] * vin[csr_e[j+7]*sc + k];
				}
			}
			#pragma ivdep
			for(ITYPE k=0; k<sc; k++) {
				#pragma omp atomic
				vout[i*sc+k] += temp_r[k];
			}
		}

    #endif // RUN_SPECIAL

	}

    auto end_cycle = readTSC();

	return (end_cycle - start_cycle);

}


// CACHE FLUSH CODE

FTYPE *A = nullptr;
FTYPE *B = nullptr;
size_t cache_flush_size = 0;
bool cache_flush_init = false;

void init_cache_flush(ITYPE size)
{
    if (!cache_flush_init) {

        cache_flush_size = size;

        A = new FTYPE[size];
        B = new FTYPE[size];

        assert( A && B && "Cache Flush init failed" );

        std::cerr << "cache flush size: " << cache_flush_size << std::endl;

        #pragma omp parallel for schedule(static)
        for(int i=0; i<cache_flush_size; i++) {
            A[i] = ((double) (i % 100)) / ((double) 100.0);
            B[i] = ((double) (i % 100)) / ((double) 100.0);
        }

        cache_flush_init = true;
    }
}

void free_cache_flush()
{
    delete[] A;
    delete[] B;
}

#pragma intel optimization_level 1
void cache_flush(ITYPE num_threads)
{
    if (cache_flush_init == false) {
        return;
    }

    T total_sum = 0.0;

    #pragma omp parallel for schedule(static) reduction(+ : total_sum)
    for(int i=0; i<cache_flush_size; i++)
    {
        // partial_sums[my_tid] += A[i] + B[i];
        total_sum += A[i] + B[i];
    }

    std::cerr << total_sum << std::endl;
}
// END CACHE FLUSH CODE

void reset_matrix(FTYPE *arr, size_t size)
{
    #pragma omp parallel for
    for (size_t i = 0; i < size; i++) {
        arr[i] = (FTYPE) 0;
    }
}

void reset_matrix(FTYPE **arr, int layers, size_t size)
{
    for (int l = 0; l < layers; l++) {

        #pragma omp parallel for
        for (size_t i = 0; i < size; i++) {
            arr[l][i] = (FTYPE) 0;
        }
    }
}

// #define TIMING_ONLY
// #define RUN_CACHE_EVENTS
// #define RUN_STALL_EVENTS
// #define RUN_REG_EVENTS
// #define RUN_DRAM_EVENTS

// #define PAGE_SIZE ((size_t) 1024 * 1024 * 128 * 2)
// #define PAGE_SIZE ((size_t) 4 * 1024)


void check_correctness(FTYPE *I, FTYPE *O, FTYPE *O2)
{
    for ( ITYPE i = 0; i < (sc * nr0); i++ ) {
        O[i] = 0;
        O2[i] = 0;
    }

    mprocess_kernel( I, O );

    for (ITYPE i = 0; i < nr0; i++) {
        auto row_start = csr_v[i];
        auto row_end = csr_v[i + 1];
        for (ITYPE j0 = row_start; j0 < row_end; j0++) {
            ITYPE j = csr_e0[j0];

            for (ITYPE k = 0; k < sc; k++) {
                O2[i * sc + k] += csr_ev0[j0] * I[ j * sc + k ];
            }
        }
    }

    // compare and check
    for ( ITYPE i = 0; i < nr0; i++ ) {
        for ( ITYPE k = 0; k < sc; k++ ) {
            ITYPE index = i * sc + k;

            FTYPE a = O[index] < 0 ? -1.0 * O[index] : O[index];
            FTYPE b = O2[index] < 0 ? -1.0 * O2[index] : O2[index];

            T diff = ((a-b) < 0 ? (b-a) : (a-b));
            T max = (a > b ? a : b);
            if ( diff / max > 0.01 ) {
                // std::cout << "ERROR in computation E: " << O2[index] << " G: " << O[index] << std::endl;
                std::cout << "ERROR in computation: E: " << O2[index] << " G: " << O[index] << " index: " << index << " Matrix: " << basename(in_file) << std::endl;
                std::exit(EXIT_FAILURE);
            }
        }
    }

    std::cout << "ASpT Correct? 1" << std::endl;

    // std::exit(EXIT_SUCCESS);
}


void mprocess()
{

  	FTYPE *vin, *vout, *ocsr_ev;

    size_t sizeB = ((size_t) nc) * ((size_t) sc) * sizeof(FTYPE);
    size_t sizeB_rounded = CEIL(sizeB, PAGE_SIZE) * PAGE_SIZE;
    size_t sizeB_alloc = sizeB_rounded * ((size_t) layers);
    size_t countB = sizeB_rounded / sizeof(FTYPE);
    size_t countB_alloc = sizeB_alloc / sizeof(FTYPE);

    std::cerr << "SizeB: " << sizeB << std::endl;
    std::cerr << "SizeB_rounded: " << sizeB_rounded << std::endl;
    std::cerr << "CountB: " << countB << std::endl;

    size_t sizeC = ((size_t) nr) * ((size_t) sc) * sizeof(FTYPE);
    size_t sizeC_rounded = CEIL(sizeC, PAGE_SIZE) * PAGE_SIZE;
    size_t sizeC_alloc = sizeC_rounded * ((size_t) layers);
    size_t countC = sizeC_rounded / sizeof(FTYPE);
    size_t countC_alloc = sizeC_alloc / sizeof(FTYPE);

	size_t sizeO = ne * sizeof(FTYPE);
    size_t sizeO_rounded = CEIL(sizeO, PAGE_SIZE) * PAGE_SIZE;
    size_t countO = CEIL(sizeO_rounded, sizeof(FTYPE));


    FTYPE *check;
    FTYPE *aspt_check;

	check = (FTYPE *) std::aligned_alloc(ALLOC_ALIGNMENT, sizeC_rounded);
    aspt_check = (FTYPE *) std::aligned_alloc(ALLOC_ALIGNMENT, sizeC_rounded);

	// size_t sizeB = ((size_t) nc) * ((size_t) sc) * ((size_t) layers) * ((size_t) sizeof(FTYPE));
	// size_t sizeC = ((size_t) nr) * ((size_t) sc) * ((size_t) layers) * ((size_t) sizeof(FTYPE));
	// int64_t countB = ((int64_t) nc) * ((int64_t) sc);
	// int64_t countC = ((int64_t) nr) * ((int64_t) sc);

#ifdef LARGE_ALLOC
	FTYPE *B = (FTYPE *) std::aligned_alloc(ALLOC_ALIGNMENT, sizeB_alloc );
	FTYPE *C = (FTYPE *) std::aligned_alloc(ALLOC_ALIGNMENT, sizeC_alloc );
#else
    FTYPE *B[layers];
    FTYPE *C[layers];
	FTYPE *O[layers];

    for ( int i = 0; i < layers; i++ ) {
        B[i] = (FTYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeB_rounded );
        C[i] = (FTYPE *) std::aligned_alloc( ALLOC_ALIGNMENT, sizeC_rounded );
    }
#endif

	// init cache flush with the default size
    size_t cf_size = 1024 * 1024 * 8 * NTHREAD; // 8 MiB doubles per thread
	init_cache_flush(cf_size);

	// __assume_aligned(csr_v, 64);
	// __assume_aligned(csr_e, 64);
	// __assume_aligned(csr_ev, 64);
	//__assume_aligned(mcsr_cnt, 64);
	//__assume_aligned(mcsr_e, 64);
	//__assume_aligned(mcsr_list, 64);

	// __assume_aligned(B, 64);
	// __assume_aligned(C, 64);

	// __assume_aligned(vin, 64);
	// __assume_aligned(vout, 64);


	// Setup the input and the output arrays

#ifdef LARGE_ALLOC
    #pragma omp parallel for
	for (size_t i = 0; i < countB_alloc ; i++) {
		B[i] = ((FTYPE) (i % 100)) / (FTYPE) 100;
	}
    reset_matrix(C, countC_alloc);
#else
    for ( int l = 0; l < layers; l++) {
        #pragma omp parallel for
        for (size_t i = 0; i < countB ; i++) {
            B[l][i] = ((FTYPE) (i % 100)) / (FTYPE) 100;
        }
    }


    reset_matrix(C, layers, countC);
#endif


    check_correctness(B[0], check, aspt_check);

	// Start with warmup run and then go to the next thing

	{ 	// Warmup
		stats_t<long long, ITYPE> duration_times;
        // stats_t<long long, ITYPE> duration_cycle_times("duration cycle counts");

        duration_times.name = "duration times";

        long long per_core_timing[NTHREAD];

		// Actual loop for the kernel calls
		for(ITYPE loop=0;loop<ITER / WARMUP_DIVIDER;loop++) {
        #ifdef LARGE_ALLOC
			vin = &B[ ((int64_t)(loop % layers)) * (countB) ];
			vout = &C[ ((int64_t)(loop % layers)) * (countC) ];
        #else
            vin = B[ (loop % layers) ];
            vout = C[ (loop % layers) ];
			ocsr_ev = O[ (loop % layers) ];
        #endif
            long long *iter_panel_cycle_counts = nullptr;
            #ifdef PER_PANEL_TIMING
                iter_panel_cycle_counts = &per_panel_cycle_counts[ num_panels * loop ];
            #endif
			cache_flush(NTHREAD);

			long long kernel_time;

			kernel_time = mprocess_kernel( vin, vout, per_core_timing, iter_panel_cycle_counts );


			duration_times.insert(kernel_time);
		}
		duration_times.process();
		duration_times.print();
        double execution_time = ((double) duration_times.median) / CLOCK_FREQUENCY;
		printf("Warmup Median Time: %f\n", execution_time);

    #ifdef LARGE_ALLOC
		reset_matrix(C, countC_alloc );
    #else
        reset_matrix(C, layers, countC);
    #endif

    }



	{ // Timing
		stats_t<long long, ITYPE> duration_times;
		duration_times.name = "duration times";

		long long per_core_timing[NTHREAD];
		#ifdef PER_CORE_TIMING
			stats_t<long long, ITYPE> per_core_timing_stats[NTHREAD];
			for (ITYPE i = 0; i < NTHREAD; i++) {
				per_core_timing_stats[i].name = "core:" + std::to_string(i);
			}
		#endif

		#ifdef PER_PANEL_TIMING
			ITYPE num_panels = nr / BH;
			long long *per_panel_cycle_counts = new long long[num_panels * ITER];
		#endif

		// Actual loop for the kernel calls
		for(ITYPE loop=0;loop<ITER;loop++) {
			#ifdef LARGE_ALLOC
				vin = &B[ ((int64_t)(loop % layers)) * (countB) ];
				vout = &C[ ((int64_t)(loop % layers)) * (countC) ];
			#else
				vin = B[ (loop % layers) ];
				vout = C[ (loop % layers) ];
				ocsr_ev = O[ (loop % layers) ];
			#endif
			long long *iter_per_panel_cycle_counts = nullptr;
			#ifdef PER_PANEL_TIMING
				iter_per_panel_cycle_counts = &per_panel_cycle_counts[loop * num_panels];
			#endif
			cache_flush(NTHREAD);
			long long kernel_time;

			kernel_time = mprocess_kernel( vin, vout, per_core_timing, iter_per_panel_cycle_counts );

			duration_times.insert(kernel_time);

			#ifdef PER_CORE_TIMING
				for (ITYPE i = 0; i < NTHREAD; i++) {
					per_core_timing_stats[i].insert(per_core_timing[i]);
				}
			#endif

		}
		duration_times.process();
		duration_times.print();
		double execution_time = ((double) duration_times.median) / CLOCK_FREQUENCY;
		printf("Median Time: %f\n", execution_time);


		auto nnz_count = ne;
		std::cout << "NNZ Count: " << nnz_count << std::endl;
		double nnz_fraction = ((double) nnz_count) / ((double) ne);
		std::cout << "NNZ Fraction: " << nnz_fraction << std::endl;
		size_t flop_count = ((size_t)2) * ((size_t)nnz_count) * ((size_t)sc);
		double gflops = ((double) flop_count) / execution_time / 1E9;
		std::cout << "GFLOPS: " << gflops << std::endl;
	}


	// release memory

    #ifdef LARGE_ALLOC
        free(B);
	    free(C);
    #else
        for ( int i = 0; i < layers; i++ ) {
            std::free(B[i]);
            std::free(C[i]);
        }
    #endif
}

int main(int argc, char **argv)
{
	ready(argc, argv);
    auto start_time = std::chrono::high_resolution_clock::now();
	gen();
    auto end_time = std::chrono::high_resolution_clock::now();
    double gen_time = std::chrono::duration<double>(end_time - start_time).count();
    // std::cout << "GEN TIME: " << basename(argv[1]) << ", " << gen_time << std::endl;

    /*
	std::string out_path_prefix = "/nethome/ajain324/USERSCRATCH/ssData/large_dataset_stm/";
	std::string in_path = std::string(in_file);
	std::string out_path = out_path_prefix + in_path.substr( in_path.find_last_of("/") + 1,
						in_path.find_last_of(".") - in_path.find_last_of("/") - 1 ) + ".stm";

	std::cout << "Serializing arrays to disk " << out_path << std::endl;

	serialize_array(out_path);

	deserialize_array(out_path);
    */


    // kernel_tile_size_histogram();
    // kernel_tile_running_histogram();
	// kernel_tile_volume_histogram();
    // kernel_tile_nnzs_histogram();
	// gen_structure();

    mprocess();
}


