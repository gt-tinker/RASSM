// This code has been adapted from the original code of the jstream paper from SC'20
// https://inria.hal.science/hal-03117491/file/main.pdf

#ifndef MODEL_CC
#define MODEL_CC

#include "model.h"
#include "common.h"
//#define LOGC(X) ceil(log2(X))
/*
#define LOGC(X) logc(X)
#define REVERSE_LOGC(X) pow(2,X)
#define AVERAGE_BIN(X) X*3/4
*/

#define FACTOR 1
#define LOGC(X) X/FACTOR
#define REVERSE_LOGC(X) X*FACTOR
#define AVERAGE_BIN(X) (2*X)/2

uint64_t ** intervalsp = new uint64_t*[1];
uint64_t ** notsmallerp = new uint64_t*[1];
uint64_t ** notsmallerweightedp = new uint64_t*[1];


unsigned int logc(unsigned int n)
{
	if (n == 0)
	return 0;

	unsigned int msb = 0;
	while (n != 0)
	{
		n = n >> 1;
		msb++;
	}

	return (msb);
}

// gold_temp_v is the input matrix in coordinate format column-major sorted
int init_model(int nr, int nc, int ne, struct v_struct *gold_temp_v)
{
    size_t init_size = nr + 1;
    if ( nc > nr ) {
        init_size = nc + 1;
    }
	uint64_t* intervals = new uint64_t[init_size];
	uint64_t* notsmaller = new uint64_t[init_size];
	uint64_t* notsmallerweighted = new uint64_t[init_size];
	for(uint64_t i = 0; i< nc+1 ; i++)
	{
		intervals [i] = 0;
		notsmaller [i] = 0;
		notsmallerweighted[i] = 0;
	}
	struct timeval starttime, midtime, endtime,timediff;
	double elapsed=0;

	uint64_t i=0;
	for (uint64_t j=0; j<nc ; j++)
	{
		uint64_t lasti = -1;
		while(gold_temp_v[i].col == j)
		{
			uint64_t d = gold_temp_v[i].row - lasti - 1;
			intervals[d] += 1;
			lasti = gold_temp_v[i].row;
			i++;
		}
		uint64_t d = nr - lasti - 1;
		intervals[d] += 1;
	}

	elapsed = ((midtime.tv_sec-starttime.tv_sec)*1000000 + midtime.tv_usec-starttime.tv_usec)/1000000.0;

	notsmaller[nr] = intervals[nr];
	notsmallerweighted[nr] = nr*intervals[nr];

	for(uint64_t Ti = nc-1; Ti > 0 ; Ti--)
	{
		notsmaller [Ti] = notsmaller [Ti + 1] + intervals[Ti];
		notsmallerweighted[Ti] += notsmallerweighted [Ti + 1] + Ti*intervals[Ti];
	}

    /*
    std::cout << "notsmaller: ";
    for (int i = 0; i < nr+1; i++) {
        if (i % 1000 == 0) {
            std::cout << notsmaller[i] << ", ";
        }
    }
    std::cout <<  std::endl;
    */

	/*
	for(uint64_t Ti = 1; Ti <= nc ; Ti++)
	{
		notsmaller [Ti] = notsmaller [Ti -1] + intervals[Ti];
		notsmallerweighted[Ti] += notsmallerweighted [Ti - 1] + Ti*intervals[Ti];
	}
	*/


	elapsed = ((endtime.tv_sec-starttime.tv_sec)*1000000 + endtime.tv_usec-starttime.tv_usec)/1000000.0;
	*intervalsp = intervals;
	*notsmallerp = notsmaller;
	*notsmallerweightedp = notsmallerweighted;

	return 0;
}

uint64_t active_cols(int Ti, int nr, int nc, int ne)
{
	// if (intervalsp == NULL || notsmallerp == NULL || notsmallerweightedp == NULL)
	// {
	// 	init_model(nr, nc, ne);
	// }

	assert( intervalsp != NULL && notsmallerp != NULL && notsmallerweightedp != NULL);

	uint64_t* intervals = *intervalsp;
	uint64_t* notsmaller = *notsmallerp;
	uint64_t* notsmallerweighted = *notsmallerweightedp;
	if(Ti > nc)
		Ti = nc;


	uint64_t div = Ti;
	uint64_t res = notsmallerweighted[Ti] - ((Ti - 1)*notsmaller[Ti]);
	uint64_t all_tiles = (nr-Ti+1);
	all_tiles *= nc;

	bool overflow = (notsmallerweighted[Ti] < ((Ti - 1)*notsmaller[Ti])) || (all_tiles < res);

	if (overflow)
		res = 0;
	else
		res = all_tiles - res;


	//res /= Ti;
	double ratio = ((double) (res)/all_tiles);  //((nr-1)/Ti+1)*nc - res;
	res = ((nr-1)/Ti+1);
	res *= nc;
	res *= ratio;


    return res;
}

int pick_tile(int &Ti, int &Tk , int Nk , int cache, int nr, int nc, int ne)
{

	uint64_t lowest = -1;
	//for(int k = 32; k <= 128 ; k*=2)
	{
		for (int i = 2; i< 2*nc /*CACHE/k */ && cache/i > 0; i*=2)
		{
			uint64_t k = MIN(Nk,cache/i);
			k = k - (k %32);
			if (k <= 0)
				continue;
			// uint64_t test =  active_cols(i,true);
			uint64_t test =  active_cols(i, nr, nc, ne);

			// uint64_t model = ((uint64_t )ne )* ((uint64_t) Nk) / k *5 / 2; // sparse part

			// uint64_t model = ((uint64_t )ne )* ((uint64_t) Nk) / k *5 / 2; // sparse part
			uint64_t model = ((uint64_t )ne )* ((uint64_t) Nk) / k * 3; // sparse part
			model += test * ((uint64_t) Nk) ; // input part
			model += 2* ((uint64_t) nr)*((uint64_t) Nk); // output part
//			cout<<i<<"\t"<<test<<'\t'<<model/8<<endl;
			if (lowest == -1 || model < lowest)
			{
				lowest = model;
				Ti = i;
				Tk = k;
			}

		}
	}

	return 0;
}

int pick_tile_sddmm(int &Ti, int &Tk , int Nk , int cache, int nr, int nc, int ne)
{
	uint64_t lowest = -1;
	//for(int k = 32; k <= 128 ; k*=2)
	{
		for (int i = 2; i< 2*nc /*CACHE/k */ && cache/i > 0; i*=2)
		{
			uint64_t k = MIN(Nk,cache/i);
			k = k - (k %32);
			if (k <= 0)
				continue;
			uint64_t test =  active_cols(i, nr, nc, ne);

			uint64_t model = ((uint64_t )ne )* ((uint64_t) Nk) / k *7 / 2; // sparse part
			model += test * ((uint64_t) Nk) ; // input part
			model += 2* ((uint64_t) nr)*((uint64_t) Nk); // output part

			if (lowest == -1 || model < lowest)
			{
				lowest = model;
				Ti = i;
				Tk = k;
			}
		}
	}

	return 0;
}


#endif

