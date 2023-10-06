#ifndef HELPER_H
#define HELPER_H
#include "parse_gen.h"
#include <algorithm>
#include <math.h>
#include "gsl/gsl_cblas.h"
#include "gsl/gsl_randist.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_eigen.h"
#include <thread>
#include <chrono>
#include <vector>
#include <fstream>
#include <numeric>
#include <unordered_set>
#include <string>


using std::ifstream; using std::string;
using std::vector; using std::unordered_map; 
using std::cout; using std::endl;
using std::pair; using std::find;

void get_shared_idx(std::vector<std::vector<size_t>>& shared_idx, size_t bd_size, \
                    mcmc_data &dat1, mcmc_data &dat2, std::vector<size_t> &pop_idx)
{
    std::unordered_set<std::string> set(dat2.id.begin(), dat2.id.end());
    for (size_t i = 0; i < bd_size; i++)
    {
        std::vector<size_t> tmplist;
        for (size_t j = dat1.boundary[i].first; j < dat1.boundary[i].second; j++)
        {
            string id = dat1.id[j];
            auto it = set.find(id);
            if (it != set.end()) 
            {
                tmplist.push_back(j - dat1.boundary[i].first);
	        }
	        else
	        {
	            pop_idx.push_back(j);
	        }
        }
        shared_idx.push_back(tmplist);
    }
}

#endif