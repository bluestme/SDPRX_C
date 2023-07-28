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

void get_shared_idx(std::vector<size_t> *shared_idx, size_t bd_size, mcmc_data &dat1, mcmc_data &dat2)
{
    std::unordered_set<std::string> set(dat2.id.begin(), dat2.id.end());
    for (size_t i = 0; i < bd_size; i++)
    {
        for (size_t j = dat1.boundary[i].first; j < dat1.boundary[i].second; j++)
        {
            string id = dat1.id[j];
            auto it = set.find(id);
            if (it != set.end()) 
            {
                shared_idx[i].push_back(j - dat1.boundary[i].first);
	        }
        }
    }
}
// Can be optimized with hash table

gsl_matrix* extractSubmatrix(const gsl_matrix* matrix, const std::vector<size_t>& Index) 
{
    // Get the dimensions of the submatrix
    size_t numRows = Index.size();
    size_t numCols = Index.size();

    // Allocate memory for the submatrix
    gsl_matrix* submatrix = gsl_matrix_alloc(numRows, numCols);

    // Extract the submatrix
    for (size_t i = 0; i < numRows; ++i) {
        for (size_t j = 0; j < numCols; ++j) {
            size_t row = Index[i];
            size_t col = Index[j];
            double value = gsl_matrix_get(matrix, row, col);
            gsl_matrix_set(submatrix, i, j, value);
        }
    }
    return submatrix;
}

void get_pop_idx(std::vector<size_t> &pop_idx, mcmc_data &dat1, mcmc_data &dat2)
{
	std::unordered_set<std::string> set(dat2.id.begin(), dat2.id.end());
    for (size_t i = 0; i < dat1.id.size(); i++)
    {
        string id = dat1.id[i];
        auto it = set.find(id);
        if (it == set.end()) 
        {
            pop_idx.push_back(i);
	    }
    }
}

std::vector<double> selectElements(const std::vector<double>& inputVector, const std::vector<int>& indices) 
{
    std::vector<double> selectedElements;

    for (int index : indices) 
	{
        selectedElements.push_back(inputVector[index]);
	}
    return selectedElements;
}

#endif