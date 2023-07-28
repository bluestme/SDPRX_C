#ifndef HELPER_LD_H
#define HELPER_LD_H
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <map>
#include <unordered_map>
#include <math.h>
#include <algorithm>
#include "gsl/gsl_statistics.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_vector.h"
#include "SDPRX_io.h"


std::vector<size_t> findMatchedId(const SnpInfo& S1, const SnpInfo& S2,\
                     const std::vector<size_t>& idx1, const std::vector<size_t>& idx2);

void scaling_LD(gsl_matrix *snp);

size_t* find_ld(gsl_matrix *snp1, gsl_matrix *snp2);

std::vector<size_t> findMatchedId(const SnpInfo& S1, const SnpInfo& S2, \
                    const std::vector<size_t>& idx1, const std::vector<size_t>& idx2) 
{
    std::unordered_map<std::string, size_t> idToIndexMap;

    // Create a mapping of id values to their indexes in S2
    for (size_t i = 0; i < idx1.size(); i++) 
	{
        idToIndexMap[S1.id[idx1[i]]] = idx1[i];
    }

    std::vector<size_t> matchingIndexes;

    // Find matching indexes in S2 for each id in S1
    for (size_t j = 0; j < idx2.size(); j++) 
    {
        auto it = idToIndexMap.find(S2.id[idx2[j]]);
        if (it != idToIndexMap.end()) 
        {
            matchingIndexes.push_back(it->second);
        }
    }
    return matchingIndexes;
}

void scaling_LD(gsl_matrix *snp)
{
    size_t nrow = snp -> size1;
    size_t ncol = snp -> size2;
    for (size_t i = 0; i < nrow; i++)
    {
        double *geno = new double[ncol];
        for (size_t j = 0; j < ncol; j++) 
        {
            geno[j] = gsl_matrix_get(snp, i, j);
        }
        double mean = gsl_stats_mean(geno, 1, ncol);
        double sd = gsl_stats_sd_m(geno, 1, ncol, mean)*sqrt(ncol - 1)/sqrt(ncol);
        if (sd == 0) sd = 1;
        
        for (size_t j = 0; j < ncol; j++) 
        {
	        gsl_matrix_set(snp, i, j, (geno[j]-mean)/sd);
	    }
        delete[] geno;
    }
}

size_t* find_ld(gsl_matrix *snp1, gsl_matrix *snp2, double r2)
{
    double cor = 0, cor1 = 0, cor2 = 0; 
    size_t nrow1 = snp1 -> size1, ncol1 = snp1 -> size2;
    size_t nrow2 = snp2 -> size1, ncol2 = snp2 -> size2;
    size_t *max_list = new size_t[nrow1];
    gsl_vector_view snp11, snp12, snp21, snp22;
    for (size_t i = 0; i < nrow1; i++) 
    {
        size_t left, right;
        if (i < 300) left = 0;
        else left = i - 300;
        max_list[i] = i;
	    
	    snp11 = gsl_matrix_row(snp1, i);
        snp21 = gsl_matrix_row(snp2, i);

	    for (size_t j = left; j < i + 300; j++) 
        {
            if (j >= nrow1) continue;
	        snp12 = gsl_matrix_row(snp1, j);
            snp22 = gsl_matrix_row(snp2, j);
	        gsl_blas_ddot(&snp11.vector, &snp12.vector, &cor1);
            gsl_blas_ddot(&snp21.vector, &snp22.vector, &cor2);
	        cor1 /= ncol1;
            cor2 /= ncol2;
            cor = std::max(cor1, cor2);

	        if (cor*cor > r2) 
            {
                max_list[i] = j;
	        }
    	}

	    if (i == 0) continue;

	    if (max_list[i] < max_list[i - 1]) 
        {
	        max_list[i] = max_list[i - 1];
	    }
    }
    return max_list;
}

#endif