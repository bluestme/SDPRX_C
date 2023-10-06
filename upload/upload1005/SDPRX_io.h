#ifndef SDPRXIO_H
#define SDPRXIO_H

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

using std::cout; using std::endl; using std::ifstream;
using std::vector; using std::string;
using std::map;

typedef struct {
    std::vector<std::string> id;
    std::vector<std::string> A1;
    std::vector<std::string> A2;
    std::vector<size_t> Pos;
    std::map<size_t, size_t> chr_idx;
} SnpInfo;



size_t get_nsamples(const std::string &fam_path); // get the number of samples

void read_bim(const std::string &bim_path, SnpInfo *snpinfo);

void read_bed(gsl_matrix *snp, const std::string &bed_path, const size_t n_samples, size_t left, size_t right, \
				const std::vector<size_t>& idxio);

#endif