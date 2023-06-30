#ifndef PARSE_GEN_H
#define PARSE_GEN_H

#include <iostream>
#include <string>
#include <vector>
#include "gsl/gsl_matrix.h"
#include <algorithm>
#include <unordered_map>
#include <stdexcept>
#include "gsl/gsl_cdf.h"
#include "gsl/gsl_eigen.h"
#include "gsl/gsl_blas.h"
#include <math.h>
#include <fstream>
#include <sstream>

typedef struct {
    std::string A1;
    std::string A2;
    bool include_ref;
    bool include_ss;
    double beta;
    int array;
    double sz;
} CoordInfo;

typedef struct {
    std::vector<std::string> id;
    std::vector<std::string> A1;
    std::vector<std::string> A2;
    std::vector<double> beta_mrg;
    std::vector<std::pair<size_t, size_t> > boundary;
    std::vector<gsl_matrix *> ref_ld_mat;
    std::vector<double> sz;
    std::vector<int> array;
} mcmc_data;

void coord(const std::string &ref_path, const std::string &ss_path, \
	const std::string &valid_path, const std::string &ldmat_path, \
	mcmc_data &dat, unsigned sz, int opt_llk);

using std::ifstream; using std::string;
using std::vector; using std::unordered_map; 
using std::cout; using std::endl;
using std::pair; using std::find;

double sign(double x) {
    if (x > 0) return 1.0;
    else if (x < 0) return -1.0;
    else return 0;
}

void parse_ref(const string &ref_path, \
	unordered_map<string, CoordInfo*> &ref_dict, \
	vector<pair<size_t, size_t> > &boundary, \
	vector<string> &SNP) {
    ifstream infile(ref_path.c_str());
    string id, A1, A2, line;

    if (!infile) {
	throw std::runtime_error("Error: cannot open ref "
		"snpInfo file: " + ref_path);
    }

    int n = 0, section = 1;
    size_t left, right;
   
    // skip the header
    getline(infile, line);
    while (getline(infile, line)) {
	if (line == "") {
	    section++;
	    cout << "Readed " << n << " LD blocks." << endl;
	    n = 0;
	    getline(infile, line); // skip the header
	    continue;
	}

	if (section == 1) {
	    std::istringstream my_stream(line);
	    my_stream >> left >> right;
	    boundary.push_back(std::make_pair(left, right));
	    n++;
	}
	else {
	    std::istringstream my_stream(line);
	    my_stream >> id >> A1 >> A2;
	    SNP.push_back(id);
	    CoordInfo *ref_info = new CoordInfo;
	    ref_info->A1 = A1; ref_info->A2 = A2;
	    ref_info->include_ref = false; 
	    ref_info->include_ss = false; // They will be true after adding the ref and ss files
	    ref_info->beta = 0; // Beta initialized as 0
	    if (!ref_dict.insert(pair<string, \
			CoordInfo*>(id, ref_info)).second) {
		throw std::runtime_error("Error: duplicate SNP found "
			"in ref snpInfo file: " + id);
	    }
	    n++;
	}
    }

    cout << "Readed " << n << " SNPs from reference panel" << endl;

    infile.close();
}

void parse_valid(const string &valid_path, \
	unordered_map<string, CoordInfo*> &ref_dict) {
    ifstream infile(valid_path.c_str());
    if (!infile) {
	throw std::runtime_error("Error: cannot open "
		"ref snpInfo file: " + valid_path);
    }

    string id, A1, A2, header;
    float genPos; unsigned chr, phyPos;
    int n = 0;
    unordered_map<string, CoordInfo*>::iterator idx;

    while (infile >> chr >> id >> genPos >> phyPos >> A1 >> A2) {
	idx = ref_dict.find(id);
	if (idx != ref_dict.end()) {
	    idx->second->include_ref = true;
	    n++;
	} // Set include_ref as true if the individual is not in the valid set
    }
    cout << n << " common SNPs between reference "
	"and validation datasets." << endl;
    infile.close();
}

void parse_ss(const string &ss_path, unordered_map<string, \
	CoordInfo*> &ref_dict, unsigned sz, int opt_llk) {
    ifstream infile(ss_path.c_str());
    if (!infile) {
	throw std::runtime_error("Error: cannot open "
		"summary statistics: " + ss_path);
    }

    string id, A1, A2, line;
    std::stringstream ss;
    double beta = 0, pval = 0, N = 1, Z = 0;
    int n = 0, array = 0;
    unordered_map<string, CoordInfo*>::iterator idx;

    vector<string> tokens;
    size_t SNP_idx, A1_idx, A2_idx;
    int beta_idx = -1, pval_idx = -1, array_idx = -1; 
    int sz_idx = -1, Z_idx = -1;

    int n_flip = 0, n_bad = 0, nline = 0;

    while (getline(infile, line, '\n')) {
	ss.str(line);
	while (ss >> line) {
	    tokens.push_back(line);
	}
	if (nline == 0) {
	    // find corresponding fields in the header
	    vector<string>::iterator token_idx; // iterate through the index

	    token_idx = find(tokens.begin(), tokens.end(), "SNP");
	    if (token_idx != tokens.end()) {
		SNP_idx = token_idx - tokens.begin();
	    }
	    else {
		throw std::runtime_error("Error: cannot find SNP column.");
	    }

	    token_idx = find(tokens.begin(), tokens.end(), "A1");
	    if (token_idx != tokens.end()) {
		A1_idx = token_idx - tokens.begin();
	    }
	    else {
		throw std::runtime_error("Error: cannot find A1 column.");
	    }

	    token_idx = find(tokens.begin(), tokens.end(), "A2");
	    if (token_idx != tokens.end()) {
		A2_idx = token_idx - tokens.begin();
	    }
	    else {
		throw std::runtime_error("Error: cannot find A2 column.");
	    }
	    
	    token_idx = find(tokens.begin(), tokens.end(), "BETA");
	    if (token_idx != tokens.end()) {
		beta_idx = token_idx - tokens.begin();
	    }

	    token_idx = find(tokens.begin(), tokens.end(), "P");
	    if (token_idx != tokens.end()) {
		pval_idx = token_idx - tokens.begin();
	    }
	    
	    token_idx = find(tokens.begin(), tokens.end(), "N");
	    if (token_idx != tokens.end()) {
		sz_idx = token_idx - tokens.begin();
	    }

	    token_idx = find(tokens.begin(), tokens.end(), "Z");
	    if (token_idx != tokens.end()) {
		Z_idx = token_idx - tokens.begin();
	    }

	    if (opt_llk == 2) {
		token_idx = find(tokens.begin(), tokens.end(), "ARRAY");
		if (token_idx != tokens.end()) {
		    array_idx = token_idx - tokens.begin();
		}
		else {
		    throw std::runtime_error("Error: cannot find ARRAY column.");
		}
	    }
		// return the position of each element (including Z, N, A1, A2) in a line
	}
	else {
	    // parse fields
	    id = tokens[SNP_idx]; 
	    A1 = tokens[A1_idx]; A2  = tokens[A2_idx];
	    if (Z_idx > 0) {
		Z = std::stod(tokens[Z_idx]);
	    } // should return -1 in the test one
	    else {
		if (pval_idx < 0 || beta_idx < 0) {
		    throw std::runtime_error("Error: cannot find BETA or P column.");
		}
		pval = std::stod(tokens[pval_idx]); 
		beta = std::stod(tokens[beta_idx]);
	    }
	    if (sz_idx > 0) {
		N = std::stod(tokens[sz_idx]); 
		sz = N;
	    } // should return -1 in the test one
	    if (opt_llk == 2) {
		array = std::stoi(tokens[array_idx]);
	    }

	    // coordination
	    idx = ref_dict.find(id);
	    if (pval <= 1e-308) {
		pval = 1e-308;
	    } // The P-value is not too small
	    if (idx != ref_dict.end() && idx->second->include_ref) {
		// It has the overlap in the valid and reference files
		if (A1 == idx->second->A1 && A2 == idx->second->A2) {
		    idx->second->include_ss = true;
		    if (Z_idx > 0) {
			idx->second->beta = Z/sqrt(sz);
		    }
		    else {
			idx->second->beta = 1.0*sign(beta)* \
			fabs(gsl_cdf_ugaussian_Pinv(pval/2.0))/sqrt(sz); // report it to Geyu
		    }
		    // Added for evaluating likelihood involving Ns
		    if (opt_llk == 2) {
			idx->second->array = array;
			idx->second->sz = sz;
		    }
		    n++;
		}
		else if (A1 == idx->second->A2 && A2 == idx->second->A1) {
		    idx->second->include_ss = true;
		    if (Z_idx > 0) {
			idx->second->beta = -1.0*Z/sqrt(sz);
		    }
		    else {
			idx->second->beta = -1.0*sign(beta)* \
			fabs(gsl_cdf_ugaussian_Pinv(pval/2.0))/sqrt(sz);
		    }
		    // Added for evaluating likelihood involving Ns
		    if (opt_llk == 2) {
			idx->second->array = array;
			idx->second->sz = sz;
		    }
		    n++;
		    n_flip++;
		} // update beta and make the paired choice
		else {
		    n_bad++;
		}
	    }
	}
	nline++;
	tokens.clear();
	ss.clear();
    }

    cout << n_flip << " SNPs have flipped alleles between summary statistics and " 
	<< "reference panel." << endl;
    cout << n_bad << " SNPs removed due to mismatch of allels between " 
	<< "summary statistics and reference panel." << endl;
    cout << n << " common SNPs among reference, validation "
	"and gwas summary statistics." << endl;
    infile.close();
}

void parse_ld_mat(const string &ldmat_path, unordered_map<string, CoordInfo*> &ref_dict, \
	const vector<pair<size_t, size_t > > &boundary, \
	const vector<string> &SNP, mcmc_data &dat, int opt_llk) 
	{
    
    FILE *fp;
    fp = fopen(ldmat_path.c_str(), "rb");

    if (!fp) {
	throw std::runtime_error("Error: cannot open LD matrix file: " + ldmat_path);
    }

    unordered_map<string, CoordInfo*>::iterator idx; //very typically used trick to iterate through a vector
    vector<size_t> snp_idx;
    size_t left = 0, right = 0;
    for (size_t i=0; i<boundary.size(); i++) {
	// Iterate through each boundary unit
	snp_idx.clear();
	for (size_t j=boundary[i].first; j<boundary[i].second; j++) {
	    idx = ref_dict.find(SNP[j-boundary[0].first]); // It returns the pointer to each SNP notes in the set
	    if (idx->second->include_ss) {// How about ref, since include ss is a part of include ref
		dat.id.push_back(SNP[j-boundary[0].first]); // push back to add something into a vector
		dat.A1.push_back(idx->second->A1);
		dat.A2.push_back(idx->second->A2);
		dat.beta_mrg.push_back(idx->second->beta);
		snp_idx.push_back(j-boundary[i].first);
		// Added for evaluating llk involving Ns
		if (opt_llk == 2) {
		    dat.sz.push_back(idx->second->sz);
		    dat.array.push_back(idx->second->array);
		}
		right++;
	    }
	}// We are actually generating each small matrix for each boundary limit
	gsl_matrix *tmp_mat = gsl_matrix_alloc(boundary[i].second-boundary[i].first, \
		                boundary[i].second-boundary[i].first);
	gsl_matrix_fread(fp, tmp_mat); // read in the matrix

	if (left == right) {
	    gsl_matrix_free(tmp_mat);
	    continue; 
	}
	dat.boundary.push_back(std::make_pair(left,right));
	gsl_matrix *tmp_mat_sub = gsl_matrix_alloc(right-left, right-left);
	// left and right updates as the standard for dataset but not
	// copy rows from original matrix to second matrix with correct SNPs 
	for (size_t j=0; j<snp_idx.size(); j++) {
	    for (size_t k=0; k<snp_idx.size(); k++) {
		double tmp = gsl_matrix_get(tmp_mat, snp_idx[j], snp_idx[k]);
		gsl_matrix_set(tmp_mat_sub, j, k, tmp);
	    }
	}

	dat.ref_ld_mat.push_back(tmp_mat_sub);
	left = right;
	gsl_matrix_free(tmp_mat);
    }
    fclose(fp);
}
    

void coord(const string &ref_path, const string &ss_path, \
	const string &valid_path, const string &ldmat_path, \
	mcmc_data &dat, unsigned sz, int opt_llk) 
{
    unordered_map<string, CoordInfo*> ref_dict;
    vector<pair<size_t, size_t > > boundary;
    vector<string> SNP;
    unordered_map<string, CoordInfo*>::iterator it;

    parse_ref(ref_path, ref_dict, boundary, SNP);
    
    if (!valid_path.empty()) {
	parse_valid(valid_path, ref_dict);
    }
    else {
	for (it=ref_dict.begin(); it != ref_dict.end(); it++) {
	    it->second->include_ref = true;
	}
    }

    parse_ss(ss_path, ref_dict, sz, opt_llk);    
    parse_ld_mat(ldmat_path, \
	    ref_dict, boundary, SNP, dat, opt_llk);
    for (it=ref_dict.begin(); it != ref_dict.end(); it++) {
	 delete(it->second);
    }    
}



#endif