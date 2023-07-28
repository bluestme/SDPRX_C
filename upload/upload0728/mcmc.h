#ifndef MCMC_H
#define MCMC_H

#include <iostream>
#include <vector>
#include "parse_gen.h"
#include "gsl/gsl_vector.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_rng.h"
#include <math.h>
#include <unordered_map>
#include <string>
#include "gsl/gsl_cblas.h"
#include "gsl/gsl_randist.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_eigen.h"

#define square(x) ((x)*(x))
std::vector<double> selectElements(const std::vector<double>& inputVector, const std::vector<int>& indices);

typedef struct{
	double a0k;
	double b0k;
	double a0;
	double b0;
} hyper;

typedef struct {
    std::vector<gsl_matrix*> A;
    std::vector<gsl_matrix*> B;
    std::vector<gsl_matrix*> L;
    std::vector<gsl_vector*> beta_mrg;
    std::vector<gsl_vector*> calc_b_tmp;
    std::vector<double> num;
    std::vector<double> denom;
} ldmat_data;

class MCMC_state {
    public:
	double alpha[4];
	double p_pop[4];
	double eta;
	double h21;
	double h22;
	double rho;
	int N1;
	int N2;
	size_t bd_size;

	int population[4];
	int num_cluster;
	int *cluster_id;

	gsl_vector *beta1;
	gsl_vector *beta2;
	gsl_vector *b1;
	gsl_vector *b2;
	hyper para;
	std::vector<int> cls_assgn1;
	std::vector<int> cls_assgn2;
	std::vector<std::vector<double> > V;
	std::vector<double> p;
	std::vector<std::vector<double> > p_cluster;
	std::vector<double> log_p;
	std::vector<double> cluster_var;
	std::vector<unsigned> suff_stats;
	std::vector<double> sumsq;
	std::vector<std::pair<size_t, size_t> > boundary1;
	std::vector<std::pair<size_t, size_t> > boundary2;
	MCMC_state(size_t num_snp1, size_t num_snp2, int *max_cluster, \
		double a0, double b0, double sz1, double sz2, mcmc_data &dat1, mcmc_data&dat2, double rho0) 
	{
		num_cluster = 0;
        for (size_t j = 0; j < 4; j++)
        {
            num_cluster = num_cluster + max_cluster[j];
        }
		cluster_id = new int[num_cluster];
		for (size_t j = 0; j < num_cluster; j++)
		{
			cluster_id[j] = j;
		}
		bd_size = dat1.boundary.size();
		for (size_t j = 0; j < bd_size; j++)
		{
			boundary1.push_back(std::make_pair(dat1.boundary[j].first, dat1.boundary[j].second));
		}
		for (size_t j = 0; j < bd_size; j++)
		{
			boundary2.push_back(std::make_pair(dat2.boundary[j].first, dat2.boundary[j].second));
		}

		N1 = sz1;
		N2 = sz2;
		para.a0k = a0;
		para.b0k = b0;
		para.a0 = 0.1;
		para.b0 = 0.1;
		n_snp1 = num_snp1;
		n_snp2 = num_snp2;
		rho = rho0;

		for (size_t j = 0; j < 4; j++)
		{
			alpha[j] = 1.0;
			p_pop[j] = 0.25;
		}
	    eta = 1; h21 = 0; h22 = 0;
	    beta1 = gsl_vector_calloc(num_snp1);
		beta2 = gsl_vector_calloc(num_snp2);
	    b1 = gsl_vector_calloc(num_snp1);
		b2 = gsl_vector_calloc(num_snp2);
	    p.assign(num_cluster, 1.0/num_cluster);
	    log_p.assign(num_cluster, 0);
		population[0] = 1;
		population[1] = max_cluster[1] + 1;
		population[2] = max_cluster[2] + max_cluster[1] + 1;
		population[3] = num_cluster;


	    for (size_t i=0; i<num_cluster; i++) 
		{
			log_p[i] = logf(p[i] + 1e-40);
	    }

	    cluster_var.assign(num_cluster, 0.0);
	    suff_stats.assign(num_cluster, 0);
	    sumsq.assign(num_cluster, 0.0);
		cls_assgn1.assign(num_snp1, 0);
		cls_assgn2.assign(num_snp1, 0);
		r = gsl_rng_alloc(gsl_rng_default);
		for (size_t j = 0; j < 4; j++)
        {
            std::vector<double> tmplist;
			if (j == 0)
			{
				tmplist.assign(1, 0);
				V.push_back(tmplist);
			}
			else
			{
				tmplist.assign(1000, 0);
				V.push_back(tmplist);
			}
        }
		for (size_t j = 0; j < 4; j++)
        {
            std::vector<double> tmplist;
			if (j == 0)
			{
				tmplist.assign(1, 1);
				p_cluster.push_back(tmplist);
			}
			else
			{
				tmplist.assign(1000, 0.001);
				p_cluster.push_back(tmplist);
			}
        }
	    
	    for (size_t i = 0; i < num_snp1; i++) 
		{
			cls_assgn1[i] = gsl_rng_uniform_int(r, num_cluster);
		}
		for (size_t i = 0; i < num_snp2; i++) 
		{
			cls_assgn2[i] = gsl_rng_uniform_int(r, num_cluster);
	    }
	}

	~MCMC_state() {
	    gsl_vector_free(beta1);
		gsl_vector_free(beta2);
	    gsl_vector_free(b1);
		gsl_vector_free(b2);
	    gsl_rng_free(r);
		delete[] cluster_id;

	}

	void sample_sigma2();
	void calc_b(size_t j, const mcmc_data &dat1, const mcmc_data &dat2, const ldmat_data &ldmat_dat1,\
						const ldmat_data &ldmat_dat2);

	void sample_assignment(size_t j, const mcmc_data &dat1, const mcmc_data &dat2,\
									 const ldmat_data &ldmat_dat1, const ldmat_data &ldmat_dat2,\
									 std::vector<size_t>& shared_idx1, std::vector<size_t>& shared_idx2,\
									 std::vector<size_t>& pop_idx1, std::vector<size_t>& pop_idx2);
	
	void update_suffstats(std::vector<size_t> &pop_idx1, std::vector<size_t> &pop_idx2,\
								  const std::vector<size_t>* shared_idx1, const std::vector<size_t>* shared_idx2);
	void sample_V();
	void update_p();
	void sample_alpha();
	void sample_p_cluster(std::vector<size_t>& idx_pop1, std::vector<size_t>& idx_pop2);
	void sample_beta(size_t j, const mcmc_data &dat1, const mcmc_data &dat2, \
	        ldmat_data &ldmat_dat1, ldmat_data &ldmat_dat2);
	void compute_h2(const mcmc_data &dat1, const mcmc_data &dat2);
	void sample_eta(const ldmat_data &ldmat_dat);

    private:
	size_t M, n_snp1, n_snp2;
	gsl_rng *r;
	void sample_indiv (float** prob, float** tmp, const std::vector<float>& Bjj, \
								const std::vector<float>& bj, size_t pop);
	void sample_cor (const std::vector<float>& B1jj, const std::vector<float>& B2jj, \
								const std::vector<float>& b1j, const std::vector<float>& b2j,\
								float** prob, float** tmp);
	std::vector<int> shared_assignment(const mcmc_data &dat1, const mcmc_data &dat2,\
									const ldmat_data &ldmat_dat1, const ldmat_data &ldmat_dat2, size_t j,\
									std::vector<size_t>& shared_idx1, std::vector<size_t>& shared_idx2);
	std::vector<int> indiv_assignment(const mcmc_data &dat, const ldmat_data &ldmat_dat, \
									size_t j, std::vector<size_t>& idx_pop, int pop);
	
	gsl_vector* sample_MVN_single(const std::vector<size_t>& causal_list, size_t j,\
							ldmat_data &ldmat_dat, size_t start_i, int pop);
};

class MCMC_samples 
{
    public:
	gsl_vector *beta1;
	gsl_vector *beta2;
	double h21;
	double h22;

	MCMC_samples(size_t num_snps1, size_t num_snps2) 
	{
	    beta1 = gsl_vector_calloc(num_snps1);
		beta2 = gsl_vector_calloc(num_snps2);
	    h21 = 0;
		h22 = 0;
	}

	~MCMC_samples() 
	{
	    gsl_vector_free(beta1);
		gsl_vector_free(beta2);
	}
};

void mcmc(const std::string &ref_path, const std::string &ss_path1, const std::string &ss_path2, \
		const std::string &valid_path, const std::string &ldmat_path1, \
		const std::string &ldmat_path2, unsigned N1, unsigned N2, \
		const std::string &out_path, double a, double rho, double a0k, double b0k, \
		int iter, int burn, int thin, unsigned n_threads, int opt_llk, double c1, double c2);


# endif