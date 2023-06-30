#ifndef STATE_H
#define STATE_H

#include <iostream>
#include <vector>
#include "parse_gen.h"
#include "gsl/gsl_vector.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_rng.h"
#include <math.h>

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
	double h12;
	int N1;
	int N2;


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
	MCMC_state(size_t num_snp1, size_t num_snp2, int *max_cluster, \
		double a0, double b0, double sz1, double sz2) 
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
		N1 = sz1;
		N2 = sz2;
		para.a0k = a0;
		para.b0k = b0;
		para.a0 = 0.1;
		para.b0 = 0.1;
		
		for (size_t j = 0; j < 4; j++)
		{
			alpha[j] = 1.0;
			p_pop[j] = 0.25;
		}
	    eta = 1; h12 = 0; h21 = 0;
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
	    
	    for (size_t i=0; i<num_snp1; i++) 
		{
			cls_assgn1[i] = gsl_rng_uniform_int(r, num_cluster);
		}
		for (size_t i=0; i<num_snp2; i++) 
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
	void calc_b(size_t j, const mcmc_data &dat, const ldmat_data &ldmat_dat);
	void sample_assignment(size_t j, const mcmc_data &dat, \
		        const ldmat_data &ldmat_dat);
	void update_suffstats();
	void sample_V();
	void update_p();
	void sample_alpha();
	void sample_beta(size_t j, const mcmc_data &dat, \
		       ldmat_data &ldmat_dat);
	void compute_h2(const mcmc_data &dat);
	void sample_eta(const ldmat_data &ldmat_dat);

    private:
	size_t M, n_snp;
	gsl_rng *r;
};

class MCMC_samples 
{
    public:
	gsl_vector *beta1;
	gsl_vector *beta2;
	double h12;
	double h21;

	MCMC_samples(size_t num_snps1, size_t num_snps2) 
	{
	    beta1 = gsl_vector_calloc(num_snps1);
		beta2 = gsl_vector_calloc(num_snps2);
	    h12 = 0;
		h21 = 0;
	}

	~MCMC_samples() 
	{
	    gsl_vector_free(beta1);
		gsl_vector_free(beta2);
	}
};

#endif