#ifndef STATE_H
#define STATE_H

#include <iostream>
#include <vector>
#include "parse_gen.h"
#include "gsl/gsl_vector.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_rng.h"
#include <math.h>
#include <unordered_map>
#include "gsl/gsl_cblas.h"
#include "gsl/gsl_randist.h"
#include "gsl/gsl_linalg.h"
#include "gsl/gsl_eigen.h"
#include "helper.h"

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

void MCMC_state::update_suffstats(std::vector<size_t> &pop_idx1, std::vector<size_t> &pop_idx2, \
								  const std::vector<size_t>* shared_idx1, const std::vector<size_t>* shared_idx2) 
{
    std::fill(suff_stats.begin(), suff_stats.end(), 0.0);
    std::fill(sumsq.begin(), sumsq.end(), 0.0);
	for (size_t i = 0; i < pop_idx1.size(); i++)
	{
		suff_stats[cls_assgn1[pop_idx1[i]]]++;
		double tmp = gsl_vector_get(beta1, pop_idx1[i]);
		sumsq[cls_assgn1[pop_idx1[i]]] += square(tmp);
	}
	for (size_t i = 0; i < pop_idx2.size(); i++)
	{
		suff_stats[cls_assgn2[pop_idx2[i]]]++;
		double tmp = gsl_vector_get(beta2, pop_idx2[i]);
		sumsq[cls_assgn2[pop_idx2[i]]] += square(tmp);
	}
	for (size_t i = 0; i < bd_size; i++)
	{
		size_t start1 = boundary1[i].first;
		size_t start2 = boundary2[i].first;
		for (size_t j = 0; j < shared_idx1[i].size(); j++)
		{
			double tmp1 = gsl_vector_get(beta1, shared_idx1[i][j] + start1);
			double tmp2 = gsl_vector_get(beta2, shared_idx2[i][j] + start2);
			suff_stats[cls_assgn2[shared_idx2[i][j] + start2]]++;
			sumsq[cls_assgn2[shared_idx2[i][j] + start2]] += square(tmp1) + square(tmp2) - 2*rho*tmp1*tmp2;
		}
	}

}

void MCMC_state::sample_sigma2() 
{
    for (size_t i = 1; i < num_cluster; i++) 
	{
		double a = suff_stats[i] / 2.0 + para.a0k;
		double b = 1.0 / (sumsq[i] / 2.0 + para.b0k);
		if (i > population[2])
		{
			b = 1.0 / (sumsq[i] / (2.0*(1 - square(rho))) + para.b0k);
		}
		cluster_var[i] = 1.0/gsl_ran_gamma(r, a, b);
		if (isinf(cluster_var[i])) 
		{
	    	cluster_var[i] = 1e5;
	    	std::cerr << "Cluster variance is infintie." << std::endl;
		}
		else if (cluster_var[i] == 0) 
		{
	    	cluster_var[i] = 1e-10;
	    	std::cerr << "Cluster variance is zero." << std::endl;
		}
    }
}

void MCMC_state::calc_b(size_t j, const mcmc_data &dat1, const mcmc_data &dat2, const ldmat_data &ldmat_dat1,\
						const ldmat_data &ldmat_dat2) 
{
    size_t start_i1 = dat1.boundary[j].first;
    size_t end_i1 = dat1.boundary[j].second;
	size_t start_i2 = dat2.boundary[j].first;
    size_t end_i2 = dat2.boundary[j].second;
    gsl_vector_view b_j1 = gsl_vector_subvector(b1, start_i1, end_i1 - start_i1);
    gsl_vector_view beta_j1 = gsl_vector_subvector(beta1, start_i1, end_i1 - start_i1);
	
	gsl_vector_view b_j2 = gsl_vector_subvector(b2, start_i2, end_i2 - start_i2);
    gsl_vector_view beta_j2 = gsl_vector_subvector(beta2, start_i2, end_i2 - start_i2);

    gsl_vector_const_view diag1 = gsl_matrix_const_diagonal(ldmat_dat1.B[j]);
	gsl_vector_const_view diag2 = gsl_matrix_const_diagonal(ldmat_dat2.B[j]);
    
    // diag(B) \times beta
    gsl_vector_memcpy(&b_j1.vector, &beta_j1.vector);
    gsl_vector_mul(&b_j1.vector, &diag1.vector);

    // eta^2 * (diag(B) \times beta) - eta^2 * B beta
    gsl_blas_dsymv(CblasUpper, -eta*eta, ldmat_dat1.B[j], &beta_j1.vector, \
	    eta*eta, &b_j1.vector);
	// Why do we specify this one? CblasUpper, print it out!

    // eta^2 * (diag(B) \times beta) - eta^2 * B beta + eta * A^T beta_mrg
    gsl_blas_daxpy(eta, ldmat_dat1.calc_b_tmp[j], &b_j1.vector);
	
	// the pointer is still pointing to the original b_j selection

	gsl_vector_memcpy(&b_j2.vector, &beta_j2.vector);
    gsl_vector_mul(&b_j2.vector, &diag2.vector);

    // eta^2 * (diag(B) \times beta) - eta^2 * B beta
    gsl_blas_dsymv(CblasUpper, -eta*eta, ldmat_dat2.B[j], &beta_j2.vector, \
	    eta*eta, &b_j2.vector);

    // eta^2 * (diag(B) \times beta) - eta^2 * B beta + eta * A^T beta_mrg
    gsl_blas_daxpy(eta, ldmat_dat2.calc_b_tmp[j], &b_j2.vector);
}

void MCMC_state::sample_assignment(size_t j, const mcmc_data &dat1, const mcmc_data &dat2,\
									 const ldmat_data &ldmat_dat1, const ldmat_data &ldmat_dat2,\
									 std::vector<size_t>& shared_idx1, std::vector<size_t>& shared_idx2,\
									 std::vector<size_t>& pop_idx1, std::vector<size_t>& pop_idx2) 
{
	size_t start_i1 = dat1.boundary[j].first;
    size_t end_i1 = dat1.boundary[j].second;

	size_t start_i2 = dat2.boundary[j].first;
    size_t end_i2 = dat2.boundary[j].second;

	size_t blk_size12 = shared_idx1.size();
	std::vector<int> assign12 = shared_assignment(dat1, dat2, ldmat_dat1, ldmat_dat2, j, shared_idx1, shared_idx2);
	for (size_t i = 0; i < blk_size12; i++)
	{
		cls_assgn1[start_i1 + shared_idx1[i]] = assign12[i];
		cls_assgn2[start_i2 + shared_idx2[i]] = assign12[i];
	}
	// Finish the shared part sampling
	std::vector<int> assign1 = indiv_assignment (dat1, ldmat_dat1, j, pop_idx1, 1);
	size_t blk_size1 = pop_idx1.size();
	for (size_t i = 0; i < blk_size1; i++) cls_assgn1[start_i1 + pop_idx1[i]] = assign1[i];

	std::vector<int> assign2 = indiv_assignment (dat2, ldmat_dat2, j, pop_idx2, 2);
	size_t blk_size2 = pop_idx1.size();
	for (size_t i = 0; i < blk_size2; i++) cls_assgn1[start_i1 + pop_idx1[i]] = assign1[i];

	// How about population 0??
}

void MCMC_state::sample_V() 
{
	for (size_t i = 1; i < 4 ;i++)
	{
		size_t m = V[i].size();
		vector<double> a(m - 1);
		a[m - 2] = suff_stats[m - 1 + population[i - 1]];
		for (int k = m - 3; k >= 0; k--) 
		{
			a[k] = suff_stats[k + 1 + population[i - 1]] + a[k + 1];
    	}
		double idx41 = 0;
    	for (size_t j = 0; j < m - 1; j++) 
		{
			if (idx41 == 1)
			{
				V[i][j] = 0;
				continue;
			}
			V[i][j] = gsl_ran_beta(r, \
				1 + suff_stats[j + population[i - 1]], \
				alpha[i] + a[j]);
			if (V[i][j] == 1) idx41 = 1;
    	}
    	V[i][m - 1] = 1.0 - idx41;
	}
}

void MCMC_state::update_p() 
{
	p[0] = p_pop[0];
	for (size_t j = 1; j < 4; j++)
	{
		int m = V[j].size();
		vector<double> cumprod(m - 1);
		cumprod[0] = 1 - V[j][0];

    	for (size_t i = 1; i < (m - 1); i++) 
		{
			cumprod[i] = cumprod[i - 1] * (1 - V[j][i]);
			if (V[j][i] == 1) 
			{
	    		std::fill(cumprod.begin() + i + 1, cumprod.end(), 0.0);
	    		break;
			}
    	}
    	p[population[j - 1]] = V[j][0]; 
    	for (size_t i = 1; i < m - 1; i++) 
		{
			p[population[j - 1] + i] = cumprod[i - 1] * V[j][i] * p_pop[j];
    	}

    	double sum = std::accumulate(p.begin() + population[j - 1], p.begin() + population[j] - 1, 0.0);
    	if (1 - sum > 0) 
		{
			p[m - 1] = 1 - sum;
    	}
    	else 
		{
			p[m - 1] = 0;
    	}

    	for (size_t i = 0; i < m; i++) 
		{
			log_p[population[j - 1] + i] = logf(p[population[j - 1] + i] + 1e-40); 
    	}
	}
}

void MCMC_state::sample_alpha() 
{
	for (size_t i = 1; i < 4; i++)
	{
		double sum = 0, m = 0;
		size_t left = population[i - 1];
		size_t right = population[i];
    	for (size_t j = left; j < right; j++) 
		{
			if (V[i][j] == 1) break;
			sum += log(1 - V[i][j]);
	    	m++;
		}
    	if (m == 0) m = 1;
    	alpha[i] = gsl_ran_gamma(r, para.a0 + m - 1, 1.0/(para.b0 - sum));
	}
}

void MCMC_state::sample_p_cluster(std::vector<size_t>& idx_pop1, std::vector<size_t>& idx_pop2)
{
	double m[] = {0, 0, 0, 0};
	int null_pop1 = 0;
	int null_pop2 = 0;
	for (size_t i = 0; i < idx_pop1.size(); i++)
	{
		if (cls_assgn1[idx_pop1[i]] == 0) null_pop1++;
	}
	for (size_t i = 0; i < idx_pop2.size(); i++)
	{
		if (cls_assgn2[idx_pop2[i]] == 0) null_pop2++;
	}
	size_t nonnull_pop1 = idx_pop1.size() - null_pop1;
	size_t nonnull_pop2 = idx_pop2.size() - null_pop2;
	m[0] = static_cast<double>(suff_stats[0] - nonnull_pop1 - nonnull_pop2);
	m[1] = static_cast<double>(std::accumulate(suff_stats.begin() + 1, suff_stats.begin() +\
											   population[1], 0) - nonnull_pop1);
	m[2] = static_cast<double>(std::accumulate(suff_stats.begin() + population[1] + 1, suff_stats.begin() + \
											   population[2], 0) - nonnull_pop2);
	m[3] = static_cast<double>(std::accumulate(suff_stats.begin() + population[2] + 1, suff_stats.begin() + \
												population[3], 0));
	double sum = 0.0;
	for (size_t i = 0; i < 4; i++)
	{
		p_pop[i] = gsl_ran_gamma(r, m[i], 1.0);
		sum += p_pop[i];
	}
	for (size_t i = 0; i < 4; i++)
	{
		p_pop[i] /= sum;
	}
}

void MCMC_state::sample_beta(size_t j, const mcmc_data &dat1, const mcmc_data &dat2, \
	        ldmat_data &ldmat_dat1, ldmat_data &ldmat_dat2) 
{
    size_t start_i1 = dat1.boundary[j].first;
    size_t end_i1 = dat1.boundary[j].second;
	size_t start_i2 = dat2.boundary[j].first;
    size_t end_i2 = dat2.boundary[j].second;

    vector <size_t>causal_list1;
	vector <size_t>causal_list2;
	vector <size_t>causal_list13;
	vector <size_t>causal_list23;
    for (size_t i = start_i1; i < end_i1; i++) 
	{
		if (cls_assgn1[i] >= 1 && cls_assgn1[i] < population[1]) 
		{
	    	causal_list1.push_back(i);
		}
		if (cls_assgn1[i] >= population[2] && cls_assgn1[i] < population[3]) 
		{
	    	causal_list13.push_back(i);
		}
    }
	for (size_t i = start_i2; i < end_i2; i++) 
	{
		if (cls_assgn2[i] >= population[1] && cls_assgn2[i] < population[2])
		{
	    	causal_list2.push_back(i);
		}
		if (cls_assgn2[i] >= population[2] && cls_assgn2[i] < population[3]) 
		{
	    	causal_list23.push_back(i);
		}
    }

    gsl_vector_view beta_1j = gsl_vector_subvector(beta1, \
	                            start_i1, end_i1 - start_i1);
	gsl_vector_view beta_2j = gsl_vector_subvector(beta2, \
	                            start_i2, end_i2 - start_i2);

    gsl_vector_set_zero(&beta_1j.vector);
	gsl_vector_set_zero(&beta_2j.vector);
	size_t idx1 = causal_list1.size();
	size_t idx2 = causal_list1.size();
	size_t pop_idx1 = causal_list1.size() + causal_list13.size();
	size_t pop_idx2 = causal_list2.size() + causal_list23.size();
	size_t all_idx = causal_list1.size() + causal_list2.size() +\
					  causal_list13.size() + causal_list23.size(); 

    if (all_idx  == 0) 
	{
		ldmat_dat1.num[j] = 0;
		ldmat_dat1.denom[j] = 0;
		ldmat_dat2.num[j] = 0;
		ldmat_dat2.denom[j] = 0;
		return;
    }
	else if (idx1 == 1 && pop_idx2 == 0)
	{
		double var_k1 = cluster_var[cls_assgn1[causal_list1[0]]];
		double b1j = gsl_vector_get(b1, causal_list1[0]);
		double B1jj = gsl_matrix_get(ldmat_dat1.B[j], \
									causal_list1[0] - start_i1, \
									causal_list1[0] - start_i1);
		double C1 = var_k1 / (N1 * var_k1 * square(eta) * B1jj + 1.0);
		double rv1 = sqrt(C1) * gsl_ran_ugaussian(r) + C1 * N1 * b1j;
		gsl_vector_set(&beta_1j.vector, causal_list1[0] - start_i1, \
						rv1);
		ldmat_dat1.num[j] = b1j * rv1;
		ldmat_dat1.denom[j] = square(rv1)*B1jj;
		return;
	}
	else if (idx2 == 1 && pop_idx1 == 0)
	{
		double var_k2 = cluster_var[cls_assgn2[causal_list2[0]]];
		double b2j = gsl_vector_get(b2, causal_list2[0]);
		double B2jj = gsl_matrix_get(ldmat_dat2.B[j], \
									causal_list2[0] - start_i2, \
									causal_list2[0] - start_i2);
		double C2 = var_k2 / (N2 * var_k2 * square(eta) * B2jj + 1.0);
		double rv2 = sqrt(C2) * gsl_ran_ugaussian(r) + C2 * N2 * b2j;
		gsl_vector_set(&beta_2j.vector, causal_list2[0] - start_i2, \
						rv2);
		ldmat_dat2.num[j] = b2j * rv2;
		ldmat_dat2.denom[j] = square(rv2)*B2jj;
		return;
	}
	else if (idx1 > 0 && pop_idx2 == 0)
	{
		gsl_vector* beta_tmp = sample_MVN_single(causal_list1, j, ldmat_dat1, start_i1, 1);
		for (size_t i = 0; i < causal_list1.size(); i++) 
		{
			gsl_vector_set(&beta_1j.vector, causal_list1[i] - start_i1, \
				gsl_vector_get(beta_tmp, i));
    	} 
		gsl_vector_free(beta_tmp);
		return;
	}
	else if (idx2 > 0 && pop_idx1 == 0)
	{
		gsl_vector* beta_tmp = sample_MVN_single(causal_list2, j, ldmat_dat2, start_i2, 2);
		for (size_t i = 0; i < causal_list2.size(); i++) 
		{
			gsl_vector_set(&beta_2j.vector, causal_list2[i] - start_i2, \
				gsl_vector_get(beta_tmp, i));
    	}
		gsl_vector_free(beta_tmp);
		return;
	}
	causal_list1.insert(causal_list1.end(), causal_list13.begin(), causal_list13.end());
	std::sort(causal_list1.begin(), causal_list1.end());
	causal_list2.insert(causal_list2.end(), causal_list23.begin(), causal_list23.end());
	std::sort(causal_list2.begin(), causal_list2.end());
	size_t B_size = causal_list1.size() + causal_list2.size();

	std::unordered_map<size_t, size_t> cor1_pop;
    for (int i = 0; i < causal_list1.size(); i++) 
	{
        cor1_pop[causal_list1[i]] = i;
    }
	std::unordered_map<size_t, size_t> cor2_pop;
    for (int i = 0; i < causal_list2.size(); i++) 
	{
        cor2_pop[causal_list2[i]] = i;
    }

    gsl_vector *A_vec = gsl_vector_alloc(B_size);
    gsl_vector *A_vec2 = gsl_vector_alloc(B_size);

    double C1 = square(eta) * N1; 
	double C2 = square(eta) * N2; 

    double *ptr = (double*) malloc(B_size * \
	    B_size * sizeof(ptr));

    if (!ptr) 
	{
		std::cerr << "Malloc failed for block " 
	    	" may due to not enough memory." << endl;
    }
	int cor_idx = 0;
	int map_idx = -1;

    for (size_t i = 0; i < B_size; i++) 
	{
		// N = 1.0 after May 21 2021
		// A_vec = N A[,idx].T beta_mrg = N A beta_mrg[idx]
		if (i < causal_list1.size())
		{
			gsl_vector_set(A_vec, i, N1 * eta * gsl_vector_get(ldmat_dat1.calc_b_tmp[j], \
		   		causal_list1[i] - start_i1));
			if (causal_list1[i] == causal_list13[cor_idx]) 
			{
        		int map_idx = cor2_pop[causal_list23[cor_idx]];
				cor_idx++;
    		}
		}
		cor_idx = 0;
		if (i >= causal_list1.size())
		{
			gsl_vector_set(A_vec, i, N2 * eta * gsl_vector_get(ldmat_dat2.calc_b_tmp[j], \
		   		causal_list2[i - causal_list1.size()] - start_i2));
			if (causal_list2[i - causal_list1.size()] == causal_list13[cor_idx]) 
			{
        		int map_idx = cor1_pop[causal_list13[cor_idx]];
				cor_idx++;
    		}
		}
	
		// B = N B_gamma + \Sigma_0^{-1}
		// auto-vectorized
		for (size_t k = 0; k < B_size; k++) 
		{
			if (i < causal_list1.size()) 
			{
				if (k >= causal_list1.size()) 
				{
					if (k == map_idx)
					{
						ptr[i * B_size + k] = -rho/(1 - square(rho))*(1.0/cluster_var[cls_assgn1[causal_list1[i]]]);
						map_idx = -1;
						continue;
					}
					ptr[i * B_size + k] = 0;
					continue;
				}
				if (i != k && k < causal_list1.size()) 
				{
					ptr[i * B_size + k] = C1 * \
					ldmat_dat1.B[j] -> data[ldmat_dat1.B[j]->tda * \
					(causal_list1[i] - start_i1) + \
					causal_list1[k] - start_i1];
					continue;
	    		}
				else 
				{
					ptr[i * B_size + k] = C1 * \
					ldmat_dat1.B[j] -> data[ldmat_dat1.B[j] -> tda * \
					(causal_list1[i] - start_i1) + \
					causal_list1[i] - start_i1] + \
					1.0/cluster_var[cls_assgn1[causal_list1[i]]] * \
					(1/(1 - square(rho)));
					continue;
	    		}
			}
			if (i >= causal_list1.size()) 
			{
				if (k < causal_list1.size()) 
				{
					if (k == map_idx)
					{
						ptr[i * B_size + k] = -rho/(1 - square(rho))*(1.0/cluster_var[cls_assgn1[causal_list1[map_idx]]]);
						map_idx = -1;
						continue;
					}
					ptr[i * B_size + k] = 0;
					continue;
				}
				size_t ii = i - causal_list1.size();
				size_t kk = k - causal_list1.size();
				if (i != k && k >= causal_list1.size()) 
				{
					ptr[i * B_size + k] = C2 * \
					ldmat_dat2.B[j] -> data[ldmat_dat2.B[j]->tda * \
					(causal_list2[ii] - start_i2) + \
					causal_list2[kk] - start_i2];
					continue;
	    		}
				else 
				{
					ptr[i * B_size + k] = C2 * \
					ldmat_dat1.B[j] -> data[ldmat_dat1.B[j] -> tda * \
					(causal_list1[ii] - start_i1) + \
					causal_list1[ii] - start_i1] + \
					1.0/cluster_var[cls_assgn2[causal_list2[ii]]] * \
					(1/(1 - square(rho)));
					continue;
	    		}
			}
		}
    }

    gsl_vector_memcpy(A_vec2, A_vec);

    gsl_matrix_view B = gsl_matrix_view_array(ptr, \
	    B_size, B_size);
	// form the vector into the matrix

    gsl_vector *beta_c = gsl_vector_alloc(B_size);
    
    for (size_t i = 0; i < B_size; i++) 
	{
		gsl_vector_set(beta_c, i, gsl_ran_ugaussian(r));
    }

	/*for (size_t i = 0; i < B_size; ++i) 
	{
        for (size_t j = 0; j < B_size; ++j) 
		{
            std::cout << gsl_matrix_get(&B.matrix, i, j) << " ";
        }
        std::cout << std::endl;
    }
	cout << j << endl;*/

    // (N B_gamma + \Sigma_0^-1) = L L^T
    gsl_linalg_cholesky_decomp1(&B.matrix);

    // \mu = L^{-1} A_vec
    gsl_blas_dtrsv(CblasLower, CblasNoTrans, \
	    CblasNonUnit, &B.matrix, A_vec);

    // N(\mu, I)
    gsl_blas_daxpy(1.0, A_vec, beta_c);

    // X ~ N(\mu, I), L^{-T} X ~ N( L^{-T} \mu, (L L^T)^{-1} )
    gsl_blas_dtrsv(CblasLower, CblasTrans, \
	    CblasNonUnit, &B.matrix, beta_c);
	
	gsl_blas_ddot(A_vec2, beta_c, &ldmat_dat1.num[j]);
	gsl_blas_ddot(A_vec2, beta_c, &ldmat_dat2.num[j]);
	
	gsl_matrix_view B1_view = gsl_matrix_submatrix(&B.matrix, 0, 0, causal_list1.size(), causal_list1.size());
    gsl_matrix *B1 = &B1_view.matrix;
	gsl_matrix_view B2_view = gsl_matrix_submatrix(&B.matrix, causal_list1.size(), causal_list1.size(), \
													causal_list2.size(), causal_list2.size());
    gsl_matrix *B2 = &B2_view.matrix;

    // compute eta related terms
    for (size_t i = 0; i < causal_list1.size(); i++) 
	{
		gsl_matrix_set(B1, i, i, \
		C1 * gsl_matrix_get(ldmat_dat1.B[j], 
	    	causal_list1[i] - start_i1, \
	    	causal_list1[i] - start_i1));
    }
	for (size_t i = 0; i < causal_list2.size(); i++) 
	{
		gsl_matrix_set(B2, i, i, \
		C2 * gsl_matrix_get(ldmat_dat2.B[j], 
	    	causal_list2[i] - start_i2, \
	    	causal_list2[i] - start_i2));
    }
	gsl_vector *beta_c1 = gsl_vector_alloc(causal_list1.size());
	gsl_vector *A1_vec = gsl_vector_alloc(causal_list1.size());
	gsl_vector *beta_c2 = gsl_vector_alloc(causal_list2.size());
	gsl_vector *A2_vec = gsl_vector_alloc(causal_list2.size());
	for (size_t i = 0; i < causal_list1.size(); i++) 
	{
		gsl_vector_set(beta_c1, i, gsl_vector_get(beta_c, i));
    }
	for (size_t i = causal_list1.size(); i < B_size; i++) 
	{
		gsl_vector_set(beta_c2, i - causal_list1.size(), gsl_vector_get(beta_c, i));
    }

	double denom1, denom2;
    gsl_blas_dsymv(CblasUpper, 1.0, B1, \
	    beta_c1, 0, A1_vec);
    gsl_blas_ddot(beta_c1, A1_vec, &denom1);
	gsl_blas_dsymv(CblasUpper, 1.0, B2, \
	    beta_c2, 0, A2_vec);
    gsl_blas_ddot(beta_c2, A2_vec, &denom2);
	ldmat_dat1.denom[j] = (denom1 + denom2) / square(eta);
	ldmat_dat2.denom[j] = (denom1 + denom2) / square(eta);
    ldmat_dat1.num[j] /= eta;
	ldmat_dat2.num[j] /= eta;

    for (size_t i = 0; i < B_size; i++) 
	{
		if (i < causal_list1.size())
		{
			gsl_vector_set(&beta_1j.vector, causal_list1[i] - start_i1, \
			gsl_vector_get(beta_c, i));
		}
		else
		{
			gsl_vector_set(&beta_2j.vector, causal_list2[i - causal_list1.size()] - start_i2, \
			gsl_vector_get(beta_c, i));
		}
    } 

    gsl_vector_free(A_vec);
	gsl_vector_free(A_vec2);
    gsl_vector_free(beta_c);
	gsl_vector_free(beta_c1);
	gsl_vector_free(beta_c2);
	gsl_vector_free(A1_vec);
	gsl_vector_free(A2_vec);
    free(ptr);
}

void MCMC_state::sample_eta(const ldmat_data &ldmat_dat) 
{
    double num_sum = std::accumulate(ldmat_dat.num.begin(), \
	    ldmat_dat.num.end(), 0.0);

    double denom_sum = std::accumulate(ldmat_dat.denom.begin(), \
	    ldmat_dat.denom.end(), 0.0);

    denom_sum += 1e-6;

    eta = gsl_ran_ugaussian(r) * sqrt(1.0/denom_sum) + \
	  num_sum / denom_sum;
}

void MCMC_state::compute_h2(const mcmc_data &dat1, const mcmc_data &dat2) 
{

    double h21_tmp, h22_tmp = 0;
    h22 = 0; h21 = 0;
    for (size_t j = 0; j < dat1.ref_ld_mat.size(); j++) 
	{
		size_t start_i1 = dat1.boundary[j].first;
		size_t end_i1 = dat1.boundary[j].second;
		size_t start_i2 = dat2.boundary[j].first;
		size_t end_i2 = dat2.boundary[j].second;
		gsl_vector *tmp1 = gsl_vector_alloc(end_i1 - start_i1);
		gsl_vector_view beta1_j = gsl_vector_subvector(beta1, \
	    	start_i1, end_i1 - start_i1);
		gsl_blas_dsymv(CblasUpper, 1.0, \
	    	dat1.ref_ld_mat[j], &beta1_j.vector, 0, tmp1);
		gsl_blas_ddot(tmp1, &beta1_j.vector, &h21_tmp);
		h21 += h21_tmp;

		gsl_vector *tmp2 = gsl_vector_alloc(end_i2 - start_i2);
		gsl_vector_view beta2_j = gsl_vector_subvector(beta2, \
	    	start_i2, end_i2 - start_i2);
		gsl_blas_dsymv(CblasUpper, 1.0, \
	    	dat2.ref_ld_mat[j], &beta2_j.vector, 0, tmp2);
		gsl_blas_ddot(tmp2, &beta2_j.vector, &h22_tmp);
		h22 += h22_tmp;

		gsl_vector_free(tmp1);
		gsl_vector_free(tmp2);
    }
}

// =========== The next ones are not related to state object ===========

void solve_ldmat(const mcmc_data &dat, ldmat_data &ldmat_dat, \
	const double a, unsigned sz, int opt_llk) 
{
    for (size_t i = 0; i < dat.ref_ld_mat.size(); i++) 
	{
		size_t size = dat.boundary[i].second - dat.boundary[i].first;
		gsl_matrix *A = gsl_matrix_alloc(size, size);
		gsl_matrix *B = gsl_matrix_alloc(size, size);
		gsl_matrix *L = gsl_matrix_alloc(size, size);
		gsl_matrix_memcpy(A, dat.ref_ld_mat[i]);
		gsl_matrix_memcpy(B, dat.ref_ld_mat[i]);
		gsl_matrix_memcpy(L, dat.ref_ld_mat[i]);


		if (opt_llk == 1) 
		{
	    	// (R + aNI) / N A = R via cholesky decomp
	    	// Changed May 21 2021 to divide by N
	    	// replace aN with a ???
	    	gsl_vector_view diag = gsl_matrix_diagonal(B);
	    	gsl_vector_add_constant(&diag.vector, a);
		}
		else 
		{
	    	// R_ij N_s,ij / N_i N_j
	    	// Added May 24 2021
	    	for (size_t j=0; j<size ; j++) 
			{
				for (size_t k=0; k<size; k++) 
				{
		    		double tmp = gsl_matrix_get(B, j, k);
		    		// if genotyped on two different arrays, N_s = 0
		    		size_t idx1 = j + dat.boundary[i].first;
		    		size_t idx2 = k + dat.boundary[i].first;
		    		if ( (dat.array[idx1] == 1 && dat.array[idx2] == 2) || \
			    	(dat.array[idx1] == 2 && dat.array[idx2] == 1) ) 
					{
						tmp = 0;
		    		}
		    	else 
				{
					tmp *= std::min(dat.sz[idx1], dat.sz[idx2]) / \
			       		(1.1 * dat.sz[idx1] * dat.sz[idx2]);
		    	}
		    	gsl_matrix_set(B, j, k, tmp);
				}
	    	}

	    	// force positive definite
	    	// B = Q \Lambda Q^T
	    	gsl_eigen_symmv_workspace *w = gsl_eigen_symmv_alloc(size);
	    	gsl_matrix *evac = gsl_matrix_alloc(size, size);
	    	gsl_matrix *eval = gsl_matrix_calloc(size, size);
	    	gsl_vector_view eval_diag = gsl_matrix_diagonal(eval);
	    	gsl_eigen_symmv(B, &eval_diag.vector, evac, w);

	    	// get minium of eigen value
	    	double eval_min = gsl_matrix_get(eval, 0, 0);
	    	for (size_t k=1; k<size; k++) 
			{
				double eval_k = gsl_matrix_get(eval, k, k);
				if (eval_k <= eval_min) 
				{
		    		eval_min = eval_k;
				}	
	    	}

	    	// restore lower half of B
	    	for (size_t j=0; j<size; j++) 
			{
				for (size_t k=0; k<j; k++) 
				{
		    		double tmp = gsl_matrix_get(B, k, j);
		    		gsl_matrix_set(B, j ,k, tmp);
				}
	    	}

	    	// if min eigen value < 0, add -1.1 * eval to diagonal
	    	for (size_t j=0; j<size; j++) 
			{
				if (eval_min < 0) 
				{
		    	gsl_matrix_set(B, j, j, \
			    	1.0/dat.sz[j+dat.boundary[i].first] - 1.1*eval_min);
				}
				else 
				{
		    		gsl_matrix_set(B, j, j, \
			    		1.0/dat.sz[j+dat.boundary[i].first]);
				}
			}
	    	gsl_matrix_free(evac);
	    	gsl_matrix_free(eval);
	    	gsl_eigen_symmv_free(w);
		}

		gsl_linalg_cholesky_decomp1(B);
		gsl_blas_dtrsm(CblasLeft, CblasLower, CblasNoTrans, \
			CblasNonUnit, 1.0, B, A);
		gsl_blas_dtrsm(CblasLeft, CblasLower, CblasTrans, \
		                CblasNonUnit, 1.0, B, A);
	
		// This creates A = (R + aNI)-1 R
	
		// Changed May 21 2021 to divide by N 
		/*if (opt_llk == 1) 
		{
	   		gsl_matrix_scale(A, sz);
		}*/

		// B = RA
		// Changed May 21 2021 as A may not be symmetric
		//gsl_blas_dsymm(CblasLeft, CblasUpper, 1.0, L, A, 0, B);
		gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, L, A, 0, B);
	
		// L = R %*% R;
		gsl_matrix_mul_elements(L, L);

		// memory allocation for A^T beta_mrg
		// Changed May 21 2021 from A to A^T
		gsl_vector *beta_mrg = gsl_vector_alloc(size);
		for (size_t j = 0; j < size; j++) 
		{
	    	gsl_vector_set(beta_mrg, j, dat.beta_mrg[j+dat.boundary[i].first]);
		}
		gsl_vector *b_tmp = gsl_vector_alloc(size);

		//gsl_blas_dsymv(CblasUpper, 1.0, A, beta_mrg, 0, b_tmp);
		// Changed May 21 2021 from A to A^T why??
		gsl_blas_dgemv(CblasTrans, 1.0, A, beta_mrg, 0, b_tmp);

		ldmat_dat.A.push_back(A);
		ldmat_dat.B.push_back(B);
		ldmat_dat.L.push_back(L);
		ldmat_dat.calc_b_tmp.push_back(b_tmp);
		ldmat_dat.beta_mrg.push_back(beta_mrg);
		ldmat_dat.denom.push_back(0);
		ldmat_dat.num.push_back(0);
    }
}

void initiate_assgn(const std::vector<size_t>* shared_idx1, const std::vector<size_t>* shared_idx2, \
					std::vector<size_t> &pop_idx1, std::vector<size_t> &pop_idx2, size_t bd_size, \
					mcmc_data &dat1, mcmc_data &dat2, MCMC_state &state, int *M)
{
	size_t n_snp1 = dat1.beta_mrg.size(); size_t n_snp2 = dat2.beta_mrg.size();
	for (size_t i = 0; i < bd_size; i++)
	{
		size_t start1 = dat1.boundary[i].first;
		size_t start2 = dat2.boundary[i].first;
		for (size_t j = 0; j < shared_idx1[i].size(); j++) 
		{
			state.cls_assgn2[start1 + shared_idx1[i][j]] = state.cls_assgn1[start2 + shared_idx2[i][j]];
    	}
	}
	gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
	for (size_t i = 0; i < n_snp1; i++)
	{
		if (std::find(pop_idx1.begin(), pop_idx1.end(), i) != pop_idx1.end())
		{
			state.cls_assgn1[i]	= gsl_rng_uniform_int(rng, M[1]);
		}
	}
	for (size_t i = 0; i < n_snp2; i++)
	{
		if (std::find(pop_idx2.begin(), pop_idx2.end(), i) != pop_idx2.end())
		{
			state.cls_assgn2[i]	= gsl_rng_uniform_int(rng, M[1]) + M[1];
		}
	}
	gsl_rng_free(rng);
}

void MCMC_state::sample_indiv (float** prob, float** tmp, const std::vector<float>& Bjj, \
								const std::vector<float>& bj, size_t pop)
{
	int N, M_left, M_right;
	if (pop == 1) 
	{
		N = N1;
		M_left = population[0];
		M_right = population[1];
	}
	
	if (pop == 2)
	{
		N = N2;
		M_left = population[1];
		M_right = population[2];
	}
	
	float C = pow(eta, 2.0) * N;
	size_t blk_size = Bjj.size();
	for (size_t i = 0; i < blk_size; i++) 
	{
		for (size_t k = M_left; k < M_right; k++) 
		{
	    	//cluster_var[k] = k*.5;
	    	prob[i][k] = C * Bjj[i] * cluster_var[k] + 1;
		}
    }
	for (size_t i = 0; i < blk_size; i++) 
	{
		for (size_t k = M_left; k < M_right; k++) 
		{
	    	tmp[i][k] = logf(prob[i][k]);
		}
    }
	for (size_t i = 0; i < blk_size; i++) 
	{
		for (size_t k = M_left; k < M_right; k++) 
		{
	    	prob[i][k] = -0.5*tmp[i][k] + log_p[k] + \
			 square(N*bj[i]) * cluster_var[k] / (2*prob[i][k]);
		}
    }
}


void MCMC_state::sample_cor (const std::vector<float>& B1jj, const std::vector<float>& B2jj, \
								const std::vector<float>& b1j, const std::vector<float>& b2j,\
								float** prob, float** tmp)
{
	int M_left = population[2];
	int M_right = population[3];
	
	size_t blk_size = B1jj.size();
	for (size_t i = 0; i < blk_size; i++) 
	{
		for (size_t k = M_left; k < M_right; k++) 
		{
			double c = rho/((1 - rho) * cluster_var[k]);
			double a1 = N1/2 * square(eta) * B1jj[i] + 1/(2 * (cluster_var[k]) * (1 - rho));
			double a2 = N2/2 * square(eta) * B2jj[i] + 1/(2 * (cluster_var[k]) * (1 - rho));
			double mu1 = (2 * a2 * N1 * b1j[i] + c * N2 * b2j[i]) / (4 * a1 * a2 - square(c));
			double mu2 = (2 * a2 * N1 * b1j[i] + c * N2 * b2j[i]) / (4 * a1 * a2 - square(c));
	    	//cluster_var[k] = k*.5;
	    	prob[i][k] = a1 * mu1 * mu1 + a2 * mu2 * mu2 - c * mu1 * mu2 + log_p[k];
			tmp[i][k] = (4 * a1 * a2 - square(c)) * (1 - square(rho)) * square(cluster_var[k]);
		}
    }
	for (size_t i = 0; i < blk_size; i++) 
	{
		for (size_t k = M_left; k < M_right; k++) 
		{
	    	tmp[i][k] = logf(prob[i][k]);
		}
    }
	for (size_t i = 0; i < blk_size; i++) 
	{
		for (size_t k = M_left; k < M_right; k++) 
		{
	    	prob[i][k] = -0.5 * tmp[i][k] + prob[i][k];
		}
    }
}

std::vector<int> MCMC_state::shared_assignment(const mcmc_data &dat1, const mcmc_data &dat2,\
									const ldmat_data &ldmat_dat1, const ldmat_data &ldmat_dat2, size_t j,\
									std::vector<size_t>& shared_idx1, std::vector<size_t>& shared_idx2)
{
	size_t start_i1 = dat1.boundary[j].first;
    size_t end_i1 = dat1.boundary[j].second;

	size_t start_i2 = dat2.boundary[j].first;
    size_t end_i2 = dat2.boundary[j].second;

	size_t blk_size = shared_idx1.size();

    vector<float> B1jj(blk_size);
    vector<float> b1j(blk_size); 
	vector<float> B2jj(blk_size);
    vector<float> b2j(blk_size); 
	float **prob = new float*[blk_size];
    float **tmp = new float*[blk_size];
	vector<float> rnd(blk_size);
    
    for (size_t i = 0; i < blk_size; i++) 
	{
		prob[i] = new float[num_cluster];
		tmp[i] = new float[num_cluster]; // initiate the 2-dim array

		B1jj[i] = gsl_matrix_get(ldmat_dat1.B[j], shared_idx1[i], shared_idx1[i]); // get the diagnol
		b1j[i] = gsl_vector_get(b1, start_i1 + shared_idx1[i]);

		B2jj[i] = gsl_matrix_get(ldmat_dat2.B[j], shared_idx2[i], shared_idx2[i]); // get the diagnol
		b2j[i] = gsl_vector_get(b2, start_i2 + shared_idx2[i]);
	
		prob[i][0] = log_p[0];
		rnd[i] = gsl_rng_uniform(r); // A randomly generated prob
    }
	sample_indiv (prob, tmp, B1jj, b1j, 1);
	sample_indiv (prob, tmp, B2jj, b2j, 2);
	sample_cor(B1jj, B2jj, b1j, b2j, prob, tmp);

	std::vector<int> assignment = sample_from_cat(blk_size, prob, num_cluster, rnd);
	for (size_t i = 0; i < blk_size; i++)
	{
		delete[] prob[i]; delete tmp[i];
	}
    delete[] prob; delete[] tmp;
	return assignment;
}

std::vector<int> MCMC_state::indiv_assignment(const mcmc_data &dat, const ldmat_data &ldmat_dat, \
									size_t j, std::vector<size_t>& idx_pop, int pop)
{
	size_t start_i = dat.boundary[j].first;
    size_t end_i = dat.boundary[j].second;

	size_t blk_size = idx_pop.size();

    vector<float> Bjj(blk_size);
    vector<float> bj(blk_size); 
	float **prob = new float*[blk_size];
    float **tmp = new float*[blk_size];
	vector<float> rnd(blk_size);
	for (size_t i = 0; i < blk_size; i++) 
	{
		prob[i] = new float[population[pop] - population[pop - 1]];
		tmp[i] = new float[population[pop] - population[pop - 1]]; // initiate the 2-dim array

		Bjj[i] = gsl_matrix_get(ldmat_dat.B[j], idx_pop[i], idx_pop[i]); // get the diagnol
		if (pop == 1) bj[i] = gsl_vector_get(b1, start_i + idx_pop[i]);
		if (pop == 2) bj[i] = gsl_vector_get(b2, start_i + idx_pop[i]);
	
		prob[i][0] = log_p[0];
		rnd[i] = gsl_rng_uniform(r); // A randomly generated prob
    }
	int N;
	if (pop == 1) N = N1;
	if (pop == 2) N = N2;
	float C = pow(eta, 2.0) * N;
	for (size_t i = 0; i < blk_size; i++) 
	{
		for (size_t k = 0; k < 1000; k++)
		{
	    	//cluster_var[k] = k*.5;
	    	prob[i][k] = C * Bjj[i] * cluster_var[k + population[pop - 1]] + 1;
		}
    }
	for (size_t i = 0; i < blk_size; i++) 
	{
		for (size_t k = 0; k < 1000; k++) 
		{
	    	tmp[i][k] = logf(prob[i][k]);
		}
    }
	for (size_t i = 0; i < blk_size; i++) 
	{
		for (size_t k = 0; k < 1000; k++) 
		{
	    	prob[i][k] = -0.5*tmp[i][k] + log_p[k + population[pop - 1]] + \
			 square(N * bj[i]) * cluster_var[k + population[pop - 1]] / (2*prob[i][k]);
		}
    }
	std::vector<int> assignment = sample_from_cat(blk_size, prob, 1000, rnd);
	for (size_t i = 0; i < blk_size; i++)
	{
		delete[] prob[i]; delete tmp[i];
	}
    delete[] prob; delete[] tmp;
	return assignment;
}

gsl_vector* MCMC_state::sample_MVN_single(const std::vector<size_t>& causal_list, size_t j,\
									ldmat_data &ldmat_dat, size_t start_i, int pop)
{
	int N;
	if (pop == 1) N = N1;
	if (pop == 2) N = N2;
	gsl_vector *A_vec = gsl_vector_alloc(causal_list.size());

    gsl_vector *A_vec2 = gsl_vector_alloc(causal_list.size());

    double C = square(eta) * N; 

    double *ptr = (double*) malloc(causal_list.size() * \
	    causal_list.size() * sizeof(ptr));

    if (!ptr) 
	{
		std::cerr << "Malloc failed for block " 
	    	" may due to not enough memory." << endl;
    }

    for (size_t i = 0; i < causal_list.size(); i++) 
	{
		// N = 1.0 after May 21 2021
		// A_vec = N A[,idx].T beta_mrg = N A beta_mrg[idx]
		gsl_vector_set(A_vec, i, N * eta * gsl_vector_get(ldmat_dat.calc_b_tmp[j], \
		   		causal_list[i] - start_i));
	
		// B = N B_gamma + \Sigma_0^{-1}
		// auto-vectorized
		for (size_t k = 0; k < causal_list.size(); k++) 
		{
	    	if (i != k) 
			{
				ptr[i * causal_list.size() + k] = C * \
				ldmat_dat.B[j] -> data[ldmat_dat.B[j]->tda * \
				(causal_list[i] - start_i) + \
				causal_list[k] - start_i];
	    	}
	    	else 
			{
				if (pop == 1)
				{
					ptr[i * causal_list.size() + k] = C * \
					ldmat_dat.B[j] -> data[ldmat_dat.B[j] -> tda * \
					(causal_list[i] - start_i) + \
					causal_list[i] - start_i] + \
					1.0/cluster_var[cls_assgn1[causal_list[i]]];
				}
				if (pop == 2)
				{
					ptr[i * causal_list.size() + k] = C * \
					ldmat_dat.B[j] -> data[ldmat_dat.B[j] -> tda * \
					(causal_list[i] - start_i) + \
					causal_list[i] - start_i] + \
					1.0/cluster_var[cls_assgn2[causal_list[i]]];
				}
	    	}
	   		// gsl_matrix_set(B, i, k, tmp);
		}// Get the matrix in the vectorized form
    }

    gsl_vector_memcpy(A_vec2, A_vec);

    gsl_matrix_view B = gsl_matrix_view_array(ptr, \
	    causal_list.size(), causal_list.size());
	// form the vector into the matrix

    gsl_vector *beta_c = gsl_vector_alloc(causal_list.size());
    
    for (size_t i = 0; i < causal_list.size(); i++) 
	{
		gsl_vector_set(beta_c, i, gsl_ran_ugaussian(r));
    }

    // (N B_gamma + \Sigma_0^-1) = L L^T
    gsl_linalg_cholesky_decomp1(&B.matrix);

    // \mu = L^{-1} A_vec
    gsl_blas_dtrsv(CblasLower, CblasNoTrans, \
	    CblasNonUnit, &B.matrix, A_vec);

    // N(\mu, I)
    gsl_blas_daxpy(1.0, A_vec, beta_c);

    // X ~ N(\mu, I), L^{-T} X ~ N( L^{-T} \mu, (L L^T)^{-1} )
    gsl_blas_dtrsv(CblasLower, CblasTrans, \
	    CblasNonUnit, &B.matrix, beta_c);

    // compute eta related terms
    for (size_t i = 0; i < causal_list.size(); i++) 
	{
		gsl_matrix_set(&B.matrix, i, i, \
		C * gsl_matrix_get(ldmat_dat.B[j], 
	    	causal_list[i] - start_i, \
	    	causal_list[i] - start_i));
    }

    gsl_blas_ddot(A_vec2, beta_c, &ldmat_dat.num[j]);
    gsl_blas_dsymv(CblasUpper, 1.0, &B.matrix, \
	    beta_c, 0, A_vec);
    gsl_blas_ddot(beta_c, A_vec, &ldmat_dat.denom[j]);
    ldmat_dat.denom[j] /= square(eta);
    ldmat_dat.num[j] /= eta;

    gsl_vector_free(A_vec);
    gsl_vector_free(A_vec2);
    free(ptr);
	return beta_c;
}

#endif