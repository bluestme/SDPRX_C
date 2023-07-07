#ifndef STATE_H
#define STATE_H

#include <iostream>
#include <vector>
#include "parse_gen.h"
#include "gsl/gsl_vector.h"
#include "gsl/gsl_matrix.h"
#include "gsl/gsl_rng.h"
#include <math.h>
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
	double h12;
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
	void calc_b(size_t j, const mcmc_data &dat1, const mcmc_data &dat2, const ldmat_data &ldmat_dat1,\
						const ldmat_data &ldmat_dat2);

	void sample_assignment(size_t j, const mcmc_data &dat, \
		        const ldmat_data &ldmat_dat);
	
	void update_suffstats(std::vector<size_t> &pop_idx1, std::vector<size_t> &pop_idx2,\
								  const std::vector<size_t>* shared_idx1, const std::vector<size_t>* shared_idx2);
	void sample_V();
	void update_p();
	void sample_alpha();
	void sample_beta(size_t j, const mcmc_data &dat, \
		       ldmat_data &ldmat_dat);
	void compute_h2(const mcmc_data &dat);
	void sample_eta(const ldmat_data &ldmat_dat);

    private:
	size_t M, n_snp1, n_snp2;
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
									 const ldmat_data &ldmat_dat1, const ldmat_data &ldmat_dat2) 
{
    size_t start_i1 = dat1.boundary[j].first;
    size_t end_i1 = dat1.boundary[j].second;

	size_t start_i2 = dat2.boundary[j].first;
    size_t end_i2 = dat2.boundary[j].second;
   
    float **prob = new float*[end_i-start_i];
    float **tmp = new float*[end_i-start_i];
	// 2 dimensional arrays (matrices) for the probability

    vector<float> B1jj(end_i-start_i);
    vector<float> b1j(end_i-start_i); 
	vector<float> B2jj(end_i-start_i);
    vector<float> b2j(end_i-start_i); 
    vector<float> rnd(end_i-start_i);

    float max_elem, log_exp_sum = 0;
    
    for (size_t i=0; i<end_i-start_i; i++) 
	{
		prob[i] = new float[num_cluster];
		tmp[i] = new float[num_cluster]; // initiate the 2-dim array

		Bjj[i] = gsl_matrix_get(ldmat_dat.B[j], i, i); // get the diagnol
		bj[i] = gsl_vector_get(b, start_i+i);
	
		prob[i][0] = log_p[0];
		rnd[i] = gsl_rng_uniform(r); // A randomly generated prob
    }

	size_t M = population[1];

    // N = 1.0 after May 21 2021
    float C = pow(eta, 2.0) * N;

    // auto vectorized
    for (size_t i=0; i<end_i-start_i; i++) 
	{
		for (size_t k=1; k<M; k++) 
		{
	    	//cluster_var[k] = k*.5;
	    	prob[i][k] = C * Bjj[i] * cluster_var[k] + 1;
		}
    }

    // unable to auto vectorize due to log
    // explicitly using SSE 
    __m128 _v, _m;
    for (size_t i=0; i<end_i-start_i; i++) 
	{
		size_t k = 1;
		for (; k<M; k+=4) 
		{ // require M >= 4
	    	_v = log_ps(_mm_loadu_ps(&prob[i][k]));
	    	_mm_storeu_ps(&tmp[i][k], _v);
		}

		for (; k<M; k++) 
		{
	    	tmp[i][k] = logf(prob[i][k]);
		}
    }

    // auto vectorized
    for (size_t i=0; i<end_i-start_i; i++) 
	{
		for (size_t k=1; k<M; k++) 
		{
	    	prob[i][k] = -0.5*tmp[i][k] + log_p[k] + \
			 square(N*bj[i]) * cluster_var[k] / (2*prob[i][k]);
		}
    }
	// The computation of the formula has finished


    for (size_t i=0; i<end_i-start_i; i++) 
	{
		// non SSE version to find max
		//max_elem = *std::max_element(&prob[i][0], &prob[i][M-1]);

		// SSE version to find max
		// https://shybovycha.github.io/2017/02/21/speeding-up-algorithms-with-sse.html
		_v = _mm_loadu_ps(prob[i]);
		size_t k = 4;
		for (; k<M; k+=4) 
		{
	    	_v = _mm_max_ps(_v, _mm_loadu_ps(&prob[i][k]));
		}
		// _v stores the maximum value of prob[i]
	
		for (size_t m=0; m<3; m++) 
		{
	    	_v = _mm_max_ps(_v, _mm_shuffle_ps(_v, _v, 0x93));
		}
		// After the loop, any remaining elements (if the length is not divisible by four) 
		// are handled in a separate loop using scalar operations

		_mm_store_ss(&max_elem, _v);
		// store the maximum value _v into max_elem

		for (; k<M; k++) 
		{
	    	max_elem = (max_elem > prob[i][k]) ? (max_elem) : (prob[i][k]);
		}

		// SSE version log exp sum
		_m = _mm_load1_ps(&max_elem); // load the maximum number into _m
		_v = exp_ps(_mm_sub_ps(_mm_loadu_ps(prob[i]), _m));
		// the exponential of the difference between each element of prob[i] and _m using the exp_ps function
	
		k = 4;
		for (; k<M; k+=4) 
		{
	    	_v = _mm_add_ps(_v, exp_ps(_mm_sub_ps(_mm_loadu_ps(&prob[i][k]), _m)));
		}

		_v = _mm_hadd_ps(_v, _v);
		_v = _mm_hadd_ps(_v, _v);
		_mm_store_ss(&log_exp_sum, _v);

		for (; k<M; k++) 
		{
	    	log_exp_sum += expf(prob[i][k] - max_elem);
		}
		// Why do we use the max_elem here?
		log_exp_sum = max_elem + logf(log_exp_sum);

		cls_assgn[i + start_i] = M - 1;
		for (size_t k = 0; k < M - 1; k++) 
		{
	    	rnd[i] -= expf(prob[i][k] - log_exp_sum);
	    	if (rnd[i] < 0) 
			{
				cls_assgn[i+start_i] = k;
				break;
	    	}
		}
		delete[] prob[i]; delete tmp[i];
    }
    delete[] prob; delete[] tmp;
}

// =========== The next ones are not related to state object ===========

void solve_ldmat(const mcmc_data &dat, ldmat_data &ldmat_dat, \
	const double a, unsigned sz, int opt_llk) {
    for (size_t i=0; i<dat.ref_ld_mat.size(); i++) {
	size_t size = dat.boundary[i].second - dat.boundary[i].first;
	gsl_matrix *A = gsl_matrix_alloc(size, size);
	gsl_matrix *B = gsl_matrix_alloc(size, size);
	gsl_matrix *L = gsl_matrix_alloc(size, size);
	gsl_matrix_memcpy(A, dat.ref_ld_mat[i]);
	gsl_matrix_memcpy(B, dat.ref_ld_mat[i]);
	gsl_matrix_memcpy(L, dat.ref_ld_mat[i]);


	if (opt_llk == 1) {
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
	if (opt_llk == 1) {
	    gsl_matrix_scale(A, sz);
	}

	// B = RA
	// Changed May 21 2021 as A may not be symmetric
	//gsl_blas_dsymm(CblasLeft, CblasUpper, 1.0, L, A, 0, B);
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, L, A, 0, B);
	
	// L = R %*% R;
	gsl_matrix_mul_elements(L, L);

	// memory allocation for A^T beta_mrg
	// Changed May 21 2021 from A to A^T
	gsl_vector *beta_mrg = gsl_vector_alloc(size);
	for (size_t j=0; j<size; j++) {
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