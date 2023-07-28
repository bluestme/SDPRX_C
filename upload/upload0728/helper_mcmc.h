#ifndef HELPER_MCMC_H
#define HELPER_MCMC_H
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
#include "mcmc.h"

using std::ifstream; using std::string;
using std::vector; using std::unordered_map; 
using std::cout; using std::endl;
using std::pair; using std::find;

std::vector<int> sample_from_cat (size_t blk_size, float **prob, size_t M, std::vector<float> &rnd);
void solve_ldmat(const mcmc_data &dat, ldmat_data &ldmat_dat, const double a, unsigned sz, int opt_llk);
void initiate_assgn(const std::vector<size_t>* shared_idx1, const std::vector<size_t>* shared_idx2, \
					std::vector<size_t> &pop_idx1, std::vector<size_t> &pop_idx2, size_t bd_size, \
					mcmc_data &dat1, mcmc_data &dat2, MCMC_state &state, int *M);

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
			 square(N * bj[i]) * cluster_var[k] / (2*prob[i][k]);
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

std::vector<int> sample_from_cat (size_t blk_size, float **prob, size_t M, std::vector<float> &rnd)
{
    std::vector<int> assignment;
	assignment.assign(M, 0);
    for (size_t i = 0; i < blk_size; i++) 
	{
		// non SSE version to find max
		float max_elem = *std::max_element(&prob[i][0], &prob[i][M-1]);
		float log_exp_sum;
		for (int k = 0; k < M; k++) 
		{
	    	log_exp_sum += expf(prob[i][k] - max_elem);
		}
		log_exp_sum = max_elem + logf(log_exp_sum);

		assignment[i] = M - 1; // why minus one?
		for (size_t k = 0; k < M - 1; k++) 
		{
	    	rnd[i] -= expf(prob[i][k] - log_exp_sum);
	    	if (rnd[i] < 0) 
			{
				assignment[i] = k;
				break;
	    	}
		}
    }
    return assignment;
}

#endif