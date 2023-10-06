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

void solve_ldmat(const mcmc_data &dat, ldmat_data &ldmat_dat, const double a, unsigned sz, int opt_llk);
void initiate_assgn(const std::vector<std::vector<size_t>>& shared_idx1, const std::vector<std::vector<size_t>>& shared_idx2, \
					std::vector<size_t> &pop_idx1, std::vector<size_t> &pop_idx2, size_t bd_size, \
					mcmc_data &dat1, mcmc_data &dat2, MCMC_state &state, int *M);

void solve_ldmat(const mcmc_data &dat, ldmat_data &ldmat_dat, \
	const double a, int opt_llk) 
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

void initiate_assgn(const std::vector<std::vector<size_t>>& shared_idx1, const std::vector<std::vector<size_t>>& shared_idx2, \
					std::vector<size_t> &pop_idx1, std::vector<size_t> &pop_idx2, size_t bd_size, \
					mcmc_data &dat1, mcmc_data &dat2, MCMC_state &state, int *M)
{
	for (size_t i = 0; i < bd_size; i++)
	{
		size_t start1 = dat1.boundary[i].first;
		size_t start2 = dat2.boundary[i].first;
		for (size_t j = 0; j < shared_idx1[i].size(); j++) 
		{
			state.cls_assgn2[start2 + shared_idx2[i][j]] = state.cls_assgn1[start1 + shared_idx1[i][j]];
    	}
	}
	gsl_rng* rng = gsl_rng_alloc(gsl_rng_default);
	for (size_t i = 0; i < pop_idx1.size(); i++)
	{
		state.cls_assgn1[pop_idx1[i]] = gsl_rng_uniform_int(rng, M[1]);
	}
	for (size_t i = 0; i < pop_idx2.size(); i++)
	{
		state.cls_assgn2[pop_idx2[i]] = gsl_rng_uniform_int(rng, M[1]) + M[1];	
	}
	gsl_rng_free(rng);
}

gsl_vector* MCMC_state::sample_MVN_single(const std::vector<size_t>& causal_list, size_t j,\
									ldmat_data &ldmat_dat, size_t start_i, int pop)
{
	double N;
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