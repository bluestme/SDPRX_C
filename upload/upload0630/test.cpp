#include <algorithm>
#include "parse_gen.h"
#include "state.h"
#include <unordered_map>
#include <stdexcept>
#include "gsl/gsl_cdf.h"
#include "gsl/gsl_eigen.h"
#include "gsl/gsl_blas.h"
#include "helper.h"
#include <math.h>
#include <fstream>
#include <sstream>

int main() 
{
	//Read in 2 files and their reference panels
    mcmc_data dat1;
	mcmc_data dat2;
	int N1 = 5000;
	int N2 = 5000;
    coord("test1/ref/chr22.snpInfo", "test1/summary_stat/sim_1.txt", \
	    "test1/genotype/eur_chr22.bim", \
	    "test1/ref/chr22.dat", dat1, N1, 1);
	coord("test2/ref/chr22.snpInfo", "test2/summary_stat/sim_1.txt", \
	    "test2/genotype/eur_chr22.bim", \
	    "test2/ref/chr22.dat", dat2, N2, 1);
	
	double a = 0.1;
	int M[] = {1, 1000, 1000, 1000};
	double a0k = 0.5; double b0k = 0.5;
	MCMC_state state = MCMC_state(dat1.beta_mrg.size(),dat2.beta_mrg.size(), M, a0k, b0k, N1, N2);
	ldmat_data ldmat_dat1;
	ldmat_data ldmat_dat2;
	solve_ldmat(dat1, ldmat_dat1, a, N1, 1);
	solve_ldmat(dat2, ldmat_dat2, a, N2, 1);
	MCMC_samples samples = MCMC_samples(dat1.beta_mrg.size(), dat2.beta_mrg.size());

	size_t bd_size = dat1.boundary.size();
	std::vector<double> shared_idx1[bd_size];
	std::vector<double> shared_idx2[bd_size];
	get_shared_idx(shared_idx1, bd_size, dat1, dat2);
	get_shared_idx(shared_idx2, bd_size, dat2, dat1);
	
	for (size_t i=0; i<dat1.ref_ld_mat.size(); i++) 
	{
		gsl_matrix_free(ldmat_dat1.A[i]);
		gsl_matrix_free(ldmat_dat1.B[i]);
		gsl_matrix_free(ldmat_dat1.L[i]);
		gsl_vector_free(ldmat_dat1.calc_b_tmp[i]);
		gsl_vector_free(ldmat_dat1.beta_mrg[i]);
		gsl_matrix_free(dat1.ref_ld_mat[i]);
    }
	for (size_t i=0; i<dat2.ref_ld_mat.size(); i++) 
	{
		gsl_matrix_free(ldmat_dat2.A[i]);
		gsl_matrix_free(ldmat_dat2.B[i]);
		gsl_matrix_free(ldmat_dat2.L[i]);
		gsl_vector_free(ldmat_dat2.calc_b_tmp[i]);
		gsl_vector_free(ldmat_dat2.beta_mrg[i]);
		gsl_matrix_free(dat2.ref_ld_mat[i]);
    }
    return 0;
}