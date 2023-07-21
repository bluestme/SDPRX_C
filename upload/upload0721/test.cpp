#include <algorithm>
#include "parse_gen.h"
#include "state.h"
#include "helper.h"
#include <unordered_map>
#include <stdexcept>
#include "gsl/gsl_cdf.h"
#include "gsl/gsl_eigen.h"
#include "gsl/gsl_blas.h"
#include "function_pool.h"
#include <math.h>
#include <thread>
#include <chrono>
#include <fstream>
#include <sstream>
#include <numeric>
#include <functional>

using namespace std::chrono;

using std::cout; using std::endl;
using std::thread; using std::ref;
using std::vector; using std::ofstream;
using std::string; using std::min;

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
	
	double a = 0.1; double rho = 0.8;
	int M[] = {1, 1000, 1000, 1000};
	double a0k = 0.5; double b0k = 0.5;
	MCMC_state state = MCMC_state(dat1.beta_mrg.size(),dat2.beta_mrg.size(), M, a0k, b0k, N1, N2, dat1, dat2, rho);
	ldmat_data ldmat_dat1;
	ldmat_data ldmat_dat2;
	solve_ldmat(dat1, ldmat_dat1, a, N1, 1);
	solve_ldmat(dat2, ldmat_dat2, a, N2, 1);
	MCMC_samples samples = MCMC_samples(dat1.beta_mrg.size(), dat2.beta_mrg.size());

	size_t bd_size = dat1.boundary.size();
	std::vector<size_t> shared_idx1[bd_size];
	std::vector<size_t> shared_idx2[bd_size];
	std::vector<size_t> pop_idx1;
	std::vector<size_t> pop_idx2;
	get_shared_idx(shared_idx1, bd_size, dat1, dat2);
	get_shared_idx(shared_idx2, bd_size, dat2, dat1);
	get_pop_idx(shared_idx1, pop_idx1, bd_size, dat1);
	get_pop_idx(shared_idx2, pop_idx2, bd_size, dat2);
	initiate_assgn(shared_idx1, shared_idx2, pop_idx1, pop_idx2, bd_size, dat1, dat2, state, M);

	state.update_suffstats(pop_idx1, pop_idx2, shared_idx1, shared_idx2);
	state.sample_sigma2();
	Function_pool func_pool(1);
	for (size_t i = 0; i < dat1.ref_ld_mat.size(); i++) 
	{
	    state.calc_b(i, dat1, dat2, ldmat_dat1, ldmat_dat2);
	}
	/*for (size_t i = 0; i < bd_size; i++) 
	{
	    func_pool.push(std::bind(&MCMC_state::sample_assignment, &state, i, ref(dat1), \
								ref(dat2), ref(ldmat_dat1), ref(ldmat_dat2), ref(shared_idx1[i]),\
								ref(shared_idx2[i]), ref(pop_idx1), ref(pop_idx2)));
	}*/
	for (size_t i = 0; i < bd_size; i++)
	{
		state.sample_assignment(i, dat1, dat2, ldmat_dat1, ldmat_dat2, shared_idx1[i], shared_idx2[i], pop_idx1, pop_idx2);
	}
	state.update_suffstats(pop_idx1, pop_idx2, shared_idx1, shared_idx2);

	state.sample_V();
	state.update_p();
	state.sample_alpha();

	for (size_t i = 0; i < bd_size; i++) 
	{
	    state.sample_beta(i, dat1, dat2, ldmat_dat1, ldmat_dat2);
	}
	for (size_t i = 0; i < bd_size; i++) 
	{
	    state.sample_beta(i, dat1, dat2, ldmat_dat1, ldmat_dat2);
	}
	state.compute_h2(dat1, dat2);
	state.sample_eta(ldmat_dat1);

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