
#include <iostream>
#include "LD.h"
#include "mcmc.h"
#include <string.h>
#include "time.h"

using std::cout; using std::endl;
using std::string;

void print_use() 
{
    cout << "Usage: SDPRX -options" << endl << endl
	<< "Example of estimating LD: " << endl 
	<< "	SDPRX -make_ref -ref_prefix1 ./test/1kg -ref_prefix2 ./test/2kg -ref_dir1 ./test/ref1 -ref_dir2 ./test/ref2 -chr 1 -make_ref" << endl << endl
	<< "Example of performing mcmc: " << endl 
	<< "	SDPR -mcmc -ss1 ./test/ss/ss1.txt -ss2 ./test/ss/ss2.txt -ref_dir ./test/ref -chr 1 -out ./test/SDPR_out.txt" << endl << endl
	<< "Full list of options: " << endl << endl
	<< "-make-ref estimate reference LD matrix." << endl << endl
	<< "-mcmc perform MCMC." << endl << endl
	<< " -ss1 (required for -make_ref) path to the summary statistics file for population 1. We recommend to follow our pipeline to clean up summary statistics by running munge_sumstats.py." << endl
	<< " -ss2 (required for -make_ref) path to the summary statistics file for population 2. We recommend to follow our pipeline to clean up summary statistics by running munge_sumstats.py." << endl << endl
	<< " The summary statistics must have the following format (you can change the name of the header line): " << endl << endl  
	<< " SNP	A1	A2	BETA	P" << endl 
	<< " rs737657        A       G       -2.044  0.0409" << endl
	<< " rs7086391       T       C       -2.257  0.024" << endl
	<< " rs1983865       T       C       3.652   0.00026" << endl
	<< " ..." << endl 
	<< " where SNP is the marker name, A1 is the effect allele, A2 is the alternative allele, BETA is the regression coefficient for quantitative traits or log odds ratio for binary traits, and P is the p value." << endl << endl
	<< " -ref_prefix1 (required for -make_ref) path to the prefix of the bim file for the reference panel for population 1, not including the .bim suffix." << endl
	<< " -ref_prefix2 (required for -make_ref) path to the prefix of the bim file for the reference panel for population 2, not including the .bim suffix." << endl << endl
	<< " -ref_dir (required) path to the directory that contains the reference LD information output by SDPRX, containing .snpInfo and .dat file." << endl << endl
	<< " -valid (optional) path to the bim file for the testing dataset, including the .bim suffix." << endl << endl
	<< " -out1 (required for -mcmc) path to the output file for population 1 containing estimated effect sizes." << endl
	<< " -out2 (required for -mcmc) path to the output file for population 2 containing estimated effect sizes." << endl << endl
	<< " -chr (required) chromsome to work on. Currently support 1-22. Recommend to run in parallel." << endl << endl
	<< " -N1 (required for -mcmc) GWAS sample size for population 1." << endl
	<< " -N2 (required for -mcmc) GWAS sample size for population 2." << endl << endl
	<< " -opt_llk (optional) Which likelihood to evaluate. 1 for vanilla modified likelihood and 2 for SNPs genotyped on different individuals. Please refer to manuscript or manual for more details. Default is 1." << endl << endl
	<< " -iter (optional) number of iterations for MCMC. Default is 1000." << endl << endl
	<< " -burn (optional) number of burn-in for MCMC. Default is 200." << endl << endl 
	<< " -thin (optional) Thinning for MCMC. Default is 1 (no thin). " << endl << endl
	<< " -n_threads (optional) number of threads to use. Default is 1." << endl << endl
	<< " -r2 (optional) r2 cut-off for parition of independent blocks. Default is 0.1." << endl << endl
	<< " -a (optional) factor to shrink the reference LD matrix. Default is 0.1. Please refer to the manual for more information." << endl << endl
	<< " -c1 (optional) factor to correct for the deflation. Default is 1 for population 1. Please refer to the manual for more information." << endl
	<< " -c2 (optional) factor to correct for the deflation. Default is 1 for population 2. Please refer to the manual for more information." << endl << endl
	<< " -a0k (optional) hyperparameter for inverse gamma distribution. Default is 0.5." << endl << endl
	<< " -b0k (optional) hyperparameter for inverse gamma distribution. Default is 0.5." << endl << endl
	<< " -rho (optional) the correlation between population 1 and 2. Default is 0.8." << endl << endl
	<< " -force_shared (optional) if two population are forced to share the effect size with each other" << endl << endl
	<< " -h print the options." << endl << endl;
}

int main(int argc, char *argv[]) 
{
    if (argc == 1) 
	{
		print_use();
		return 0;
    }

    string ss_path1, ref_prefix1, ref_dir, valid_bim;
	string ss_path2, ref_prefix2, out_path1, out_path2;
    int N1 = 0, N2 = 0, n_threads = 1, chr = 0;
    double a = 0.1, c1 = 1, c2 = 1, a0k = .5, b0k = .5, r2 = .1, rho = .8;
    int iter = 1000, burn = 200, thin = 1;
    int make_ref = 0, run_mcmc = 0;
    int opt_llk = 1;
	int f_shared = 1;

    // pass command line arguments
    int i = 1;
    while (i < argc) 
	{
		char *end;

		if (strcmp(argv[i], "-ss1") == 0) 
		{
	    	ss_path1 = argv[i + 1];
	    	i += 2;
		}
		else if (strcmp(argv[i], "-ss2") == 0) 
		{
	    	ss_path2 = argv[i + 1];
	    	i += 2;
		}
		else if (strcmp(argv[i], "-force_shared") == 0) 
		{
			f_shared = strtol(argv[i + 1], &end, 10);
			if (f_shared != 0 && f_shared != 1)
			{
				cout << "Incorrect force shared option: " << argv[i + 1] << endl;
				return 0;
			}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-ref_prefix1") == 0) 
		{
	    	ref_prefix1 = argv[i + 1];
	    	i += 2;
		}
		else if (strcmp(argv[i], "-ref_prefix2") == 0) 
		{
	    	ref_prefix2 = argv[i + 1];
	    	i += 2;
		}
		else if (strcmp(argv[i], "-ref_dir") == 0) 
		{
	    	ref_dir = argv[i + 1];
	    	i += 2;
		}
		else if (strcmp(argv[i], "-valid") == 0) 
		{
	    	valid_bim = argv[i + 1];
	    	i += 2;
		}
		else if (strcmp(argv[i], "-out1") == 0) 
		{
	    	out_path1 = argv[i + 1];
	    	i += 2;
		}
		else if (strcmp(argv[i], "-out2") == 0) 
		{
	    	out_path2 = argv[i + 1];
	    	i += 2;
		}
		else if (strcmp(argv[i], "-rho") == 0) 
		{
	    	rho = strtod(argv[i + 1], &end);
	    	i += 2;
		}
		else if (strcmp(argv[i], "-chr") == 0) 
		{   
	    	chr = strtol(argv[i + 1], &end, 10);
	    	if (*end != 0 || chr > 22 || chr < 0) 
			{
				cout << "Incorrect chromosome: " << argv[i + 1] << endl;
				return 0;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-N1") == 0) 
		{
	    	N1 = strtol(argv[i+1], &end, 10);
	    	if (*end != '\0' || N1 <= 0) 
			{
				cout << "Incorrect N1: " << argv[i+1] << endl;
				return 0;
	    	}
	    	if (N1 <= 1000) 
			{
				cout << "Warning: sample size too small for population 1, might" \
		    		" not achieve good performance." << endl;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-N2") == 0) 
		{
	    	N2 = strtol(argv[i+1], &end, 10);
	    	if (*end != '\0' || N2 <= 0) 
			{
				cout << "Incorrect N2: " << argv[i+1] << endl;
				return 0;
	    	}
	    	if (N2 <= 1000) 
			{
				cout << "Warning: sample size too small for population 2, might" \
		    		" not achieve good performance." << endl;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-a") == 0) 
		{
	    	a = strtod(argv[i + 1], &end);
	    	if (*end != '\0' || a < 0) 
			{
				cout << "Incorrect a: " << argv[i + 1] << endl;
				return 0;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-c1") == 0) 
		{
	    	c1 = strtod(argv[i + 1], &end);
	    	if (*end != '\0' || c1 <= 0) 
			{
				cout << "Incorrect c1: " << argv[i + 1] << endl;
				return 0;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-c2") == 0) 
		{
	    	c2 = strtod(argv[i + 1], &end);
	    	if (*end != '\0' || c2 <= 0) 
			{
				cout << "Incorrect c2: " << argv[i + 1] << endl;
				return 0;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-a0k") == 0) 
		{
	    	a0k = strtod(argv[i + 1], &end);
	    	if (*end != '\0' || a0k <= 0) 
			{
				cout << "Incorrect a0k: " << argv[i + 1] << endl;
				return 0;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-b0k") == 0) 
		{
	    	b0k = strtod(argv[i + 1], &end);
	    	if (*end != '\0' || b0k <= 0) 
			{
				cout << "Incorrect b0k: " << argv[i + 1] << endl;
				return 0;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-iter") == 0) 
		{
	    	iter = strtol(argv[i + 1], &end, 10);
	    	if (*end != '\0' || iter <= 0) 
			{
				cout << "Incorrect iteration: " << argv[i + 1] << endl;
				return 0;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-burn") == 0) 
		{
	    	burn = strtol(argv[i + 1], &end, 10);
	    	if (*end != '\0' || burn <= 0) 
			{
				cout << "Incorrect number of iterations: " << argv[i + 1] << endl;
				return 0;
	    	}
	    	if (burn >= iter) 
			{
				cout << "Error: burnin is larger than number of iterations." << endl;
				return 0;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-thin") == 0) 
		{
	    	thin = strtol(argv[i + 1], &end, 10);
	    	if (*end != '\0' || thin <= 0) 
			{
				cout << "Incorrect thin: " << argv[i + 1] << endl;
				return 0;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-n_threads") == 0) 
		{
	    	n_threads = strtol(argv[i + 1], &end, 10);
	    	if (*end != '\0' || n_threads <= 0) 
			{
				cout << "Incorrect number of threads: " << argv[i + 1] << endl;
				return 0;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-r2") == 0) 
		{
	    	r2 = strtod(argv[i + 1], &end);
	    	if (r2 <= 0) 
			{
				cout << "Incorrect r2: " << argv[i+1] << endl;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-opt_llk") == 0) 
		{
	    	opt_llk = strtol(argv[i + 1], &end, 10);
	    	if (opt_llk != 1 && opt_llk != 2) 
			{
				cout << "opt_llk must be in 1 or 2." << endl;
	    	}
	    	i += 2;
		}
		else if (strcmp(argv[i], "-h") == 0) 
		{
	    	print_use();
	    	return 0;
		}
		else if (strcmp(argv[i], "-make_ref") == 0) 
		{
	    	make_ref = 1;
	    	i++;
		}
		else if (strcmp(argv[i], "-mcmc") == 0) 
		{
	    	run_mcmc = 1;
	    	i++;
		}
		else 
		{
	    	cout << "Invalid option: " << argv[i] << endl;
	    	return 0;
		}
    }

    if (make_ref && run_mcmc) 
	{
		cout << "both -make_ref and -mcmc are specified. Please specify one of them." << endl;
		return 0;
    }

    if (!chr) 
	{
		cout << "Invalid chromosome specified." << endl;
		return 0;
    }

    if (ref_dir.empty()) 
	{
		cout << "Did not specify the directory of reference LD." << endl;
		return 0;
    }

    // compute LD
    if (make_ref) 
	{

		if (ref_prefix1.empty()) 
		{
	    	cout << "Did not specify the prefix of the bim file for the reference panel for population 1." << endl;
	    	return 0;
		}
		if (ref_prefix2.empty()) 
		{
	    	cout << "Did not specify the prefix of the bim file for the reference panel for population 2." << endl;
	    	return 0;
		}

		div_block(ref_prefix1, ref_prefix2, ref_dir, chr, n_threads, r2);
    }

    // mcmc 
    if (run_mcmc) 
	{

		if (ss_path1.empty()) 
		{
	    	cout << "Did not specify the path to summary statistics for population 1." << endl;
	    	return 0;
		}
		if (ss_path2.empty()) 
		{
	    	cout << "Did not specify the path to summary statistics for population 2." << endl;
	    	return 0;
		}

		if (out_path1.empty()) 
		{
	    	cout << "Did not specify the path of the output file for population 1." << endl;
	    	return 0;
		}

		if (out_path2.empty()) 
		{
	    	cout << "Did not specify the path of the output file for population 2." << endl;
	    	return 0;
		}

		if (!N1) 
		{
	    	cout << "Did not specify GWAS sample size for population 1." << endl;
	    	return 0;
		}

		if (!N2) 
		{
	    	cout << "Did not specify GWAS sample size for population 2." << endl;
	    	return 0;
		}

		string ref_ldmat1 = ref_dir + "/chr" + \
	       	std::to_string(chr) + "pop1.dat";
		string ref_snpinfo = ref_dir + "/chr" + \
	       std::to_string(chr) + ".snpInfo";
		string ref_ldmat2 = ref_dir + "/chr" + \
	       	std::to_string(chr) + "pop2.dat";
		
		mcmc(ref_snpinfo, ss_path1, ss_path2, valid_bim, ref_ldmat1, ref_ldmat2, \
		 N1, N2, out_path1, out_path2, a, rho, a0k, b0k, iter, burn, thin, n_threads, opt_llk, c1, c2, f_shared);
    }
    return 0;
}

// ./SDPRX -mcmc -ref_dir /Users/blue/Desktop/SDPRX_test/output -ss1 /Users/blue/Desktop/SDPRX_test/output/input/SDPRX_EUR.txt -ss2 /Users/blue/Desktop/SDPRX_test/output/input/SDPRX_EAS.txt -valid /Users/blue/Desktop/SDPRX_test/output/Ukb_imp_v2.bim -N1 71504 -N2 71504 -out ./output -chr 1