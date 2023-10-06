#ifndef LD_H
#define LD_H
#include <string>
#include <thread>
using std::string;



void div_block(const string &pfile1, const string &pfile2, \
	const string &out_dir, \
	unsigned chrom, size_t n_thread, double r2);



#endif