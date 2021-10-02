#include <stdio.h>
#include <iostream>
#include "include/util.hpp"
#include <Eigen/Dense>
#include <chrono>

using namespace std;
using Eigen::MatrixXd;

int main() {
	cout << "foo\n";
	PARAM p;
	RESULT r;
	init_params(p);
	chrono::steady_clock::time_point tot_begin = chrono::steady_clock::now();
 	while (r.q_err > p.q_crit){
 		r.vf_err = 100;
		chrono::steady_clock::time_point begin = chrono::steady_clock::now();
 		while( r.vf_err > p.vf_crit){
 			bellman(r, p);
 		}
		cout << r.vf_err << endl;
		cout << "done with vfi" << "\n"; 
 		populat_a_change_mat(r, p);
		cout << "done with pop" << "\n"; 
 		find_stat_dist(r, p);
		cout << "done with stat" << "\n"; 
 		q_error(r, p);
		cout << r.q_err << "\n"; 
		chrono::steady_clock::time_point end = chrono::steady_clock::now();
		cout << "Time difference = " << chrono::duration_cast<chrono::seconds>(end - begin).count() << "[s]" << endl;
 	}
 	write_all(r);
	chrono::steady_clock::time_point tot_end = chrono::steady_clock::now();
	cout << "Time difference = " << chrono::duration_cast<chrono::seconds>(tot_end - tot_begin).count() << "[s]" << endl;
	cout << r.q;
	return 0;
}
