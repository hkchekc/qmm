#include <stdio.h>
#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include "include/aiyagari.hpp"

using namespace std;
using namespace aiyagari;
using Eigen::MatrixXd;

int main() {
	cout << "aiyagari start\n";  // check if running
    chrono::steady_clock::time_point tot_begin = chrono::steady_clock::now();
	PARAM p;
	RESULT r;
	init_params(p);
    cout << "start big loop \n";
    find_beta(r, p);
    // the last beta is the correct beta, I can continue
	cout << "end beta loop \n";
	cout << "====================== \n";
    calc_gini(r, p);
 	write_all(r, p, "aiyagari");
	chrono::steady_clock::time_point tot_end = chrono::steady_clock::now();
	cout << "Time difference = " << chrono::duration_cast<chrono::seconds>(tot_end - tot_begin).count() << "[s]" << endl;
	cout << r.beta << "final beta" << endl;
	return 0;
}
