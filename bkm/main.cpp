#include <stdio.h>
#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include "include/aiyagari.hpp"
#include "include/bkm.hpp"

using namespace std;
using namespace bkm;

int main() {
	cout << "start bkm \n";
	chrono::steady_clock::time_point tot_begin = chrono::steady_clock::now();
	BKM_PARAM bp;
	BKM_RES br;
	PARAM p;
	RESULT r;
	init_all(r, p, br, bp);
	init_path(p, br, bp);
	cout << "done with init path \n";
    cout << "=============================== \n";
	while (br.path_err > bp.path_crit){
		egm(r, p, br, bp);
		cout << "done with egm \n";
    	cout << "=============================== \n";
        simulate_dist(r, p, br, bp);
		cout << "done with dist sim \n";
    	cout << "=============================== \n";
        get_agg_var_path(p, br, bp);
		cout << br.new_ak_path(bp.TIME-1) << " last aggregate capital \n";
		cout << "done with AK path \n";
    	cout << "=============================== \n";
        get_implied_price_path(p, br, bp);
		cout << "done with price path \n";
    	cout << "=============================== \n";
        update_error(br, p, bp);
		// br.path_err = 0.;
		cout << br.path_err << " new error \n";
    	cout << "=============================== \n";
	}
	write_all(br);
	chrono::steady_clock::time_point tot_end = chrono::steady_clock::now();
	cout << "BKM Time difference = " << chrono::duration_cast<chrono::seconds>(tot_end - tot_begin).count() << "[s]" << endl;
	return 0;
}