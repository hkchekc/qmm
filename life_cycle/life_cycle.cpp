#include <stdio.h>
#include <numeric>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <Eigen/Dense>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_interp.h>
#include "include/life_cycle.hpp"

#define EIGEN_USE_BLAS
using std::cout, std::vector;
using Eigen::MatrixXd;

// void read_csv(std::string fname, vector<double> &target){

// }

void init_params(PARAM &p){
	// Markov Matrix
	p.markov <<   0.9586 ,   0.0314 ,   0.0000 ,   0.0000 ,   0.0000 ,   0.0097 ,   0.0003 ,   0.0000 ,   0.0000 ,   0.0000,
    0.0232 ,   0.9137 ,   0.0531 ,   0.0000 ,   0.0000 ,   0.0002 ,   0.0092 ,   0.0005 ,   0.0000 ,   0.000,
    0.0000 ,   0.0474 ,   0.8952 ,   0.0474 ,   0.0000 ,   0.0000 ,   0.0005 ,   0.0090 ,   0.0005 ,   0.000,
    0.0000 ,   0.0000 ,   0.0531 ,   0.9137 ,   0.0232 ,   0.0000 ,   0.0000 ,   0.0005 ,   0.0092 ,   0.000,
    0.0000 ,   0.0000 ,   0.0000 ,   0.0314 ,   0.9586 ,   0.0000 ,   0.0000 ,   0.0000 ,   0.0003 ,   0.009,
    0.9586 ,   0.0314 ,   0.0000 ,   0.0000 ,   0.0000 ,   0.0097 ,   0.0003 ,   0.0000 ,   0.0000 ,   0.000,
    0.0232 ,   0.9137 ,   0.0531 ,   0.0000 ,   0.0000 ,   0.0002 ,   0.0092 ,   0.0005 ,   0.0000 ,   0.000,
    0.0000 ,   0.0474 ,   0.8952 ,   0.0474 ,   0.0000 ,   0.0000 ,   0.0005 ,   0.0090 ,   0.0005 ,   0.000,
    0.0000 ,   0.0000 ,   0.0531 ,   0.9137 ,   0.0232 ,   0.0000 ,   0.0000 ,   0.0005 ,   0.0092 ,   0.000,
    0.0000 ,   0.0000 ,   0.0000 ,   0.0314 ,   0.9586 ,   0.0000 ,   0.0000 ,   0.0000 ,   0.0003 ,   0.0097;
    // Populate States (perm and trans)
	p.states = {0.5324, 0.7415, 1., 1.3487, 1.8784};
	for (size_t j =0;j<p.NTRANS;++j){
	    for (size_t i =0;i<p.NPERM;++i){
		    p.states.push_back(p.states[i]*p.tmp_income[j+1]);
	    }
	}
	// Populate discount factor (incl. survival rate)
	double survive_reduce = (1. - p.last_surv_prob)/p.leave_age;
	for (size_t i=p.working_age-1; i<p.life; ++i){
		p.discount_vec(i) -= survive_reduce*(i+1-p.working_age); 
	}
	p.discount_vec *= p.beta;
}

// void spline_linear(Eigen::ArrayXd xval, Eigen::ArrayXd yval, size_t time_now, RESULT &r, PARAM &p){
// 	gsl_interp_accel* accel_ptr = gsl_interp_accel_alloc();
// 	gsl_spline* spline_ptr;

// 	spline_ptr = gsl_spline_alloc(gsl_interp_linear, xval.size() ); // gsl_interp_cspline for cubic, gsl_interp_linear for lienar
// 	gsl_spline_init( spline_ptr, &xval[0], &yval[0], xval.size() );
// 	cout << "interp_foo\n";
// 	for (size_t aidx=0; aidx<p.NA; ++aidx){
// 		r.exo_pension_consum_arr(aidx, time_now) = 	gsl_spline_eval( spline_ptr, p.tmr_a_grid(aidx) , accel_ptr);

// 	}
// 	gsl_spline_free( spline_ptr );
// 	gsl_interp_accel_free( accel_ptr );
// }

void interp_linear(Eigen::ArrayXd xval, Eigen::ArrayXd yval, size_t time_now, RESULT &r, PARAM &p){
	gsl_interp_accel* accel_ptr = gsl_interp_accel_alloc();
	gsl_interp* interp_ptr;

	interp_ptr = gsl_interp_alloc(gsl_interp_linear, xval.size() ); // gsl_interp_cspline for cubic, gsl_interp_linear for lienar
	gsl_interp_init( interp_ptr, &xval[0], &yval[0], xval.size() );
	for (size_t aidx=0; aidx<p.NA; ++aidx){
		r.exo_pension_consum_arr(aidx, time_now-p.working_age) = interp_ptr->type->eval( interp_ptr->state,&xval[0], &yval[0] , interp_ptr->size,
		r.exo_pension_cash_on_hand(aidx), accel_ptr, &yval[0]); // super obscure
		// try{
		// r.exo_pension_consum_arr(aidx, time_now) = 	gsl_interp_eval( interp_ptr,&xval[0], &yval[0], p.exo_pension_cash_on_hand(aidx) , accel_ptr);
		// }
		// catch(...){
		// 	extrapolation(RESULT &r, Eigen::ArrayXd xval,Eigen::ArrayXd yval, int aidx, double tmr_a)
		// }

	}
	gsl_interp_free( interp_ptr );
	gsl_interp_accel_free( accel_ptr );
}

// no states in retired ages
void retired_endo_grid(PARAM p, RESULT &r, size_t time_now){
	size_t retire_time = time_now - p.working_age;
	for (size_t aidx=0; aidx<p.NA; ++aidx){
			// with retired, no uncertainty v_hat = v
			// Find implied endogenous current consumption from 
			r.implied_pension_consum_arr(aidx, 0) = 1/pow(p.discount_vec(time_now+1)*r.pension_next_value(aidx, 0),
					p.inv_gamma);
			for (size_t j=0;j<10;++j){
				r.implied_cash_on_hand(aidx+j*p.NA, time_now) = 
				r.implied_pension_consum_arr(aidx, 0)+p.tmr_a_grid(aidx);
			}
			// r.implied_cash_on_hand(aidx, time_now) = r.implied_pension_consum_arr(aidx, time_now)+p.tmr_a_grid(aidx);
			r.greatest_constrainted_cash_on_hand(time_now) = r.implied_cash_on_hand(0, time_now);
	}
	// Interpolation, as policy is monotonic and almost linear, it make sense to use linear instead of cubic	
	interp_linear(r.implied_cash_on_hand.block(0,time_now, p.NA, 1).array(), r.implied_pension_consum_arr.array(), time_now, r, p);
	// Make sure no negative consumptions
	for (size_t aidx=0; aidx<p.NA; ++aidx){
	if (p.tmr_a_grid(aidx) <r.greatest_constrainted_cash_on_hand(time_now,0)){
		cout << "foo\n";
		r.exo_pension_consum_arr(aidx, retire_time) = r.exo_pension_cash_on_hand(aidx, 0);
	}
	}
	// Get next period values with exogenous consumption grids
	// we are not using next period value after getting this.consumption, therefore I can flush it.
	for (size_t aidx=0; aidx<p.NA; ++aidx){
		r.pension_next_value(aidx) = p.interest/pow(r.exo_pension_consum_arr(aidx,time_now), p.gamma);
	}
	//flush implied values  (tmp arrays)
	r.implied_pension_consum_arr = MatrixXd::Zero(p.NA, 1);
}

void working_endo_grid(PARAM p, RESULT &r, size_t time_now){

}

void backward_induction(PARAM &p, RESULT &r){
	int pension_capital_start_idx;
	double pension_income=p.lc_profile[p.working_age-1]*p.pension_rate;
	// Define last period exogenous consumptions
	for (size_t i=0; i<p.NA; ++i){
		r.implied_pension_consum_arr(i, 0) = p.interest*p.tmr_a_grid(i)+pension_income;
		r.exo_pension_consum_arr(i, p.last_retire) = r.implied_pension_consum_arr(i, 0);
		pension_capital_start_idx = i*p.NZ;
		for (size_t j=0;j>10;++j){
		r.implied_cash_on_hand(pension_capital_start_idx+j, p.last_idx) = r.exo_pension_consum_arr(i, p.last_retire)+p.tmr_a_grid(i);
		}
		r.pension_next_value(i) = p.interest*pow(r.exo_pension_consum_arr(i,p.last_retire), -p.gamma) ;
	}
	cout << "back_foo\n"; 
	// in last period, exo and endo consumption is defined on the same grid, so no interpolation
	// precalculate retired cash on hand, as there is no time dependent element
	for (size_t i=0; i<p.NA; ++i){
		r.exo_pension_cash_on_hand(i, 0) = p.interest*p.tmr_a_grid(i)+pension_income;
	}
	// Backward induce exogenous consumptions and endogenous COH grids.
	for (size_t time_now=p.working_age+p.leave_age-2; time_now>p.working_age; --time_now){
			cout << time_now << "\n";
			retired_endo_grid(p, r, time_now);
	}
	cout << "back_foo\n"; 

		for (size_t time_now=p.working_age; time_now>0; --time_now){
			working_endo_grid(p, r, time_now);
	}
}

void write_all(RESULT r, std::string dir){
	std::string path =dir+"/data_output/";
	std::ofstream fs;
	fs.open(path+"exo_pension_consum_arr.txt");
	if (fs.is_open()){
		fs << r.exo_pension_consum_arr;
	}
	fs.close();
	fs.open(path+"implied_cash_on_hand.txt");
	if (fs.is_open()){
		fs << r.implied_cash_on_hand;
	}
	fs.close();
}

// void simulation(RESULT &r){
// 	// not implemented
// }