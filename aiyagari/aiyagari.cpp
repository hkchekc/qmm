#include <stdio.h>
#include <numeric>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_interp.h>
#include <Eigen3/unsupported/Eigen/MatrixFunctions>
#include "include/aiyagari.hpp"
#include "include/util.hpp"

#define EIGEN_USE_BLAS
using namespace std;
using Eigen::MatrixXd, Eigen::MatrixXi;

void aiyagari::init_params(PARAM &p){
	MatrixXd tmp(p.NZ, p.NZ+1);
	tmp = qmm_util::tauchenhussey(p.NZ, p.mu, p.rho, p.sigma);
	for (size_t i =0; i<p.NZ; ++i){
		p.states[i] = tmp(i, p.NZ);
		p.states[i] = exp(p.states[i]);
	}
	p.markov = tmp.block(0,0,p.NZ,p.NZ);
    // logged grids - hard for doing gini-coefficient
 	// double steps = log(p.a_max-p.a_min + 1.)/(float)p.NA;
 	// for (int i=0; (unsigned)i<p.NA; ++i){
 	// 	p.a_grid[i] = exp(steps*i)-1.+p.a_min;
 	// }
    // uniform grids
    double steps = (p.a_max-p.a_min)/(float)p.NA;
 	for (int i=0; (unsigned)i<p.NA; ++i){
 		p.a_grid[i] = steps*i+p.a_min;
 	}
}

void aiyagari::calc_moment(RESULT &r, PARAM p){
    double y = pow(r.agg_cap ,p.alpha)*pow(r.agg_lab , 1. - p.alpha);  //TODO; agg_lab =1 always, can omit
    cout << r.agg_cap << " aggregate capital \n";
    r.implied_interest = p.alpha*y/r.agg_cap +1 - p.delta;  // just check convergence
    r.this_wage = (1- p.alpha)*y/r.agg_lab;  // for next loop
}

void aiyagari::bellman(RESULT &r, PARAM p){
	double consum, util, cond_util, cu, nu, this_wealth, this_a;
	MatrixXd abs_diff(p.NA, p.NZ);
	MatrixXd next_util_arr(p.NA, p.NZ);

	next_util_arr = r.vf*p.markov.transpose(); // not sure if in general show I transpose
	# pragma omp parallel for
	for (size_t aidx=0; aidx<p.NA; ++aidx){
		this_a = p.interest*p.a_grid[aidx];
		for (size_t zidx=0; zidx<p.NZ; ++zidx){
			cond_util = -1000.;
			this_wealth = p.states[zidx]*r.this_wage +this_a;
			for (size_t choice=0; choice<p.NA; ++choice){
				if (this_wealth <p.a_grid[choice]) { // make sure consumption >0
					break; // if this choice is over current state wealth, all later a' are over
				}
				// cu = pow(consum, 1.-p.gamma)/(1.-p.gamma);// current utility
				consum = this_wealth- p.a_grid[choice];
                // TODO: maybe problem with gamma
				cu = 1/consum + 1; // more efficient by putting the negative sign only when calc tot util
				// nu = p.markov.row(zidx).dot(r.vf.row(choice)); // next utility
				nu = next_util_arr(choice, zidx);
				util = r.beta*nu- cu;
				if (util < cond_util){
					continue;
				}
				r.new_vf(aidx, zidx) = util;
				r.pfunc(aidx, zidx) = choice;
				cond_util = util;
				r.consum_arr(aidx, zidx) = consum;
				
			}

		}
	}
	abs_diff = r.new_vf - r.vf;
	// abs_diff = abs_diff.cwiseAbs();
	r.vf_err = max(abs_diff.maxCoeff(), abs(abs_diff.minCoeff()));
	r.vf = r.new_vf;
}

void aiyagari::vfi(RESULT &r, PARAM p){
    r.vf_err = 100; 
	// initial guess of value function (implying expected vprime is also )
	r.vf = MatrixXd::Zero(p.NA, p.NZ);
	for (size_t zidx=0; zidx<p.NZ; ++zidx){
	for (size_t aidx=0; aidx<p.NA; ++aidx){
		r.expected_vprime(aidx, zidx) = pow(r.beta*(p.a_grid[aidx]), -p.gamma) ; // just guess more or less the correct shape
	
	}
	}
    while (r.vf_err > p.vf_crit){
        aiyagari::egm(r, p);
        // aiyagari::bellman(r, p);
    }
	aiyagari::get_pfunc(r, p);
}

void aiyagari::get_pfunc(RESULT &r, PARAM p){
    double a_prime, this_constraint; // to find policy
	int tmp_floor, tmp_ceil;
	for (size_t zidx=0; zidx<p.NZ; ++zidx){
	this_constraint = r.implied_cash_on_hand(0, zidx);
    for (size_t aidx=0; aidx<p.NA; ++aidx){
        if (r.exo_cash_on_hand(aidx, zidx) < this_constraint){
            r.consum_arr(aidx, zidx) = r.exo_cash_on_hand(aidx, zidx); 
            r.pfunc(aidx, zidx) = (int)0;
        } else{
            a_prime = r.exo_cash_on_hand(aidx, zidx) - r.consum_arr(aidx, zidx); // by budget constraint
            tmp_floor = std::floor((a_prime - p.a_min)/p.a_inc); // I think it must always be in grid
            tmp_ceil = std::ceil((a_prime - p.a_min)/p.a_inc); 
            if ( std::abs(a_prime-p.a_grid[tmp_floor])> std::abs(a_prime-p.a_grid[tmp_ceil]) ){
                r.pfunc(aidx, zidx) = tmp_ceil;
            }else{
                r.pfunc(aidx, zidx) = tmp_floor;
            }
            // make sure no outside of range - it is ok, shouldn't matter for eq.
            if ((unsigned) r.pfunc(aidx, zidx) >= p.NA){
                r.pfunc(aidx, zidx) = p.NA - 1;
            }
        }
    }
	}
}

void aiyagari::egm(RESULT &r, PARAM p){
		MatrixXd vfunc_new = MatrixXd::Zero(p.NA, p.NZ);// to be populated in loop

    for (size_t zidx=0; zidx<p.NZ; ++zidx){
		for (size_t aidx=0; aidx<p.NA; ++aidx){
            r.implied_consum_arr(aidx, zidx) = pow(r.beta*r.expected_vprime(aidx, zidx), -1/p.gamma);
            r.implied_cash_on_hand(aidx, zidx) = r.implied_consum_arr(aidx, zidx) + p.a_grid[aidx];
            r.exo_cash_on_hand(aidx, zidx) = r.this_wage*p.states[zidx] + p.interest*p.a_grid[aidx]; 
        }
		aiyagari::interp_linear(r.implied_cash_on_hand.block(0, zidx, p.NA, 1).array(), 
				r.implied_consum_arr.block(0, zidx, p.NA, 1).array(), r, p, zidx);
    }
	r.expected_vprime = MatrixXd::Zero(p.NA, p.NZ);
	// populate expected vprime
	for (size_t nzidx=0; nzidx<p.NZ; ++nzidx){
	for (size_t aidx=0; aidx<p.NA; ++aidx){
		vfunc_new(aidx, nzidx) = p.interest/pow(r.consum_arr(aidx, nzidx), p.gamma);
		for (size_t zidx=0; zidx<p.NZ; ++zidx){
			r.expected_vprime(aidx, zidx) += p.markov(zidx, nzidx) *vfunc_new(aidx, nzidx) ;
		}
	}
	}
	MatrixXd abs_diff = vfunc_new - r.vf;
	r.vf_err =  max(abs_diff.maxCoeff(), abs(abs_diff.minCoeff()));
	r.vf = vfunc_new;
}

void aiyagari::interp_linear(Eigen::ArrayXd xval, Eigen::ArrayXd yval, RESULT &r, PARAM p, size_t zi){
	gsl_interp_accel* accel_ptr = gsl_interp_accel_alloc();
	gsl_interp* interp_ptr;

	interp_ptr = gsl_interp_alloc(gsl_interp_linear, xval.size() ); // gsl_interp_cspline for cubic, gsl_interp_linear for lienar
	gsl_interp_init( interp_ptr, &xval[0], &yval[0], xval.size() );
	for (size_t aidx=0; aidx<p.NA; ++aidx){
		interp_ptr->type->eval( interp_ptr->state,&xval[0], &yval[0] , interp_ptr->size,
		r.exo_cash_on_hand(aidx, zi), accel_ptr, &r.consum_arr(aidx, zi)); // super obscure
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

void aiyagari::populat_a_change_mat(RESULT &r, const PARAM p){
	aiyagari::get_a_change_mat(r.a_change_mat, r.pfunc, p);
}

void aiyagari::get_a_change_mat(MatrixXd &a_mat, const MatrixXi pol, const PARAM p){
	// this matrix is transposed.
	int choice, current_state, next_state;
	a_mat = MatrixXd::Zero(p.NA*p.NZ, p.NA*p.NZ);
	for (size_t aidx=0; aidx<p.NA; ++aidx){
		for (size_t zidx=0; zidx<p.NZ; ++zidx){
			current_state = zidx*p.NA+aidx;
			choice = pol(aidx, zidx);
			for (size_t nzidx=0; nzidx< p.NZ; ++nzidx){
				next_state = nzidx*p.NA + choice;
				a_mat(next_state, current_state) += p.markov(zidx, nzidx); 
			}
		}
	}
// not sure but I think the eigenvector should be the same with or without normalizing
  	float sum_row;
  	for (size_t i = 0; i < p.NA*p.NZ; ++i) {
  		sum_row = a_mat.col(i).sum();
  		for (size_t j = 0; j < p.NA*p.NZ; ++j) {
  		 	a_mat(j,i) /= sum_row;
  		}
  	}
 
}

void aiyagari::find_stat_dist(RESULT &r, PARAM p){
	double uniform  = 1/(double)p.NA/(double)p.NZ;

	Eigen::MatrixXf abs_diff(p.NA*p.NZ, 1);
	Eigen::MatrixXf new_stat_dist(p.NA*p.NZ,1);
	Eigen::MatrixXf stat_dist(p.NA*p.NZ, 1);
	stat_dist.fill(uniform);
	r.dist_err = 100;
	// atempts to imporve performance - if updating too much as once, will lead to instability
	// use float instead of doubles to improve performance
	Eigen::MatrixXf tmp_a_mat = r.a_change_mat.cast <float> ();
	Eigen::MatrixXf tmp_tmp_a_mat = tmp_a_mat*tmp_a_mat;
	// MatrixXd tmp_tmp_tmp_a_mat = tmp_tmp_a_mat*tmp_tmp_a_mat;
	// Eigen::MatrixPower<MatrixXd> Apow(r.a_change_mat);
	// MatrixXd tmp_a_mat = Apow(2.);
	// r.stat_dist = tmp_a_mat*r.stat_dist;
	// for (size_t i=0; i<50; ++i){
	// 	r.stat_dist = r.a_change_mat * r.stat_dist;
	// }
	while (r.dist_err > p.dist_crit){
		new_stat_dist = tmp_tmp_a_mat * stat_dist;
		abs_diff = new_stat_dist - stat_dist;
		r.dist_err = max(abs_diff.maxCoeff(),abs( abs_diff.minCoeff()));
		stat_dist = new_stat_dist;
	}
	r.stat_dist = stat_dist.cast <double> ();
}

void aiyagari::beta_error(RESULT &r, PARAM p){
	r.agg_cap = 0;
	double ratio = 0.5;
	for (size_t aidx=0; aidx < p.NA; ++aidx){
		for (size_t zidx=0; zidx< p.NZ; ++zidx){
			r.agg_cap += p.a_grid[r.pfunc(aidx, zidx)]* r.stat_dist(zidx*p.NA+aidx);
		}
	}
    aiyagari::calc_moment(r, p);
    //TODO: check the direction
	if (r.implied_interest - p.interest < 0.){
		r.high_beta = (1-ratio)*r.high_beta+ratio*r.beta; 
	}else {
		r.low_beta = (1-ratio)*r.low_beta+ratio*r.beta;
	}
	r.beta_err = abs(r.implied_interest - p.interest);
	// r.beta_err = abs(r.agg_cap - p.targeted_ak);
	// r.beta = (1-ratio)*(r.high_beta+r.low_beta)/2 +ratio*r.beta ;
	r.beta = (r.high_beta+r.low_beta)/2;
	cout << r.implied_interest << "interest" << "\n";
}

void aiyagari::find_beta(RESULT &r, PARAM p){
    while (r.beta_err > p.beta_crit){
 		r.vf_err = 100;
		r.new_vf = MatrixXd::Zero(p.NA, p.NZ);
		r.vf = MatrixXd::Zero(p.NA, p.NZ);
		chrono::steady_clock::time_point begin = chrono::steady_clock::now();
 		aiyagari::vfi(r, p);
		cout << r.vf_err << " vf err\n";
		cout << "done with vfi" << "\n"; 
 		aiyagari::populat_a_change_mat(r, p);
		cout << "done with pop" << "\n"; 
 		aiyagari::find_stat_dist(r, p);
		cout << "done with stat" << "\n"; 
 		aiyagari::beta_error(r, p);
		cout << r.beta << " beta \n";
		chrono::steady_clock::time_point end = chrono::steady_clock::now();
		cout << "Time difference = " << chrono::duration_cast<chrono::seconds>(end - begin).count() << "[s]" << endl;
        cout << "====================================\n";
 	}
}

void aiyagari::solve_quantile(MatrixXd &result_arr, size_t length, MatrixXd x_dist, MatrixXd sorted_arr, size_t which_x, vector<double> sum_arr){
	cout << "Calling Quantile \n";
	double cum_x_dist = 0., old_cumsum_x, old_cum_x_dist;
	double cumsum_x = 0.;
	size_t qidx = 0;
	for (size_t xidx=0; xidx<length; ++xidx){
		old_cum_x_dist = cum_x_dist;
		cum_x_dist += x_dist(xidx);
		old_cumsum_x = cumsum_x;
		cumsum_x += x_dist(xidx)*sorted_arr(xidx);
		while (cum_x_dist>=.2){  
			result_arr(qidx, which_x) = (old_cumsum_x+(.2-old_cum_x_dist)*sorted_arr(xidx))/sum_arr[which_x]; 
			cum_x_dist -= .2;
			cumsum_x = cum_x_dist*sorted_arr(xidx) ; // not sure if it is correct, is all the residual this index? 
			qidx += 1;
			// in case the while loop continues
			old_cumsum_x = 0;
			old_cum_x_dist = 0;
		}
		// TODO: This part may break things but on the other handcan capture some precision errors
		// if (xidx == length-1){
		// 	result_arr(qidx, which_x) = cumsum_x/sum_arr[which_x];
		// }
	}
}

void aiyagari::get_joint_dist(MatrixXd &result_arr, size_t length, MatrixXd joint_dist, MatrixXd joint_arr, size_t which_x, vector<double> sum_arr){
	// joint is a 2d array, with shape (a_len, y_len)
	// which_x: 0 is c, 1 is y, 2 is a
	cout << "Calling Joint \n";
	double cum_x_dist = 0., old_cumsum_x=0., old_cum_x_dist=0.;
	double cumsum_x = 0.;
	double unchanged_cumsum=0.;
	double quantile = .2;
	size_t qidx = 0;
	double this_weighted_average;
	MatrixXd x_dist = joint_dist.rowwise().sum();
	for (size_t xidx=0; xidx<length; ++xidx){
		old_cum_x_dist = cum_x_dist;
		cum_x_dist += x_dist(xidx); // dividing is not needed
		old_cumsum_x = cumsum_x;
		this_weighted_average = joint_dist.row(xidx).dot(joint_arr.row(xidx));
		cumsum_x += this_weighted_average;
		unchanged_cumsum += x_dist(xidx);
		while (cum_x_dist>=quantile){  
			result_arr(qidx, which_x) = (old_cumsum_x+(quantile-old_cum_x_dist)*this_weighted_average)/sum_arr[which_x]; 
			cum_x_dist -= quantile;
			// maybe 1 - (quantile-old_cum_x_dist)/cum_x_dist?
			cumsum_x = this_weighted_average; // * cum_x_dist, I don;t know why this is not needed for me it is wrong. TODO
			qidx += 1;
			// in case the while loop continues
			old_cumsum_x = 0;
			old_cum_x_dist = 0;
		}
	}
	cout << unchanged_cumsum << " "<< sum_arr[which_x] << which_x << " match? \n";
}

void aiyagari::calc_gini(RESULT &r, PARAM p){
	vector<double> sum_arr= vector<double>(3), total_area_arr= vector<double>(3),
					 b_area= vector<double>(3);
	MatrixXd stat_dist_2d = MatrixXd(p.NA, p.NZ);
	// build sorted list of consumption 
	// MatrixXd sorted_consum = r.consum_arr.rowwise().sum()/3.;
	MatrixXd sorted_consum_dist = MatrixXd(p.NA*p.NZ, 2); // 1st col
	sorted_consum_dist.col(1) = r.stat_dist;
	unsigned index;
	for (size_t aidx=0; aidx<p.NA; ++aidx){
        for (size_t zidx=0; zidx<p.NZ; ++zidx){
			// populate sorted consumption (not yet sorted)
			index = zidx*p.NA + aidx;
			sorted_consum_dist(index, 0) = r.consum_arr(aidx, zidx);
        }
    }
	sorted_consum_dist = qmm_util::sorted_rows_by_head(sorted_consum_dist);
	MatrixXd sorted_consum = sorted_consum_dist.col(0); 
	cout << "Get Sum \n";
	// Get sum of variables  - c, y, a
	//TODO: remove this part
    for (size_t aidx=0; aidx<p.NA; ++aidx){
        for (size_t zidx=0; zidx<p.NZ; ++zidx){
            index = zidx*p.NA + aidx;
			sum_arr[0] += r.stat_dist(index)*r.consum_arr(aidx, zidx);
			sum_arr[1] += r.stat_dist(index)*p.states[zidx];
            sum_arr[2] += r.stat_dist(index)*p.a_grid[aidx];
			// make stat_dist 2d
			stat_dist_2d(aidx, zidx) = r.stat_dist(index);
        }
    }
	MatrixXd a_dist = stat_dist_2d.rowwise().sum();
	MatrixXd y_dist = stat_dist_2d.colwise().sum();
	MatrixXd c_dist = sorted_consum_dist.col(1);
	cout << "Get GINIs \n";
	// Solve GINI - c, y, a
	// for (size_t idx=0;idx<3; ++idx){
	// 	total_area_arr[idx] = sum_arr[idx]/2.; // triangle and total pop is 1
	// }
	cout << " consumption GINI \n";

	// consumption gini
	double this_consum, cumsum_consum=0.;
	for (size_t cidx=0; cidx<p.NA*p.NZ; ++cidx){
			this_consum = c_dist(cidx)*sorted_consum(cidx); 
			b_area[0] += this_consum*c_dist(cidx)/2.; // triangle
			b_area[0] += cumsum_consum*c_dist(cidx); // the rectangular part
			// add cumsum at last
			cumsum_consum += this_consum;

	}
	cout << " income GINI \n";

	// income gini
	
	double this_income, cumsum_income=0.;
	for (size_t zidx=0; zidx<p.NZ; ++zidx){
			this_income = y_dist(zidx)*p.states[zidx];
			b_area[1] += this_income*y_dist(zidx)/2.; // triangle3
			b_area[1] += cumsum_income*y_dist(zidx); // the rectangular part
			// add cumsum at last
			cumsum_income += this_income;
	}
	cout << " wealth GINI \n";
	// wealth gini
	double this_wealth, cumsum_wealth=0.;
	for (size_t aidx=0; aidx<p.NA; ++aidx){
			this_wealth = a_dist(aidx)*p.a_grid[aidx]; 
			b_area[2] += this_wealth*a_dist(aidx)/2.; // triangle
			b_area[2] += cumsum_wealth*a_dist(aidx); // the rectangular part
			// add cumsum at last
			cumsum_wealth += this_wealth;

	}
	// Get Gini
	for (size_t idx=0;idx<3; ++idx){
		total_area_arr[idx] = sum_arr[idx]/2.; // triangle and total pop is 1
		r.gini_arr(idx) = (total_area_arr[idx]-b_area[idx])/total_area_arr[idx];

	}
	cout << "Get marginal \n";

	// marginal distribution wrt to its  own quantile
	// sum is known
	//consumption
	aiyagari::solve_quantile(r.marginal_dist, p.NZ*p.NA, c_dist, sorted_consum, 0, sum_arr);
	// income -  first retype states
	MatrixXd y_vec = MatrixXd(p.NZ, 1);
	for (size_t yidx=0; yidx<p.NZ; ++yidx){
		y_vec(yidx) = p.states[yidx];
	}
	aiyagari::solve_quantile(r.marginal_dist, p.NZ, y_dist, y_vec, 1, sum_arr);
	// wealth
	MatrixXd a_vec = MatrixXd(p.NA, 1);
	for (size_t aidx=0; aidx<p.NA; ++aidx){
		a_vec(aidx) = p.a_grid[aidx];
	}
	aiyagari::solve_quantile(r.marginal_dist, p.NA, a_dist, a_vec, 2, sum_arr);
	// joint distribution wrt to wealth and marginal wealth
	r.joint_dist.col(2) = r.marginal_dist.col(2);
	// consumption
	aiyagari::get_joint_dist(r.joint_dist, p.NA, stat_dist_2d, r.consum_arr, 0, sum_arr);
	// income
	MatrixXd ay_mat = MatrixXd(p.NA, p.NZ);
	// TODO: 
	for (size_t aidx=0; aidx<p.NA; ++aidx){
		for (size_t zidx=0; zidx<p.NZ; ++zidx){
			ay_mat(aidx, zidx) = p.states[zidx];
		}
	}
	aiyagari::get_joint_dist(r.joint_dist, p.NA, stat_dist_2d, ay_mat, 1, sum_arr); // wrong
	
}


void aiyagari::write_all (RESULT r, PARAM p,string dir){
	const int len = 5;
	string path =dir+"/data_output/";
	string fname[len] = {"vf.txt", "consum.txt", "marginal_dist.txt", "joint_dist.txt", "gini_arr.txt"};
	MatrixXd *pmat[len] = {&r.vf, &r.consum_arr, &r.marginal_dist, &r.joint_dist, &r.gini_arr};
	ofstream fs;
	for (size_t i=0;i <len; ++i){
		fs.open(path+fname[i]);
		if (fs.is_open()){
			fs << *pmat[i];
		}
		fs.close();
	}
	fs.open(path+"a_grid.txt"); //vector
	if (fs.is_open()){
		for (auto &ele: p.a_grid){ 
			fs << ele << endl;
		}
	}
	fs.close();
	fs.open(path+"pfunc.txt"); // eigen matrix
	if (fs.is_open()){
		fs << r.pfunc;
	}
	fs.close();
}
