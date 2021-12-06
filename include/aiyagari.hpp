#include <Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <cmath>
#include <iostream>
#include <stdio.h>
using Eigen::MatrixXd, Eigen::Tensor;
using std::vector;

#ifndef AIYAGARI_H
#define AIYAGARI_H

struct PARAM{
	const double gamma=2.0, delta=0.06, alpha=0.36;
	const double a_min=0, a_max=24.0;
	const double net_agg_asset=0.0;
	const unsigned NA=1000, NZ=3;
	vector<double> a_grid = vector<double>(NA);
	vector<double> states= vector<double>(NZ);
	MatrixXd markov = MatrixXd(NZ, NZ);
	const double mu=0, rho=0.95, sigma=0.015;
    const double vf_crit = 1e-4, beta_crit = 1e-4, dist_crit=1e-7;
    const double interest = 1.084;
	const double targeted_ak = 4.18582688958;
    const double calibrated_beta = 0.917493;
	const double a_inc = (a_max-a_min)/(double)NA;
};

struct RESULT {
	PARAM param;
    double high_beta=.99900000000000033, low_beta=0.80000000000000000000000033;
    double beta=0.9;  // defined at start of
    double agg_cap = 1.;
    const double agg_lab = 1. ;  // 
	double this_wage=.64;
    double implied_interest;
	Eigen::MatrixXd stat_dist = Eigen::MatrixXd(param.NA*param.NZ, 1);
	MatrixXd a_change_mat= MatrixXd(param.NA*param.NZ,param.NA*param.NZ);
    // for VFI
	MatrixXd vf= MatrixXd(param.NA, param.NZ), new_vf= MatrixXd(param.NA, param.NZ), consum_arr = MatrixXd(param.NA, param.NZ); // exo
	Eigen::MatrixXi pfunc= Eigen::MatrixXi(param.NA, param.NZ);
	double vf_err=100, dist_err=100, beta_err=100; // need reset vor while ausser q.
    // for EGM, not necessary
    MatrixXd expected_vprime = MatrixXd(param.NA, param.NZ);
    MatrixXd implied_consum_arr = MatrixXd(param.NA, param.NZ);
    MatrixXd exo_cash_on_hand = MatrixXd(param.NA, param.NZ);
    MatrixXd implied_cash_on_hand = MatrixXd(param.NA, param.NZ);
	// for part 2 quantiles - 
	MatrixXd marginal_dist = MatrixXd(5, 3), joint_dist = MatrixXd(5, 3), gini_arr = MatrixXd(3, 1);
};

namespace aiyagari{

void init_params(PARAM &p);
void calc_moment(RESULT &r, PARAM p);
void bellman(RESULT &r, PARAM p);
void egm(RESULT &r, PARAM p);
void get_pfunc(RESULT &r, PARAM p);
void interp_linear(Eigen::ArrayXd xval, Eigen::ArrayXd yval, RESULT &r, PARAM p, size_t zi);
void populat_a_change_mat(RESULT &r, PARAM p);
void find_stat_dist(RESULT &r, PARAM p);
void beta_error(RESULT &r, PARAM p);
void write_all (RESULT r, PARAM p, std::string dir);
void find_beta(RESULT &r, PARAM p);
void calc_gini(RESULT &r, PARAM p);
void vfi(RESULT &r, PARAM p);
void get_a_change_mat(MatrixXd &a_mat, const Eigen::MatrixXi pol, const PARAM p);
void solve_quantile(MatrixXd &result_arr, size_t length, MatrixXd x_dist, MatrixXd sorted_arr, size_t which_x, vector<double> sum_arr);
void get_joint_dist(MatrixXd &result_arr, size_t length, MatrixXd joint_dist, MatrixXd joint_arr, size_t which_x, vector<double> sum_arr);
}
#endif
