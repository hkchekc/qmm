#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <stdio.h>
using Eigen::MatrixXd;
using std::vector;

#ifndef HUGGET_H
#define HUGGET_H

struct PARAM{
	const double beta=0.993362, gamma=3.0;
	const double a_min=-3.0, a_max=24.0;
	const double net_agg_asset=0.0;
	const double vf_crit=1e-6, dist_crit=1e-7, q_crit=1e-3;
	const unsigned NA=1000, NZ=3;
	vector<double> a_grid = vector<double>(NA);
	vector<double> states= vector<double>(NZ);
	MatrixXd markov = MatrixXd(NZ, NZ);
	const double mu=0, rho=0.95, sigma=0.015;
};

struct RESULT {
	PARAM param;
	double high_q=3, low_q=0.5, q=1.5;
	MatrixXd vf= MatrixXd(param.NA, param.NZ), new_vf= MatrixXd(param.NA, param.NZ), consum_arr = MatrixXd(param.NA, param.NZ);
	Eigen::MatrixXd stat_dist = Eigen::MatrixXd(param.NA*param.NZ, 1);
	MatrixXd a_change_mat= MatrixXd(param.NA*param.NZ,param.NA*param.NZ);
	Eigen::MatrixXi pfunc= Eigen::MatrixXi(param.NA, param.NZ);
	double vf_err=100, dist_err=100, q_err=100; // need reset vor while ausser q.
};

// float normal_pdf(float x, float m, float s);
// MatrixXd tauchenhussey(const unsigned n,double mu,double  rho,double sigma);
void init_params(PARAM &p);
void bellman(RESULT &r, PARAM &p);
void par_bellman(RESULT &r, PARAM &p); // currently useless
void populat_a_change_mat(RESULT &r, PARAM &p);
void find_stat_dist(RESULT &r, PARAM &p);
void q_error(RESULT &r, PARAM &p);
void write_all (RESULT r, PARAM p, std::string dir);
#endif
