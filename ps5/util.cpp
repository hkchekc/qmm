#include <stdio.h>
#include <numeric>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include "include/util.hpp"

using namespace std;
using Eigen::MatrixXd, Eigen::MatrixXi, Eigen::VectorXd;

float normal_pdf(float x, float m, float s)
{
    static const float inv_sqrt_2pi = 1/M_SQRT2/sqrt(M_PI);
    float a = (x - m) / s;
    return inv_sqrt_2pi / s * exp(-0.5f * a * a);
}

 MatrixXd tauchenhussey(const unsigned n,double mu,double  rho,double sigma){
	//     Adapted from Martin Floden, Stockholm School of Economics
	//     January 2007 (updated August 2007)
	//
	//     This procedure is an implementation of Tauchen and Hussey's
	//     algorithm, Econometrica (1991, Vol. 59(2), pp. 371-396)
	double baseSigma = (.5+rho/4)*sigma+(.5-rho/4)*sigma/sqrt(1.0-pow(rho, 2.0));
	MatrixXd m(n, n+1);
	// create grids
	// gauss hermite nodes and weights
 	unsigned maxit = 10;
 	static double pim4 = .7511255444649425, crit = 3e-14;
 	vector<double> x_arr(n), w_arr(n);
 	int len = static_cast<int>(floor(n+1)/2);
 	double z;
 	for (int i=0; i<len; i++){
 		double pp=0;
 		if (i==0){
			z = sqrt((2*n+1)-1.85575*pow((2*n+1), -0.16667));
		} else if (i==1){ 
			z = z - 1.14*(pow(n,0.426))/z;
		}else if (i==2){ 
			z = 1.86*z - 0.86*x_arr[0];
		}else if (i==3){ 
			z = 1.91*z - 0.91*x_arr[1];
		}else{ 
			z = 2*z - x_arr[i-1];
 		}
 		for (int j=0; j<maxit; j++){
 			double p1=pim4, p2=0.0, p3=0.0;
 			for (int k=0; k<n; k++){
 				p3 = p2;
 				p2 = p1;
 				p1 = z*sqrt(2/(k+1))*p2 - sqrt(k/(k+1))*p3;
 			}
 			pp = sqrt(2.0*n)*p2;
 			double z1 = z;
 			z = z1 - p1/pp;
 			if (abs(z-z1) < crit){ break;}}
 		x_arr[i] = z;
 		x_arr[n-i-1] = -z;
 		w_arr[i] = 2./pp/pp;
 		w_arr[n-i-1] = w_arr[i];
 	}
 	reverse(x_arr.begin(), x_arr.end());
 	// end of gauss hermite
 	// gauss normal
 	for (int i=0; i<n; i++){
 		x_arr[i] = x_arr[i]*baseSigma*M_SQRT2 + mu;
 		w_arr[i] /= sqrt(M_PI);
 		}
 	// end gauss normal
 	// Calculate transition probabilities
 	double ezPrime;
 	for (int i = 0; i < n; i++) {
 		for (int j = 0; j < n; j++) {
 			ezPrime = (1-rho)*mu + rho*x_arr[i];
 			m(i,j) = w_arr[j] * normal_pdf(x_arr[j], ezPrime, sigma)/normal_pdf(x_arr[j], mu, baseSigma);
 		}
 	}
 	//normalize probabilities
 	float sum_row;
 	for (int i = 0; i < n; i++) {
 		sum_row = m.row(i).sum();
 		for (int j = 0; j < n; j++) {
 		 	m(i,j) /= sum_row;
 		}
 	}
 	for (int i=0; i<n ; i++) {
 		m(i,n) = x_arr[i];
 	}
	return m;
} 


MatrixXd th_matlab(string file,const unsigned n){
	ifstream f;
	f.open(file);
	MatrixXd m(n,n+1);
	for(int i=0; i<n; i++){
		for(int j=0;j<n+1;j++){
			f >> m(i,j);}}
	return m;
}

void init_params(PARAM &p){
	MatrixXd tmp(p.NZ, p.NZ+1);
	tmp = tauchenhussey(p.NZ, p.mu, p.rho, p.sigma);
 	for (int i =0; i<p.NZ; i++){
 		p.states[i] = tmp(i, p.NZ);
 		p.states[i] = exp(p.states[i]);
 	}
 	p.markov = tmp.block(0,0,p.NZ,p.NZ);
	p.markov << 0.9702,    0.0298,    0.0000,
    		   0.0252,    0.9497,    0.0252,
    		   0.0000,    0.0298,    0.9702;
	p.states = {0.9598, 1., 1.0419};
 	double steps = log(p.a_max-p.a_min + 1.)/(float)p.NA;
 	for (int i=0; i<p.NA;i++){
 		p.a_grid[i] = exp(steps*i)-1.+p.a_min;
 	}
}

void bellman(RESULT &r, PARAM &p){
	double consum, util, cond_util, cu, nu;
	MatrixXd abs_diff(p.NA, p.NZ);

	for ( int aidx=0; aidx<p.NA;aidx++){
		for (int zidx=0; zidx<p.NZ;zidx++){
			cond_util = -1e10;
			for (int choice=0; choice<p.NA; choice++){
				consum = p.states[zidx] + p.a_grid[aidx] - r.q * p.a_grid[choice];
				if (consum > 0.){
					cu = pow(consum, 1.-p.gamma)/(1.-p.gamma);// current utility
					nu = p.markov.row(zidx).transpose().dot(r.vf.row(choice)); // next utility
					util = cu+ p.beta*nu;
					if (util > cond_util){
						r.new_vf(aidx, zidx) = util;
						r.pfunc(aidx, zidx) = choice;
						cond_util = util;
						r.consum_arr(aidx, zidx) = consum;
					}
				}
			}
		}
	}
	abs_diff = r.new_vf - r.vf;
	// abs_diff = abs_diff.cwiseAbs();
	r.vf_err = max(abs_diff.maxCoeff(), abs(abs_diff.minCoeff()));
	r.vf = r.new_vf;
	cout << r.vf_err  << endl;
	cout << "====================================\n";
}

void populat_a_change_mat(RESULT &r, PARAM &p){
	int choice, current_state, next_state;

	r.a_change_mat = MatrixXd::Zero(p.NA*p.NZ, p.NA*p.NZ);
	for (int aidx=0; aidx<p.NA;aidx++){
		for (int zidx=0; zidx<p.NZ;zidx++){
			current_state = zidx*p.NA+aidx;
			choice = r.pfunc(aidx, zidx);
			for (int nzidx=0; nzidx< p.NZ; nzidx++){
				next_state = nzidx*p.NA + choice;
				r.a_change_mat(current_state, next_state) += p.markov(zidx, nzidx); 
			}
		}
	}
}

void find_stat_dist(RESULT &r, PARAM &p){
	double uniform  = 1/(double)p.NA/(double)p.NZ;

	VectorXd abs_diff(p.NA*p.NZ);
	VectorXd new_stat_dist(p.NA*p.NZ);
	r.stat_dist = VectorXd::Constant(p.NA*p.NZ, uniform);
	r.dist_err = 100;
	for (int i=0; i<1000; i++){
		r.stat_dist = r.a_change_mat * r.stat_dist;
	}
	while (r.dist_err > p.dist_crit){
		new_stat_dist = r.a_change_mat * r.stat_dist;
		abs_diff = new_stat_dist - r.stat_dist;
		r.dist_err = abs_diff.cwiseAbs().maxCoeff();
		r.stat_dist = new_stat_dist;
		cout << r.dist_err << "\n";
	}
}

void q_error(RESULT &r, PARAM &p){
	double net_asset;
	net_asset = 0;
	for (int aidx=0; aidx < p.NA; aidx++){
		for (int zidx=0; zidx< p.NZ; zidx++){
			net_asset += p.a_grid[r.pfunc(aidx, zidx)]* r.stat_dist(zidx*p.NA+aidx);
		}
	}
	if (net_asset > 0.){
		r.high_q =r.q;
	}else {
		r.low_q = r.q;
	}
	r.q_err = abs(r.high_q - r.low_q);
	r.q = (r.high_q+r.low_q)/2;
}
