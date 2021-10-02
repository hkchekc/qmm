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
