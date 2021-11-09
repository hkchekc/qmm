#include <string>
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <stdio.h>

using Eigen::MatrixXd;
using std::vector;

#ifndef QMM_UTIL_H
#define QMM_UTIL_H
namespace qmm_util{
    float normal_pdf(float x, float m, float s);
    MatrixXd tauchenhussey(unsigned n,double mu,double  rho,double sigma);
    // MatrixXd th_matlab(std::string file, unsigned n);
    void write_file(MatrixXd *arr, std::string dir);
}
#endif
