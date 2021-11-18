#include <Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <vector>
#include <iostream>
#include <stdio.h>
#include "aiyagari.hpp"
using Eigen::MatrixXd, Eigen::Tensor;
using std::vector;

#ifndef BKM_H
#define BKM_H
struct BKM_PARAM{
    const unsigned seed = 1000;
    std::mt19937 rng;
    const unsigned TIME= 500; // time for converging back to SS
    const unsigned long NH = 5000; // number of housholds in simulating 
    const double shock = 0.0165; // consistent with KS. 
    const double path_crit = 1e-3;  // crit for path convergence
    const double rho=0.9; 
};

struct BKM_RES{
    BKM_PARAM bkm_param;
    PARAM param; // from aiyagari
    MatrixXd productivity = MatrixXd(bkm_param.TIME, 1);
    // prices
    MatrixXd r_path = MatrixXd(bkm_param.TIME, 1), new_r_path = MatrixXd(bkm_param.TIME, 1); // interest
    MatrixXd wage_path = MatrixXd(bkm_param.TIME, 1); // wage
    // aggregate moments
    MatrixXd ak_path = MatrixXd(bkm_param.TIME, 1), new_ak_path = MatrixXd(bkm_param.TIME, 1); // agg cap
    MatrixXd agg_c_path = MatrixXd(bkm_param.TIME, 1), agg_output_path = MatrixXd(bkm_param.TIME, 1);
    // distribution path
    MatrixXd dist_path= MatrixXd(param.NA*param.NZ, bkm_param.TIME);
    double path_err = 100; // on price
    // for EGM
    // temp array that don't have time tidx
    MatrixXd expected_vprime = MatrixXd(param.NA, param.NZ);
    MatrixXd exo_consum_arr = MatrixXd(param.NA, param.NZ), exo_cash_on_hand = MatrixXd(param.NA, param.NZ); 
    // arrays with time idx
    MatrixXd implied_consum_arr= MatrixXd(param.NA, param.NZ*bkm_param.TIME) , implied_cash_on_hand= MatrixXd(param.NA, param.NZ*bkm_param.TIME);
    Eigen::MatrixXi pfunc= Eigen::MatrixXi(param.NA, param.NZ*bkm_param.TIME);

};

namespace bkm{
    void init_all(RESULT &r, PARAM &p, BKM_RES &br, BKM_PARAM &bp);
    void init_params(BKM_PARAM &bkm_p);
    void gen_prod_process(BKM_RES &bkm_r,BKM_PARAM bkm_p);
    void run_aiyagari(RESULT &r, PARAM &p); // solving steady state
    void init_path(const PARAM p, BKM_RES &bkm_r, const BKM_PARAM bkm_p);
    void egm(const RESULT r, const PARAM p, BKM_RES &br, const BKM_PARAM bp);
    void simulate_dist(const RESULT r, const PARAM p, BKM_RES &br, const BKM_PARAM bp);
    void get_agg_var_path(const PARAM p, BKM_RES &bkm_r, const BKM_PARAM bkm_p);
    void get_implied_price_path(const PARAM p, BKM_RES &bkm_r, const BKM_PARAM bkm_p);
    void update_error(BKM_RES &bkm_r);
    void write_all(); // get all aggregate paths, to see IRFs
    void simulation(); // to get variance and cov of agg moments
    // helper functions
    size_t get_3d_index(const size_t zidx, const size_t tidx);
    void interp_linear(Eigen::ArrayXd xval, Eigen::ArrayXd yval, BKM_RES &r, PARAM p);
}

#endif
