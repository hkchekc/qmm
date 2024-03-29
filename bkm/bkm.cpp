#include <stdio.h>
#include <numeric>
#include <iostream>
#include <random>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <chrono>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_interp.h>
#include "include/aiyagari.hpp"
#include "include/util.hpp"
#include "include/bkm.hpp"

using std::cout;
namespace chrono =  std::chrono;
using Eigen::MatrixXd;

void bkm::init_all(RESULT &r, PARAM &p, BKM_RES &br, BKM_PARAM &bp){
    // Init BKM params and init randoms
    bkm::init_params(bp);
    cout << "done with init params \n";
    cout << "=============================== \n";
    // get steady state equilibrium results
    bkm::run_aiyagari(r, p);
        cout << "done with init aiyagari \n";
    cout << "=============================== \n";
    // Generate productivity process
    bkm::gen_prod_process(br, bp);
            cout << "done with init productivity process \n";
    cout << "=============================== \n";

}

void bkm::init_params(BKM_PARAM &bkm_p){ 
    // set randoms
    bkm_p.rng.seed(bkm_p.seed);
    //TODO: set some steady state results
}

void bkm::gen_prod_process(BKM_RES &bkm_r,BKM_PARAM bkm_p){
    bkm_r.productivity(0) = 1. + bkm_p.shock;  // impulse
    // let the persistant term to 
    for (size_t idx=1; idx<bkm_p.TIME; ++idx){
        bkm_r.productivity(idx) = 1. +bkm_p.rho* (bkm_r.productivity(idx-1)-1);
    }
    bkm_r.productivity(bkm_p.TIME-1) = 1.;  // force the last period to be steady state
}

void bkm::run_aiyagari(RESULT &r ,PARAM &p){
    cout << "aiyagari start\n";  // check if running
    chrono::steady_clock::time_point tot_begin = chrono::steady_clock::now();

	aiyagari::init_params(p);
    cout << "start big loop \n";
    aiyagari::find_beta(r, p);
    // the last beta is the correct beta, I can continue
	cout << "end beta loop \n";
	cout << "====================== \n";
	chrono::steady_clock::time_point tot_end = chrono::steady_clock::now();
	cout << "Time difference = " << chrono::duration_cast<chrono::seconds>(tot_end - tot_begin).count() << "[s] \n";
	cout << r.beta << "final beta \n";
}

void bkm::init_path(const PARAM p, const RESULT r, BKM_RES &bkm_r, const BKM_PARAM bkm_p){
    // initialize a guess path for r, store in BKM_RES
    double deviation = bkm_p.shock/10.; // a guess on what is the first period interest rate (after shock)
    // or maybe better with loop let's see
    bkm_r.r_path.col(0).setLinSpaced(bkm_p.TIME, p.interest+deviation, p.interest);
    bkm_r.wage_path.col(0).setLinSpaced(bkm_p.TIME, r.this_wage+deviation, r.this_wage);
}

size_t bkm::get_3d_index(const size_t zidx, const size_t tidx){
    PARAM this_p; // very stupid, but work
    size_t index;
    index = zidx + tidx*this_p.NZ;  //TODO: extremely error prone
    return index;
}

void bkm::egm(const RESULT r, const PARAM p, BKM_RES &br, const BKM_PARAM bp){
    double this_val, this_constraint;
    // last period
    		cout << "start with egm \n";
    	cout << "=============================== \n";
    for (size_t zidx=0; zidx<p.NZ; ++zidx){
        for (size_t aidx=0; aidx<p.NA; ++aidx){
            br.implied_consum_arr(aidx, get_3d_index(zidx, bp.TIME-1)) = r.consum_arr(aidx, zidx);
            br.implied_cash_on_hand(aidx, get_3d_index(zidx, bp.TIME-1)) = r.implied_interest*p.a_grid[aidx]+ r.this_wage*p.states[zidx];
            br.exo_consum_arr(aidx, get_3d_index(zidx, bp.TIME-1)) = r.consum_arr(aidx, zidx);
            br.exo_cash_on_hand(aidx, get_3d_index(zidx, bp.TIME-1)) = r.implied_interest*p.a_grid[aidx]+ r.this_wage*p.states[zidx];
            // for some reason coh -consum  != k_grid[pol] ~.04 off
        }
        this_constraint = br.implied_cash_on_hand(0, get_3d_index(zidx, bp.TIME-1));
        for (size_t aidx=0; aidx<p.NA; ++aidx){
            if (br.exo_cash_on_hand(aidx, get_3d_index(zidx, bp.TIME-1))< this_constraint){
                br.exo_consum_arr(aidx, get_3d_index(zidx, bp.TIME-1)) = br.exo_cash_on_hand(aidx, get_3d_index(zidx, bp.TIME-1));
            }
            br.pfunc(aidx, get_3d_index(zidx, bp.TIME-1)) = r.pfunc(aidx, zidx);
        }
    }
    br.expected_vprime = MatrixXd::Zero(p.NA, p.NZ);
    for (size_t last_zidx=0; last_zidx<p.NZ; ++last_zidx){
        for (size_t aidx=0; aidx<p.NA; ++aidx){
            this_val = r.implied_interest/pow(br.exo_consum_arr(aidx, get_3d_index(last_zidx, bp.TIME-1)), p.gamma); // for gamma =2
            // cout << this_val << "\n";
            for (size_t zidx=0; zidx<p.NZ; ++zidx){
                br.expected_vprime(aidx, zidx) += p.markov(zidx, last_zidx)* this_val;
            } 
        }
    }

    // backward induction
    unsigned second_index;
    cout << "start backward \n";
    cout << "=============================== \n";
    for (size_t tidx= bp.TIME-2; tidx> 0; --tidx){
        for (size_t zidx=0; zidx<p.NZ; ++zidx){
            second_index = get_3d_index(zidx, tidx);
            for (size_t aidx=0; aidx<p.NA; ++aidx){
                br.implied_consum_arr(aidx, second_index) = pow(r.beta*br.expected_vprime(aidx, zidx), -1./p.gamma);
                br.implied_cash_on_hand(aidx, second_index) = br.implied_consum_arr(aidx, second_index) + p.a_grid[aidx];
                // actually the aidx here is a_now and above is a', but I just conveniently put it in 1 loop
                br.exo_cash_on_hand(aidx, second_index) = p.states[zidx]*br.wage_path(tidx) + br.r_path(tidx)*p.a_grid[aidx];  
            }
            this_constraint = br.implied_cash_on_hand(0, second_index);
            // check where cash constraint is binding
            bkm::interp_linear(br.implied_cash_on_hand.block(0, second_index, p.NA, 1).array(),
             br.implied_consum_arr.block(0, second_index, p.NA, 1).array(), br, p, second_index);
            for (size_t aidx=0; aidx<p.NA; ++aidx){
            if (br.exo_cash_on_hand(aidx, second_index)< this_constraint){
                br.exo_consum_arr(aidx, second_index) = br.exo_cash_on_hand(aidx, get_3d_index(zidx, bp.TIME-1));
            }
        }
            bkm::get_pfunc(p, br, zidx, tidx);
        }

        // get value from last 
        br.expected_vprime = MatrixXd::Zero(p.NA, p.NZ);
        for (size_t last_zidx=0; last_zidx<p.NZ; ++last_zidx){
            for (size_t aidx=0; aidx<p.NA; ++aidx){
                this_val = br.r_path(tidx)/pow(br.exo_consum_arr(aidx, get_3d_index(last_zidx, tidx)), p.gamma);
                for (size_t zidx=0; zidx<p.NZ; ++zidx){
                    br.expected_vprime(aidx, zidx) += p.markov(zidx, last_zidx)* this_val;
                } 
            }
        }
    }
}

void bkm::get_pfunc(const PARAM p, BKM_RES &br, const size_t zidx, const size_t tidx){
    size_t second_index = get_3d_index(zidx, tidx);
    double this_constraint = br.implied_cash_on_hand(0, second_index);
    double a_prime; // to find policy
    int tmp_floor, tmp_ceil; // policy
    for (size_t aidx=0; aidx<p.NA; ++aidx){
        if (br.exo_cash_on_hand(aidx, second_index) < this_constraint){
            br.exo_consum_arr(aidx, second_index) = br.exo_cash_on_hand(aidx, second_index); 
            br.pfunc(aidx, second_index) = (int)0;
        } else{
            a_prime = br.exo_cash_on_hand(aidx, second_index) - br.exo_consum_arr(aidx, second_index); // by budget constraint
            tmp_floor = std::floor((a_prime - p.a_min)/p.a_inc); // I think it must always be in grid
            tmp_ceil = std::ceil((a_prime - p.a_min)/p.a_inc); 
            if ( std::abs(a_prime-p.a_grid[tmp_floor])> std::abs(a_prime-p.a_grid[tmp_ceil]) ){
                br.pfunc(aidx, second_index) = tmp_ceil;
            }else{
                br.pfunc(aidx, second_index) = tmp_floor;
            }
            // make sure no off the grid - it is ok, shouldn't matter for eq.
            if ((unsigned) br.pfunc(aidx, second_index) >= p.NA){
                br.pfunc(aidx, second_index) = p.NA - 1;
            }
        }
    }
}

MatrixXd bkm::youngs_dist(const PARAM p, const BKM_RES br, const size_t tidx){
    // use youngs method to get distribution
    // this consum
    MatrixXd a_mat = MatrixXd(p.NA*p.NZ, p.NA*p.NZ);
    MatrixXd this_consum_arr, this_coh, this_a_prime, this_aprime_pos, this_mod;
    size_t second_index = get_3d_index(0, tidx);
    this_consum_arr = br.exo_consum_arr.block(0, second_index, p.NA, p.NZ);
    this_coh = br.exo_cash_on_hand.block(0, second_index, p.NA, p.NZ);
    this_a_prime = this_coh - this_consum_arr;
    this_aprime_pos = (this_a_prime.array() - p.a_min)/p.a_inc; 
    Eigen::MatrixXi pol_floor = Eigen::MatrixXi(p.NA, p.NZ), pol_ceil = Eigen::MatrixXi(p.NA, p.NZ);
    // # pragma omp parallel for
    for (size_t aidx=0; aidx<p.NA; ++aidx){
        for (size_t zidx=0; zidx<p.NZ; ++zidx){
            int int_ver = (int)this_aprime_pos(aidx, zidx);
            pol_floor(aidx, zidx) = int_ver - (int_ver > this_aprime_pos(aidx, zidx) );
            pol_ceil(aidx, zidx) = int_ver + (int_ver < this_aprime_pos(aidx, zidx) );
        }
    }
    // # pragma omp parallel for
    for (size_t aidx=0; aidx<p.NA; ++aidx){
        for (size_t zidx=0; zidx<p.NZ; ++zidx){
            // catch out of bounds
            if (pol_ceil(aidx, zidx) == p.NA){
                pol_ceil(aidx, zidx) = p.NA -1; 
            }
            if (pol_ceil(aidx, zidx) > p.NA){
                pol_ceil(aidx, zidx) = p.NA -1; 
                pol_floor(aidx, zidx) =  p.NA -1; 
            }
        }
    }
    this_mod = this_aprime_pos - pol_floor.cast<double>();
    MatrixXd change_mat_floor=MatrixXd(p.NA*p.NZ, p.NA*p.NZ), change_mat_ceil=MatrixXd(p.NA*p.NZ, p.NA*p.NZ);
    aiyagari::get_a_change_mat(change_mat_floor, pol_floor, p);
    aiyagari::get_a_change_mat(change_mat_ceil, pol_ceil, p);
    
    MatrixXd this_floor_weight = this_mod.replicate(p.NZ, p.NA); // wrong?
    // this_floor_weight.resize(p.NA*p.NZ, p.NA*p.NZ);
    MatrixXd this_ceil_weight = (1.-this_floor_weight.array());
    a_mat = this_floor_weight.cwiseProduct(change_mat_floor) + this_ceil_weight.cwiseProduct(change_mat_ceil);  // not same size
    return a_mat;
}

void bkm::simulate_dist(const RESULT r, const PARAM p, BKM_RES &bkm_r, const BKM_PARAM bkm_p, const bool young){
    bkm_r.new_dist_path.col(0) = r.stat_dist;
    MatrixXd last_a_mat = MatrixXd(p.NA*p.NZ, p.NA*p.NZ);
    Eigen::MatrixXi last_policy;
    for (size_t tidx= 1; tidx< bkm_p.TIME; ++tidx){
         last_policy = bkm_r.pfunc.block(0, get_3d_index(0, tidx), p.NA, p.NZ); 
        if (!young){
            last_a_mat.fill(0);
            aiyagari::get_a_change_mat(last_a_mat , last_policy, p); 
        } else {
            last_a_mat = bkm::youngs_dist(p, bkm_r, tidx); 
        }
        MatrixXd last_dist = bkm_r.new_dist_path.col(tidx-1);
        bkm_r.new_dist_path.col(tidx) = last_a_mat*last_dist; 
    }

}

void bkm::interp_linear(Eigen::ArrayXd xval, Eigen::ArrayXd yval, BKM_RES &br, const PARAM p, size_t zi){
	gsl_interp_accel* accel_ptr = gsl_interp_accel_alloc();
	gsl_interp* interp_ptr;
	interp_ptr = gsl_interp_alloc(gsl_interp_linear, xval.size() ); // gsl_interp_cspline for cubic, gsl_interp_linear for lienar
	gsl_interp_init( interp_ptr, &xval[0], &yval[0], xval.size() );

    // some obsolete version of GSL support extrapolation, the current one don't
    for (size_t aidx=0; aidx<p.NA; ++aidx){
	    interp_ptr->type->eval( interp_ptr->state,&xval[0], &yval[0] , interp_ptr->size,
	    br.exo_cash_on_hand(aidx, zi), accel_ptr, &br.exo_consum_arr(aidx, zi)); // super obscure
    }
	gsl_interp_free( interp_ptr );
	gsl_interp_accel_free( accel_ptr );
}

void bkm::get_agg_var_path(const RESULT r, PARAM p, BKM_RES &bkm_r, const BKM_PARAM bkm_p){
    // we ignore the income process and add up the amount of capital
    // get_agg_var_path
    double this_a_dist;
    unsigned index;
    bkm_r.new_ak_path = MatrixXd::Zero(bkm_p.TIME, 1); // because later increment, be safe
    for (size_t tidx= 0; tidx< bkm_p.TIME; ++tidx){
        for (size_t aidx= 0; aidx< p.NA; ++aidx){
            this_a_dist = 0;
            // Summing up 
            for (size_t zidx= 0; zidx< p.NZ; ++zidx){
                index = zidx*p.NA+aidx;
                this_a_dist += bkm_r.new_dist_path(index, tidx);
            }
            bkm_r.new_ak_path(tidx, 0) += this_a_dist*p.a_grid[aidx];
        }
        // To avoid infinite interest in transition
        if (bkm_r.new_ak_path(tidx, 0)==0.){
            bkm_r.new_ak_path(tidx, 0) = .01;
        }
        // can calculate when things converge
        bkm_r.agg_output_path(tidx, 0) = bkm_r.productivity(tidx)*pow(bkm_r.new_ak_path(tidx, 0), p.alpha);
        if (tidx != 0){
            bkm_r.agg_invest_path(tidx, 0) = (bkm_r.new_ak_path(tidx, 0)- bkm_r.new_ak_path(tidx-1, 0))+p.delta*bkm_r.new_ak_path(tidx-1, 0);
        } else{
            bkm_r.agg_invest_path(tidx, 0) = (bkm_r.new_ak_path(tidx, 0)- r.agg_cap)+p.delta*r.agg_cap;
        }
        bkm_r.agg_c_path(tidx, 0) = bkm_r.agg_output_path(tidx, 0) - bkm_r.agg_invest_path(tidx, 0);
    }
}

void bkm::get_implied_price_path(const PARAM p, BKM_RES &bkm_r, const BKM_PARAM bkm_p){
    // check convergence of wage too?
    for (size_t tidx= 0; tidx< bkm_p.TIME; ++tidx){
        bkm_r.wage_path(tidx) = (1-p.alpha)*bkm_r.productivity(tidx)*pow(bkm_r.new_ak_path(tidx, 0), p.alpha);
        bkm_r.new_r_path(tidx) = 1.+p.alpha*bkm_r.productivity(tidx)/pow(bkm_r.new_ak_path(tidx, 0), (1.-p.alpha)) - p.delta;
        if ( !std::isfinite(bkm_r.new_r_path(tidx, 0)) ){
            bkm::write_all(bkm_r);
            throw std::invalid_argument( "received inf value" );
        }
    }

}

void bkm::update_error(BKM_RES &bkm_r, PARAM p, BKM_PARAM bp){
    	MatrixXd abs_diff = bkm_r.new_r_path - bkm_r.r_path;
		bkm_r.path_err = std::max(abs_diff.maxCoeff(),abs( abs_diff.minCoeff()));
        bkm_r.dist_path = bkm_r.new_dist_path;
        double ratio = 0.99*(bkm_r.path_err / (bkm_r.path_err+1e-3)); // TODO: relate this to abs_diff, if abs_diff is small, more weight on new  
        bkm_r.ak_path = (1.-ratio)*bkm_r.new_ak_path+ratio*bkm_r.ak_path;
        bkm_r.r_path = .1*bkm_r.new_r_path+.9*bkm_r.r_path;

}

void bkm::write_all(BKM_RES br){
    const int len = 8;
    std::string dir = "bkm";
	std::string path =dir+"/data_output/";
	std::string fname[len] = {"prod_path.txt", "c_path.txt", "ak_path.txt", "r_path.txt", "wage_path.txt", "i_path.txt", "dist_path.txt", "y_path.txt"};
    MatrixXd *pmat[len] = {&br.productivity, &br.agg_c_path, &br.new_ak_path, &br.new_r_path, &br.wage_path, &br.agg_invest_path, &br.dist_path, &br.agg_output_path};	for (size_t i=0;i <len; ++i){
        qmm_util::write_file(pmat[i], path+fname[i]) ; 
    }
    qmm_util::write_file(&br.implied_consum_arr, path+"implied_consum.txt"); 
    qmm_util::write_file(&br.implied_cash_on_hand, path+"implied_cash_on_hand.txt"); 
}