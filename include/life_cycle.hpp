#include <Eigen/Dense>
#include <vector>
using Eigen::MatrixXd;

#ifndef LC_H
#define LC_H
struct PARAM{
	const double beta = 0.94, interest=1.04, gamma=1.5, pension=0.6, last_surv_prob = 0.92, inv_gamma=1./gamma;
	const unsigned working_age = 40, leave_age = 20, life=working_age+leave_age, last_idx=life-1,
	 last_retire=leave_age-1, last_working=working_age-1;
	const double leave_rate = 0.92, pension_rate=.6;
	const unsigned NA=1000, NPERM=5, NTRANS=2;
	const unsigned NZ = NPERM*NTRANS;
	const double a_min = 0. , a_max=80.;
	const Eigen::VectorXd a_grid = Eigen::VectorXd::LinSpaced(NA, a_min, a_max), tmr_a_grid= Eigen::VectorXd::LinSpaced(NA, a_min, a_max);
	std::vector<double> states;
	const std::vector<double> tmp_income = {1, .4}, tmp_income_prob = {.99, .01};
	const std::vector<double> lc_profile =  {1.0, 1.078655243, 1.153599438, 1.224204445,
	1.290025519, 1.350793773, 1.406401226, 1.456880637, 1.502382228, 1.543149102, 1.579492827, 1.611770268, 1.640362411,
	 1.665655601, 1.688025327, 1.707822556, 1.725362408, 1.740914935, 1.754697715, 1.766869965, 1.777527938, 1.786701388,
	  1.794351006, 1.800366761, 1.804567228, 1.806700042, 1.806443726, 1.803411231, 1.797155613, 1.787178287, 1.77294038,
	   1.753877621, 1.729419149, 1.699010461, 1.662140451, 1.618372153, 1.567376371, 1.508966833, 1.443134956, 1.370081737};
	Eigen::VectorXd discount_vec = Eigen::VectorXd::Ones(life);
	MatrixXd markov = MatrixXd(NZ, NZ);
	MatrixXd profile = Eigen::VectorXd(life);
	const double mu=0, rho=0.95, sigma=0.015;
};

struct RESULT {
	PARAM param;
	MatrixXd pension_next_value= MatrixXd::Zero(param.NA, 1), working_next_value_sum=MatrixXd::Zero(param.NA, 1); // the implied t-1 period consumption 
	MatrixXd implied_working_consum_arr= MatrixXd::Zero(param.NA, param.NZ), 
	implied_pension_consum_arr = MatrixXd::Zero(param.NA, 1);
	MatrixXd exo_working_consum_arr= MatrixXd::Zero(param.NA*param.NZ, param.working_age), 
			exo_pension_consum_arr = MatrixXd::Zero(param.NA, param.leave_age);
	MatrixXd implied_cash_on_hand = MatrixXd(param.NA*param.NZ, param.life), exo_pension_cash_on_hand = MatrixXd(param.NA, 1)
	, exo_working_cash_on_hand = MatrixXd(param.NA*param.NZ, 1);
	MatrixXd greatest_constrainted_cash_on_hand = MatrixXd::Zero(param.life, 1);
};

// struct SIM_PARAM{
// 	// Not implemented
// 	PARAM param;
// 	const unsigned init_income=3; // position
// 	const double init_a_mean = -2.5, init_a_var=4;
// 	const double init_a_max=exp(init_a_mean+3*init_a_var);
// 	const unsigned NHOUSE = 5000;
// 	MatrixXd perm_shock_arr = MatrixXd(NHOUSE, param.life);
// };

void init_params(PARAM &p); // read csv and populate leave vector
void retired_endo_grid(PARAM p, RESULT &r, size_t time_now);
void backward_induction(PARAM &p, RESULT &r);
void simulation(PARAM &p, RESULT &r);
void write_all(RESULT r, std::string dir);
#endif
