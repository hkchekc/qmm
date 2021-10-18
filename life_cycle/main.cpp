#include <cmath>
#include <vector>
#include <stdio.h>
#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <gsl/gsl_spline.h>
#include "include/life_cycle.hpp"

using std::vector, std::cout;

int main(void)
{
	cout << "foo\n"; 
	PARAM p;
	RESULT r;
	cout << "foo2\n"; 
	init_params(p);
	cout << "foo3\n"; 
	backward_induction(p, r);
	write_all(r, "life_cycle");	
	return 0;
}
