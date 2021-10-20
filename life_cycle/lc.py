import numpy as np
import math
import matplotlib.pyplot as plt
from numba import jit
from scipy import interpolate
from pandas import Series

# parameters
profile = np.genfromtxt("input/lc_profile.csv", delimiter=",")
profile = profile[1:, 1]
beta = .94
interest = 1.04
work_time = 40
retire_time = 20
last_surv_rate = 0.92
life_time = work_time+retire_time
discount_arr = np.ones(life_time)
survive_reudction = (1-last_surv_rate)/retire_time
for t in range(work_time, life_time):
    discount_arr[t] = discount_arr[t-1] - survive_reudction
discount_arr *= beta
gamma = 1.5
inv_gamma = 1/gamma
pension_rate = 0.6
NA = 1000
a_min = 0.
a_max = 80
tmr_a_grid = np.linspace(a_min, a_max, NA)
markov = np.array([[0.9586, 0.0314, 0.0000, 0.0000, 0.0000, 0.0097, 0.0003, 0.0000, 0.0000, 0.0000],
                    [0.0232, 0.9137, 0.0531, 0.0000, 0.0000, 0.0002, 0.0092, 0.0005, 0.0000, 0.000],
                    [0.0000, 0.0474, 0.8952, 0.0474, 0.0000, 0.0000, 0.0005, 0.0090, 0.0005, 0.000],
                    [0.0000, 0.0000, 0.0531, 0.9137, 0.0232, 0.0000, 0.0000, 0.0005, 0.0092, 0.000],
                    [0.0000, 0.0000, 0.0000, 0.0314, 0.9586, 0.0000, 0.0000, 0.0000, 0.0003, 0.009],
                    [0.9586, 0.0314, 0.0000, 0.0000, 0.0000, 0.0097, 0.0003, 0.0000, 0.0000, 0.000],
                    [0.0232, 0.9137, 0.0531, 0.0000, 0.0000, 0.0002, 0.0092, 0.0005, 0.0000, 0.0000],
                    [0.0000, 0.0474, 0.8952, 0.0474, 0.0000, 0.0000, 0.0005, 0.0090, 0.0005, 0.000],
                    [0.0000, 0.0000, 0.0531, 0.9137, 0.0232, 0.0000, 0.0000, 0.0005, 0.0092, 0.0002],
                    [0.0000, 0.0000, 0.0000, 0.0314, 0.9586, 0.0000, 0.0000, 0.0000, 0.0003, 0.0097]])
# markov = np.array([[0.9586, 0.0314, 0.0000, 0.0000, 0.0000],
#                     [0.0232, 0.9137, 0.0531, 0.0000, 0.0000],
#                     [0.0000, 0.0474, 0.8952, 0.0474, 0.0000],
#                     [0.0000, 0.0000, 0.0531, 0.9137, 0.0232],
#                     [0.0000, 0.0000, 0.0000, 0.0314, 0.9586]])
NZ = markov.shape[0]
markov_t = markov.T
states = np.array([0.5324, 0.7415, 1., 1.3487, 1.8784, 0.5324, 0.7415, 1., 1.3487, 1.8784])
trans_shock = .4
trans_shock_prob = 0.01
states[5:] *= trans_shock

# states = np.array([0.5324, 0.7415, 1., 1.3487, 1.8784])


#result arrays
exo_consum_arr = np.zeros((NA, NZ, life_time))
implied_cash_on_hand = np.zeros((NA, NZ,life_time))
greatest_constrainted_cash_on_hand = np.zeros(life_time)
# tmp array
implied_consum_arr = np.zeros((NA, NZ))
next_value = np.zeros((NA, NZ))

# start with last period
pension_income = profile[work_time-1]*pension_rate

# Populate last period
# precalculate retired cash on hand, as there is no time dependent element
exo_pension_cash_on_hand = interest*tmr_a_grid+pension_income
for aidx in range(NA):
    # I just define the init coh as correct grid
    implied_consum_arr[aidx, :] = exo_pension_cash_on_hand[aidx] + 0.01
    exo_consum_arr[aidx, :, life_time-1] = implied_consum_arr[aidx, :]
    next_value[aidx, :] = interest/(exo_consum_arr[aidx, 0, life_time-1]**gamma)


# leave it out and calc in loop
exo_working_cash_on_hand = np.zeros((NA, NZ,work_time))

# Backward induction
for tidx in range(life_time-2, -1, -1):
    if tidx >= work_time:
        for aidx in range(NA):
            implied_consum_arr[aidx, :] = 1/((discount_arr[tidx+1]*next_value[aidx, 0])**inv_gamma)
            if implied_consum_arr[aidx, 0] < 0.00001:
                print(implied_consum_arr[aidx, 0], next_value[aidx, 0], tidx)
                raise ValueError
            implied_cash_on_hand[aidx, :, tidx] = implied_consum_arr[aidx, :]+tmr_a_grid[aidx]
        greatest_constrainted_cash_on_hand[tidx] = implied_cash_on_hand[0, 0, tidx]
        # first populate the first state and copy it to other state just for completeness
        # exo_consum_arr[:, 0, tidx] = np.interp(exo_pension_cash_on_hand,
        #                                        implied_cash_on_hand[:, 0, tidx], implied_consum_arr[:, 0])
        interpolate_f = interpolate.interp1d(implied_cash_on_hand[:, 0, tidx], implied_consum_arr[:, 0],
                                             fill_value="extrapolate", kind="linear")
        exo_consum_arr[:, 0, tidx] = interpolate_f(exo_pension_cash_on_hand)
        # Make sure no negative consumption or over asset constraint
        mx = np.ma.masked_less_equal(exo_pension_cash_on_hand, greatest_constrainted_cash_on_hand[tidx]).mask
        exo_consum_arr[:, 0, tidx][mx] = exo_pension_cash_on_hand[mx]
        for i in range(1, NZ):
            exo_consum_arr[:, i, tidx] = exo_consum_arr[:, 0, tidx]
        # not using next period value after getting this.consumption, therefore I can flush it here.
        next_value[:, 0] = interest/(exo_consum_arr[:, 0, tidx]**gamma)
        for i in range(1, NZ):
            next_value[:, i] = next_value[:, 0]
    else:
        for zidx in range(NZ):
            for aidx in range(NA):
                implied_consum_arr[aidx, zidx] = 1 / ((discount_arr[tidx + 1] * next_value[aidx, zidx]) ** inv_gamma)
                if implied_consum_arr[aidx, zidx] < 0.00001:
                    print(implied_consum_arr[aidx, zidx], next_value[aidx, zidx], tidx)
                    raise ValueError
                implied_cash_on_hand[aidx, zidx, tidx] = implied_consum_arr[aidx, zidx] + tmr_a_grid[aidx]
            # see where it binds
            greatest_constrainted_cash_on_hand[tidx] = implied_cash_on_hand[0, zidx, tidx]
            exo_working_cash_on_hand[:, zidx, tidx] = interest*tmr_a_grid+states[zidx]*profile[tidx]
            # exo_consum_arr[:, zidx, tidx] = np.interp(exo_working_cash_on_hand[:, zidx, tidx],
            #                                        implied_cash_on_hand[:, zidx, tidx], implied_consum_arr[:, zidx])
            interpolate_f = interpolate.interp1d(implied_cash_on_hand[:, zidx, tidx], implied_consum_arr[:, zidx],
                                                 fill_value="extrapolate", kind="linear")
            exo_consum_arr[:, zidx, tidx] = interpolate_f(exo_working_cash_on_hand[:, zidx, tidx])
            # Make sure no negative consumption or over asset constraint
            mx = np.ma.masked_less_equal(exo_working_cash_on_hand[:, zidx, tidx], greatest_constrainted_cash_on_hand[tidx]).mask
            exo_consum_arr[:, zidx, tidx][mx] = exo_working_cash_on_hand[:, zidx, tidx][mx]
        # not using next period value after getting this.consumption, therefore I can flush it here.
        next_value = np.zeros((NA, NZ))
        tmp_consum = exo_consum_arr[:, :, tidx]
        # tmp_consum[tmp_consum <= 0] = 0.0000000001
        for last_zidx in range(NZ):
            for aidx in range(NA):
                this_val = interest /(tmp_consum[aidx, last_zidx] ** gamma)
                for zidx in range(NZ):
                    next_value[aidx, zidx] += markov[zidx, last_zidx] * this_val  # error-prone


np.savetxt("data_output/consumption_policy.csv", exo_consum_arr[:,:,work_time], delimiter=",")
np.savetxt("data_output/coh.csv", exo_working_cash_on_hand[:,:,work_time-1], delimiter=",")

# simulation
@jit(nopython=True, parallel=True)
def map_to_nearest_grid(income_grids, income_processes):
    approx_income = np.zeros(income_processes.shape)
    for hi, household_income in enumerate(income_processes):
        for i, income in enumerate(household_income):
            idx = np.abs(income_grids-income).argmin()
            approx_income[hi, i] = idx
    return approx_income.astype(np.int8)  # returning indexes of income... or is it better to return both


rho_arr = np.linspace(.7,.995, 10)

rho = .97
# for rho in rho_arr:
rng = np.random.default_rng(seed=10000)
init_income = 1
NHOUSE = 5000
init_cap_mean = -2.5
init_cap_se = 2
initial_capital = np.exp(rng.normal(init_cap_mean, init_cap_se, NHOUSE))
init_cap_cap = math.exp(3*init_cap_se+init_cap_mean)
initial_capital[initial_capital > init_cap_cap] = init_cap_cap

# worktime income process
perm_shock_mean = 0.
perm_shock_se = math.sqrt(0.015)
income_process = np.ones((NHOUSE, life_time))  # initial period is one, don't change
perm_shock_mat = np.zeros((NHOUSE, life_time))
this_total_income = income_process[:, 0]
for tidx in range(1, work_time):
    this_income = rho*np.log(this_total_income)# no profile as it will be part of the policy as tidx
    this_shock = rng.normal(perm_shock_mean, perm_shock_se, NHOUSE)
    perm_shock_mat[:, tidx] = this_shock
    tmp_shock = rng.choice([1, trans_shock], NHOUSE,  p=[1-trans_shock_prob, trans_shock_prob])
    this_total_income = np.exp(this_income+this_shock)
    # this_total_income[this_total_income< 0] = 1
    income_process[:, tidx] = tmp_shock* this_total_income
income_process[:, work_time:] = pension_income

for tidx in range(1, work_time):
    income_process[1:, tidx] *= profile[tidx]

# survival_mat = np.ones((NHOUSE, tidx))
# for house in range(NHOUSE):
#     pass


approx_income_idx = map_to_nearest_grid(states, income_process)

# Handy for calculating cash in hand, no interest...
life_time_wealth_matrix = np.cumsum(income_process, axis=1)  # but I use real income process
for i in range(NHOUSE):
    life_time_wealth_matrix[i, :] += initial_capital[i]
simu_cash_on_hand = life_time_wealth_matrix[:, :]
simu_consum = np.zeros((NHOUSE, life_time))
simu_asset = np.zeros((NHOUSE, life_time))
simu_asset[:, 0] = initial_capital

death_rate = discount_arr/beta
for tidx in range(life_time):
    for house in range(NHOUSE):
        # if tidx >= work_time:
        #     death = rng.choice([True, False],1, p=[death_rate[tidx], 1 - death_rate[tidx]])
        #     if death:
        #         simu_asset[house, tidx:] = 0
        #         simu_consum[house, tidx:] = 0
        #         break
        this_cash_on_hand = interest*simu_asset[house, tidx] + income_process[house, tidx]
        state = approx_income_idx[house, tidx]
        if tidx < work_time:
            exo_cash_on_hand = exo_working_cash_on_hand[:, state, tidx]
        else:
            exo_cash_on_hand = exo_pension_cash_on_hand
        # simu_consum[house, tidx] = np.interp(simu_cash_on_hand[house, tidx],
        #                                      exo_cash_on_hand, exo_consum_arr[:, state, tidx])
        interpolate_f = interpolate.interp1d(exo_cash_on_hand, exo_consum_arr[:, state, tidx],
                                             fill_value="extrapolate", kind="linear")
        simu_consum[house, tidx] = interpolate_f(this_cash_on_hand)
        #  The second condition is when extrapolation kind of do a problematic job
        if simu_consum[house, tidx] > this_cash_on_hand or simu_consum[house, tidx] <= 0:
            simu_consum[house, tidx] = this_cash_on_hand
        if tidx == life_time-1:
            simu_consum[house, tidx] = this_cash_on_hand
            continue  # no asset holding in last period
        simu_asset[house, tidx+1] = this_cash_on_hand - simu_consum[house, tidx]

        if simu_consum[house, tidx] <= 0:
            print(house, tidx, state, this_cash_on_hand, simu_consum[house, tidx])
            simu_consum[house, tidx] = 0.001  # force things to work


# report values and graphs

# growth
log_consum = np.log(simu_consum)
avg_consum = np.mean(simu_consum, axis=0)
avg_log_consum = np.mean(log_consum, axis=0)
var_log_consum = np.var(log_consum, axis=0)
log_consum_growth = np.mean(np.amax(log_consum, axis=1)-log_consum[:, 0])

log_income = np.log(income_process)
avg_income = np.mean(income_process, axis=0)
avg_log_income = np.mean(log_income, axis=0)
var_log_income = np.var(log_income, axis=0)
log_income_growth = np.mean(np.amax(log_income, axis=1)-log_income[:, 0])

avg_asset = np.mean(simu_asset, axis=0)

start_age = 25
x_val = np.arange(start_age, start_age+life_time, step=1)
f_dict = {"consumption": avg_consum, "income": avg_income, "asset": avg_asset,
          "variance of income": var_log_income, "variance of consumption": var_log_consum}
for idx, it in enumerate(f_dict.keys()):
    fig = plt.figure(idx)
    fig.canvas.set_window_title(it)
    plt.plot(x_val, f_dict[it])

print("consum growth: {}, income growth: {}".format(log_consum_growth, log_income_growth))
# insurance
consum_lag = np.diff(simu_consum, axis=1)
covar_income_shock = Series(np.mean(consum_lag, axis=0)).cov(Series(np.mean(perm_shock_mat[:, 1:], axis=0)))
print("phi is", 1-covar_income_shock/(perm_shock_se**2))

plt.show()


