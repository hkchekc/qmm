import numpy as np
from time import time
import math
import matplotlib.pyplot as plt
from numba import njit
from scipy import interpolate, stats
from numpy.random import MT19937, Generator


start = time()
@njit
def map_to_nearest_grid(income_grids, income_processes):
    approx_income = np.zeros(income_processes.shape)
    for hi, household_income in enumerate(income_processes):
        for ti, income in enumerate(household_income):
            hidx = np.abs(income_grids - income).argmin()
            approx_income[hi, ti] = hidx
    return approx_income.astype(np.int8)  # returning indexes of income... or is it better to return both


# define function for marginal utilty from bequest:
def bequest_implied_consumption(tmr_a):
    if phi1 == 0:
        return tmr_a * interest + pension_income
    ugly_term = (phi2 / (1 - gamma) / phi1 / net_tax) ** inv_gamma
    c = (ugly_term / (phi2 + net_tax * ugly_term)) * (phi2 + net_tax * (pension_income + interest * tmr_a))
    return c


@njit
def phi_bequest_prime(tmr_a):
    if phi1 == 0:
        return 0
    pre_term = net_tax * (1 - gamma) * phi1 / phi2
    next_term = 1 + (net_tax * tmr_a) / phi2
    return pre_term / (next_term ** gamma)


# parameters
profile = np.genfromtxt("input/lc_profile.csv", delimiter=",")
profile = profile[1:-1, 1]
beta = .95
interest = 1.04
work_time = 40
retire_time = 30
last_surv_rate = 0.82
life_time = work_time + retire_time
discount_arr = np.ones(life_time)
survive_reudction = (1 - last_surv_rate) / (retire_time - 1)
for t in range(work_time + 1, life_time):
    discount_arr[t] = discount_arr[t - 1] - survive_reudction
print(discount_arr[40])
gamma = 1.5
inv_gamma = 1 / gamma
pension_rate = 0.6
NA = 1000
a_min = 0.
a_max = 150
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
NZ = markov.shape[0]
markov_t = markov.T
states = np.array([0.5324, 0.7415, 1., 1.3487, 1.8784, 0.5324, 0.7415, 1., 1.3487, 1.8784])
trans_shock = .4
trans_shock_prob = 0.01
states[5:] *= trans_shock
bequest_min = 0
bequest_max = 150
NB = 20
net_receive_tax = 1
net_tax = 0.85  # sending tax
bequest_arr = np.linspace(bequest_min, bequest_max, num=NB) * net_receive_tax
phi1 = -10
phi2 = 10
# part of solution
bequest_params = np.array([0.4, 0.38, 2.22])  # preliminary bequest parameters, prob_0, mean, variance
new_bequest_params = np.zeros(3)
bequest_err = 100
bequest_time = work_time
crit = 5e-2  # loop through the gmm of parametric distribution

# start with last period
pension_income = profile[-1] * pension_rate
# precalculate retired cash on hand, as there is no time dependent element
exo_pension_cash_on_hand = interest * tmr_a_grid + pension_income

# simulation parameters
rho = .97
rng = Generator(MT19937(1234))
init_income = 1
NHOUSE = 2000

init_cap_mean = -2.5
init_cap_se = 2
initial_capital = np.exp(rng.normal(init_cap_mean, init_cap_se, NHOUSE))
init_cap_cap = math.exp(3 * init_cap_se + init_cap_mean)
initial_capital[initial_capital > init_cap_cap] = init_cap_cap

# for simulation
# precalculate worktime income process should be the same for all bequest parameters
perm_shock_mean = 0.
perm_shock_se = math.sqrt(0.015)
income_process = np.ones((NHOUSE, life_time))  # initial period is one, don't change
perm_shock_mat = np.zeros((NHOUSE, life_time))
this_total_income = income_process[:, 0]  # which is one
for tidx in range(1, work_time):
    this_income = rho * np.log(this_total_income)  # no profile as it will be part of the policy as tidx
    this_shock = rng.normal(perm_shock_mean, perm_shock_se, NHOUSE)
    perm_shock_mat[:, tidx] = this_shock
    tmp_shock = rng.choice([1, trans_shock], NHOUSE, p=[1 - trans_shock_prob, trans_shock_prob])
    this_total_income = np.exp(this_income + this_shock)
    # this_total_income[this_total_income< 0] = 1
    income_process[:, tidx] = tmp_shock * this_total_income * profile[tidx]
income_process[:, work_time:] = pension_income
print(income_process[:1])

death_mat = rng.uniform(0, 1, (NHOUSE, retire_time)) < discount_arr[work_time:]
surive_mat = np.cumprod(death_mat, axis=1)
die_time_arr = (surive_mat == 0).argmax(axis=1)
die_time_arr += work_time
die_time_arr[die_time_arr == work_time] += 1

# simu - discretize income
approx_income_idx = map_to_nearest_grid(states, income_process)

maxit = 20
cit = 1

while bequest_err > crit and maxit > cit:
    # result arrays, the fourth dimension not used except for
    exo_consum_arr = np.zeros((NA, NZ, life_time, NB))
    greatest_constrainted_cash_on_hand = np.zeros(life_time)
    exo_cash_on_hand = np.zeros((NA, NZ, life_time, NB))
    # at every age, either all nan or sum(NB) = 1
    l_norm = stats.lognorm([bequest_params[2]], loc=bequest_params[1])
    prob_bequest = np.zeros(NB)
    prob_bequest[0] = bequest_params[0]
    tmp = l_norm.cdf(bequest_arr[:])
    tmp[1:] -= tmp[:-1]
    tmp = tmp[1:]
    prob_bequest[1:] = tmp / sum(tmp) * (1 - bequest_params[0])
    bequest_prob_mat = np.empty((NB, life_time))
    bequest_prob_mat[:, :] = np.nan
    bequest_prob_mat[:, bequest_time] = prob_bequest
    # tmp array
    implied_consum_arr = np.zeros((NA + 1, NZ, life_time))
    next_value = np.zeros((NA, NZ))
    implied_cash_on_hand = np.zeros((NA + 1, NZ, life_time))

    # Populate last period
    for aidx in range(NA):
        # I just define the init coh as correct grid
        implied_consum_arr[aidx+1, :, -1] = bequest_implied_consumption(tmr_a_grid[aidx])  # actually a'
        exo_consum_arr[aidx, :, -1, 0] = implied_consum_arr[aidx+1, :, -1]
        implied_cash_on_hand[aidx+1, :, -1] = interest * tmr_a_grid[aidx] + pension_income
        next_value[aidx, :] = interest / (exo_consum_arr[aidx, 0, -1, 0] ** gamma)

    # leave it out and calc in loop
    exo_working_cash_on_hand = np.zeros((NA, NZ, work_time))

    # Backward induction
    for tidx in range(life_time - 2, -1, -1):
        if tidx >= work_time:  # retired
            for aidx in range(1, NA + 1):
                implied_consum_arr[aidx, :, tidx] = 1 / ((next_value[aidx - 1, 0]) ** inv_gamma)
                implied_cash_on_hand[aidx, :, tidx] = implied_consum_arr[aidx, :, tidx] + tmr_a_grid[aidx - 1]
            greatest_constrainted_cash_on_hand[tidx] = implied_cash_on_hand[1, 0, tidx]
            if tidx == bequest_time:  # last period of using these object, so no worries of wrong shape etc.
                # the exo cash on hand should be more  and also in different shape
                exo_pension_cash_on_hand = np.zeros((NA, NB))
                for bidx in range(NB):
                    exo_pension_cash_on_hand[:, bidx] = interest * tmr_a_grid + pension_income + bequest_arr[bidx]
            else:
                exo_pension_cash_on_hand = interest * tmr_a_grid + pension_income

            # interpolate_f = interpolate.interp1d(implied_cash_on_hand[:, 0, tidx], implied_consum_arr[:, 0, tidx],
            #                                      fill_value="extrapolate", kind="linear")
            if tidx != bequest_time:
                interpolate_f = interpolate.interp1d(implied_cash_on_hand[:, 0, tidx], implied_consum_arr[:, 0, tidx],
                                                 fill_value="extrapolate", kind="linear")
                exo_consum_arr[:, 0, tidx, 0] = interpolate_f(exo_pension_cash_on_hand)
                # Make sure no negative consumption or over asset constraint
                mx = np.ma.masked_less_equal(exo_pension_cash_on_hand, greatest_constrainted_cash_on_hand[tidx]).mask
                exo_consum_arr[:, 0, tidx, 0][mx] = exo_pension_cash_on_hand[mx]
            else:  # again rely on the bequest time being the last period that use this piece of code
                interpolate_f = interpolate.interp1d(implied_cash_on_hand[:, 0, tidx], implied_consum_arr[:, 0, tidx],
                                                 fill_value="extrapolate", kind="linear")
                for bidx in range(NB):
                    for aidx in range(NA):
                        exo_consum_arr[aidx, 0, tidx, bidx] = interpolate_f(exo_pension_cash_on_hand[aidx, bidx])
                    mx = np.ma.masked_less_equal(exo_pension_cash_on_hand[:, bidx],
                                                 greatest_constrainted_cash_on_hand[tidx]).mask
                    exo_consum_arr[:, 0, tidx, bidx][mx] = exo_pension_cash_on_hand[:, bidx][mx]

            # not using next period value after getting this.consumption, therefore I can flush it here.
            next_value = np.zeros((NA, NZ))
            if tidx != bequest_time:
                # no bequest received, last arguement = 0
                next_value[:, 0] = beta * discount_arr[tidx] * interest / (exo_consum_arr[:, 0, tidx, 0] ** gamma) \
                                   + (1 - discount_arr[tidx]) * phi_bequest_prime(tmr_a_grid)
            else:
                for aidx in range(NA):
                    for bidx in range(NB):
                        next_value[aidx, 0] += bequest_prob_mat[bidx, bequest_time] * (
                                    beta * discount_arr[tidx] * interest /
                                    (exo_consum_arr[aidx, 0, bequest_time, bidx] ** gamma))
                    next_value[aidx, 0] += (1 - discount_arr[tidx]) * phi_bequest_prime(tmr_a_grid[aidx])
            # just for completeness in fact not important.
            for i in range(1, NZ):
                next_value[:, i] = next_value[:, 0]

        else:  # working age
            for zidx in range(NZ):
                for aidx in range(1, NA + 1):
                    implied_consum_arr[aidx, zidx, tidx] = 1 / ((next_value[aidx - 1, zidx]) ** inv_gamma)
                    implied_cash_on_hand[aidx, zidx, tidx] = implied_consum_arr[aidx, zidx, tidx] + tmr_a_grid[aidx - 1]
                # see where it binds
                greatest_constrainted_cash_on_hand[tidx] = implied_cash_on_hand[1, zidx, tidx]
                exo_working_cash_on_hand[:, zidx, tidx] = interest * tmr_a_grid + states[zidx] * profile[tidx]
                interpolate_f = interpolate.interp1d(implied_cash_on_hand[:, zidx, tidx],
                                                     implied_consum_arr[:, zidx, tidx],
                                                     fill_value="extrapolate", kind="linear")
                exo_consum_arr[:, zidx, tidx, 0] = interpolate_f(exo_working_cash_on_hand[:, zidx, tidx])
                # Make sure no negative consumption or over asset constraint
                mx = np.ma.masked_less_equal(exo_working_cash_on_hand[:, zidx, tidx],
                                             greatest_constrainted_cash_on_hand[tidx]).mask
                exo_consum_arr[:, zidx, tidx, 0][mx] = exo_working_cash_on_hand[:, zidx, tidx][mx]
            # not using next period value after getting this.consumption, therefore I can flush it here.
            next_value = np.zeros((NA, NZ))
            tmp_consum = exo_consum_arr[:, :, tidx, 0]
            for last_zidx in range(NZ):
                for aidx in range(NA):
                    this_val = beta * interest / (tmp_consum[aidx, last_zidx] ** gamma)  # step 5
                    for zidx in range(NZ):
                        next_value[aidx, zidx] += markov[zidx, last_zidx] * this_val  # error-prone

    np.savetxt("data_output/consumption_policy.csv", implied_consum_arr[:, :, 0], delimiter=",")
    np.savetxt("data_output/income.csv", implied_consum_arr[:, :, 40], delimiter=",")
    np.savetxt("data_output/coh.csv", implied_consum_arr[:, :, 39], delimiter=",")

    # simulation
    simu_consum = np.zeros((NHOUSE, life_time))
    simu_asset = np.zeros((NHOUSE, life_time))
    simu_asset[:, 0] = initial_capital
    simu_bequest_sent = np.zeros(NHOUSE)
    simu_survival = np.ones(retire_time)

    # get bequests
    tmp_bequest = np.exp(rng.normal(bequest_params[1], bequest_params[2], size=NHOUSE))
    bequest_received = tmp_bequest
    bequest_received_cap = math.exp(3 * bequest_params[2] + bequest_params[1])
    # tmp_rand = rng.uniform(0, 1, size=NHOUSE) < bequest_params[0]
    # bequest_received[tmp_rand] = 0
    # bequest_received[bequest_received > bequest_received_cap] = bequest_received_cap
    for house in range(NHOUSE):
        bequest_received[house] = rng.choice([bequest_received[house], 0],
                                             1, p=[1 - bequest_params[0], bequest_params[0]])
        bequest_received[bequest_received > bequest_received_cap] = bequest_received_cap

    for house in range(NHOUSE):
        for tidx in range(life_time):
            this_cash_on_hand = 0
            if tidx == bequest_time:
                this_cash_on_hand += bequest_received[house] * net_receive_tax
            if die_time_arr[house] == tidx:
                simu_bequest_sent[house] = net_tax * (simu_asset[house, tidx])
                break
            this_cash_on_hand += simu_asset[house, tidx] + income_process[house, tidx]

            if tidx < work_time:  # working people
                state = approx_income_idx[house, tidx]
                implied_coh = implied_cash_on_hand[:, state, tidx]
                try:
                    simu_consum[house, tidx] = np.interp(this_cash_on_hand, implied_coh, implied_consum_arr[:, state, tidx])
                except ValueError:
                     interpolate_f = interpolate.interp1d(implied_coh, implied_consum_arr[:, state, tidx],
                                                          fill_value="extrapolate", kind="linear")
                     simu_consum[house, tidx] = interpolate_f(this_cash_on_hand)
            else:
                implied_coh = implied_cash_on_hand[:, 0, tidx]
                try:
                    simu_consum[house, tidx] = np.interp(this_cash_on_hand, implied_coh, implied_consum_arr[:, 0, tidx])
                except ValueError:
                    interpolate_f = interpolate.interp1d(implied_coh, implied_consum_arr[:, 0, tidx],
                                                     fill_value="extrapolate", kind="linear")
                    simu_consum[house, tidx] = interpolate_f(this_cash_on_hand)
            #  The second condition is when extrapolation kind of do a problematic job
            if simu_consum[house, tidx] > this_cash_on_hand or simu_consum[house, tidx] <= 0:
                simu_consum[house, tidx] = this_cash_on_hand
            if tidx == life_time - 1:
                if phi1 != 0:
                    simu_bequest_sent[house] = net_tax * (this_cash_on_hand - simu_consum[house, tidx])
                else:
                    simu_bequest_sent[house] = 0
                    simu_consum[house, tidx] = this_cash_on_hand
                continue  # no asset holding in last period
            simu_asset[house, tidx + 1] = interest * (this_cash_on_hand - simu_consum[house, tidx])

    print(bequest_params)
    # new_bequest_params[0] = (NHOUSE - np.count_nonzero(simu_bequest_sent)) / NHOUSE
    new_bequest_params[0] = (np.count_nonzero(simu_bequest_sent == 0)) / NHOUSE
    tmp = np.log(simu_bequest_sent[simu_bequest_sent > 0])
    new_bequest_params[1] = np.mean(tmp)
    new_bequest_params[2] = math.sqrt(np.var(tmp))
    # update parameters
    print(new_bequest_params, bequest_params, len(tmp))
    bequest_err = np.sum(np.abs(new_bequest_params - bequest_params))
    print(bequest_err)
    print("=================================")
    # bequest_err = 0  # for now
    ratio = 0
    bequest_params[:] = ratio * bequest_params[:] + (1 - ratio) * new_bequest_params

    cit += 1

    # report values and graphs

    if maxit <= cit or bequest_err <= crit:
        non_zero_consum = simu_consum[simu_consum > 0]
        log_consum = np.log(simu_consum)

        surv_per_time = np.mean((simu_consum > 0), axis=0)
        print(surv_per_time)
        avg_consum = np.mean(simu_consum, axis=0) / surv_per_time
        avg_log_consum = np.mean(log_consum, axis=0)

        log_income = np.log(income_process)
        avg_income = np.mean(income_process, axis=0)
        avg_log_income = np.mean(log_income, axis=0)
        var_log_income = np.var(log_income, axis=0)
        log_income_growth = np.mean(np.amax(log_income, axis=1) - log_income[:, 0])

        avg_asset = np.mean(simu_asset, axis=0) / surv_per_time
        #
        gini_arr = np.zeros(life_time)
        wealth_20 = np.zeros(life_time)
        wealth_5 = np.zeros(life_time)
        wealth_1 = np.zeros(life_time)
        for tidx in range(life_time):
            if tidx >= work_time:
                this_asset = np.sort(simu_asset[:, tidx][surive_mat[:, tidx-work_time] != 0])
                len_this_asset = len(this_asset)
                # avoid outliners ruining the graph, cut out last 20 people
                this_asset = this_asset[:-5]
            else:
                this_asset = np.sort(simu_asset[:, tidx])
                # avoid outliners ruining the graph, cut out last 20 people
                this_asset = this_asset[:-5]
            # this_cumsum = np.cumsum(this_asset)
            # print(this_cumsum[-10:], "cumsum", len(this_cumsum))
            # this_degree = np.linspace(0, np.sum(this_asset), NHOUSE)
            # print(this_degree[-10:], "degree", this_cumsum[-1])
            # print("================")
            # this_gini = 0
            # this_area = ((this_cumsum[-1])**2)/2
            # for idx in range(1, len(this_asset)):
            #
            #     if this_degree[idx] > this_asset[idx]:
            #         this_gini += 1- this_cumsum[idx]/this_degree[idx]
            # if this_degree[-1] == 0:
            #     gini_arr[tidx] = 0
            #     continue
            # gini_arr[tidx] = this_gini/len(this_asset)
            # from wiki
            this_gini = 0
            for i in range(len(this_asset)):
                for j in range(len(this_asset)):
                    this_gini += abs(this_asset[i]-this_asset[j])
            gini_arr[tidx] = this_gini/2/NHOUSE/NHOUSE/np.mean(this_asset)
            # share of wealth held by 20, 5, 1
            this_sum = np.sum(this_asset)
            wealth_20[tidx] = np.sum(this_asset[this_asset>=np.quantile(this_asset, 0.8)])/this_sum
            wealth_5[tidx] = np.sum(this_asset[this_asset>=np.quantile(this_asset, 0.95)])/this_sum
            wealth_1[tidx] = np.sum(this_asset[this_asset>=np.quantile(this_asset, 0.99)])/this_sum

        end = time()
        print("Elapse time is {}".format(end-start))
        print("=======================================================")
        start_age = 25
        x_val = np.arange(start_age, start_age + life_time, step=1)
        f_dict = {"consumption": avg_consum, "income": avg_income, "asset": avg_asset, "gini": gini_arr}
        for idx, it in enumerate(f_dict.keys()):
            fig = plt.figure(idx)
            fig.canvas.set_window_title(it)
            plt.plot(x_val, f_dict[it])
            fname = '{}_tax{}_phi{}'.format(it, net_tax, phi1)
            plt.savefig('data_output/{}.png'.format(fname))
            plt.title(fname)

        plt.figure(idx + 1)
        plt.plot(exo_working_cash_on_hand[:, 2, 39], exo_consum_arr[:, 2, 39, 0], color='red')
        plt.plot(exo_working_cash_on_hand[:, 2, 38], exo_consum_arr[:, 2, 38, 0], color='blue')
        plt.plot(exo_pension_cash_on_hand[:, 0], exo_consum_arr[:, 0, 40, 0], color='orange')
        plt.figure(idx + 2)
        x = np.linspace(simu_bequest_sent.min(), np.quantile(simu_bequest_sent, .98), NHOUSE)
        # KDE
        kde = stats.gaussian_kde(simu_bequest_sent, bw_method='scott')
        p = kde.pdf(x)
        print(sum(p), "p1")
        plt.plot(x, p, color="blue", label="sent")
        kde = stats.gaussian_kde(bequest_received)
        p = kde.pdf(x)
        print(sum(p), "p2")
        plt.plot(x, p, color="red", label="received")
        plt.legend()
        fname = '{}_tax{}_phi{}'.format("kde", net_tax, phi1)
        plt.savefig('data_output/{}.png'.format(fname))
        plt.title(fname)

        # wealth quantile
        plt.figure(idx + 3)
        plt.plot(x_val, wealth_20, color="red", label="q=20")
        plt.plot(x_val, wealth_5, color="blue", label="q=5")
        plt.plot(x_val, wealth_1, color="orange", label="q=1")
        plt.legend()
        fname = '{}_tax{}_phi{}'.format("quantiles", net_tax, phi1)
        plt.savefig('data_output/{}.png'.format(fname))
        plt.title(fname)


plt.show()
