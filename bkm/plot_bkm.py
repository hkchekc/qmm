import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# usage: python3 plot.py directory_intended
dir = ""

one = np.genfromtxt("./data_output/dist_path.txt")

abs_diff = np.abs(one[:, 0]-one[:, -1])
max_diff = np.max(abs_diff)
print(max_diff, np.argmax(abs_diff), "max diff")
print("first 100 states")
print(sum(one[:50, -2]), np.count_nonzero(one[:, -2]))
print(sum(one[:50, 0]), np.count_nonzero(one[:, 0 ]))
print("=====================================")
print("last 100 states")
print(sum(one[-50:, -2]))
print(sum(one[-50:, 0]))
# raise ValueError
# don't have to change
data_dir = "data_output"
dpath = os.path.join(dir, data_dir)
arr_li = [f.split('.')[0] for f in os.listdir(dpath) if os.path.isfile(os.path.join(dpath, f)) and "benchmark" not in f]
f_dict = dict()
arr_li = arr_li[1:]  # first file is .
for arr in arr_li:
    f_dict[arr] = np.genfromtxt(dpath+"/"+arr+".txt")

ngraph = len(arr_li) -1  # a grid is x axis

drop = 1
time = 120

# KS IRF
def ks_forcast(current_k, productivity):
    slope =np.array([0.92039774, 0.94923057])
    intercept = np.array([0.10849442, 0.06327283])
    ks_states = np.array([.9832, 1.0157])
    # interpolate
    interp_slope = np.interp(productivity, ks_states, slope)
    interp_intercept = np.interp(productivity, ks_states, intercept)
    return np.exp(np.log(current_k) * interp_slope + interp_intercept)  # log normalized
alpha = 0.36
delta =.06
agg_c = np.zeros(time)
agg_i = np.zeros(time)
agg_k = np.ones(time)
agg_y = np.zeros(time)
agg_shocks = f_dict["prod_path"]
for tidx in range(1, time):
    agg_k[tidx] = ks_forcast(agg_shocks[tidx - 1], agg_k[tidx - 1])
    agg_i[tidx] = (agg_k[tidx] - agg_k[tidx - 1]) + delta * agg_k[tidx - 1]
    agg_y[tidx] = agg_shocks[tidx]*(agg_k[tidx]**alpha)
    agg_c[tidx] = agg_y[tidx] - agg_i[tidx]

ks_dict = {"y_path":agg_y, "c_path":agg_c, "ak_path":agg_k, "i_path":agg_i}


for idx, it in enumerate(arr_li):
    if 'path' not in it or 'dist' in it or 'png' in it:
        continue
    fig  = plt.figure(idx)
    fig.canvas.set_window_title(it)
    tmp_log_this = np.log(f_dict[it])[:-drop]
    plt.plot(range(time)[:-drop], tmp_log_this-np.log(f_dict[it][-1]), color="orange", label="BKM")
    try:
        tmp_log_this = np.log(ks_dict[it])[:-drop]
        plt.plot(range(time)[:-drop], tmp_log_this-np.log(ks_dict[it][-1]), color="blue", label="KS")
        print(ks_dict[it][-1], it)
    except KeyError:
        print("{} not in KS".format(it))
    plt.legend()
    fname = '{}'.format(it)
    plt.suptitle(fname)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    # plt.savefig('data_output/{}.png'.format(fname))


plt.show()
