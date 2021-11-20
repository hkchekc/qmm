import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# usage: python3 plot.py directory_intended
dir = ""

# one = np.genfromtxt("./data_output/dist_path.txt")
#
# abs_diff = np.abs(one[:, 0]-one[:, -50])
# max_diff = np.max(abs_diff)
# print(max_diff)
# print(*one[2000:2200, 0])
# raise ValueError
# don't have to change
data_dir = "data_output"
dpath = os.path.join(dir, data_dir)
arr_li = [f.split('.')[0] for f in os.listdir(dpath) if os.path.isfile(os.path.join(dpath, f))]
f_dict = dict()
arr_li = arr_li[1:]  # first file is .
for arr in arr_li:
    f_dict[arr] = np.genfromtxt(dpath+"/"+arr+".txt")


ngraph = len(arr_li) -1  # a grid is x axis

drop = 50

for idx, it in enumerate(arr_li):
    if 'path' not in it or 'dist' in it or 'png' in it:
        continue
    fig  = plt.figure(idx)
    fig.canvas.set_window_title(it)
    plt.plot(range(200)[:-drop], np.log(f_dict[it])[:-drop])
    plt.legend()
    fname = '{}'.format(it)
    plt.suptitle(fname)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    # plt.savefig('data_output/{}.png'.format(fname))


plt.show()
