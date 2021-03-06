import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# usage: python3 plot.py directory_intended
dir = ""


# don't have to change
data_dir = "data_output"
dpath = os.path.join(dir, data_dir)
arr_li = [f.split('.')[0] for f in os.listdir(dpath) if os.path.isfile(os.path.join(dpath, f))]
f_dict = dict()
arr_li = arr_li[1:]  # first file is .
for arr in arr_li:
    f_dict[arr] = np.genfromtxt(dpath+"/"+arr+".txt")


ngraph = len(arr_li) -1  # a grid is x axis



for idx, it in enumerate(arr_li):
    if 'path' not in it or 'dist' in it or 'png' in it:
        continue
    fig  = plt.figure(idx)
    fig.canvas.set_window_title(it)
    plt.plot(range(200), f_dict[it])
    plt.legend()
    fname = '{}'.format(it)
    plt.suptitle(fname)
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    # plt.savefig('data_output/{}.png'.format(fname))


plt.show()
