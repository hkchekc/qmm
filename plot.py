import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# usage: python3 plot.py directory_intended
dir = sys.argv[1]


# don't have to change
data_dir = "data_output"
dpath = os.path.join(dir, data_dir)
arr_li = [f.split('.')[0] for f in os.listdir(dpath) if os.path.isfile(os.path.join(dpath, f))]
f_dict = dict()
arr_li = arr_li[1:]  # first file is .
for arr in arr_li:   
    f_dict[arr] = np.genfromtxt(dpath+"/"+arr+".txt")


ngraph = len(arr_li) -1  # a grid is x axis

# adjust policy function
for i in range(f_dict['pfunc'].shape[0]):
    for j in range(f_dict['pfunc'].shape[1]):
        f_dict['pfunc'][i,j] = f_dict['a_grid'][int(f_dict['pfunc'][i,j])]

for idx, it in enumerate(arr_li):
    if it=='a_grid':
        continue
    fig  = plt.figure(idx)
    fig.canvas.set_window_title(it)
    plt.plot(f_dict['a_grid'], f_dict[it])

# problem set 1 specific
norm_grid = np.linspace(-3, 24, len(f_dict['a_grid']))
fig = plt.figure(100)
fig.canvas.set_window_title("log space grid")
plt.plot(norm_grid, f_dict['a_grid'])

plt.show()





