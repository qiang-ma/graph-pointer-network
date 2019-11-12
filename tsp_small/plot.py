import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

# optimal solutions by LKH
optimal_20 = 3.8358657360076904
optimal_50 = 5.696028145550453
# optimal_100 = 7.756416320800781

# solutions by 2opt
two_opt_20 = 3.9597170789140352
two_opt_50 = 6.116784301417292
# two_opt_100 = 8.509583544970718

# solutions by 2opt
Christofides_20 = 4.173815034182981
Christofides_50 = 6.306419113363065
# Christofides_100 = 8.684402587273238

# solutions by Christofides
NN_20 = 4.484824772173980
NN_50 = 6.943632284694007
# NN_100 = 9.685737044969972

GPN_20 = 3.8725168704986572
GPN_50 = 5.955830812454224
# GPN_100 = 0

n_groups = 2

AM = (0.34, 1.76)

GPN = (100*(GPN_20/optimal_20 - 1), 100*(GPN_50/optimal_50 - 1))

PN = (1.83, 4.75)

DQN = (1.42, 5.16)

OPT2 = (100*(two_opt_20/optimal_20 - 1), 100*(two_opt_50/optimal_50 - 1))

RI = (4.36, 7.65)

Christo = (100*(Christofides_20/optimal_20 - 1), 100*(Christofides_50/optimal_50 - 1))

NN = (100*(NN_20/optimal_20-1), 100*(NN_50/optimal_50-1))


fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.1

opacity = 0.4
error_config = {'ecolor': '0.3'}

rects1 = ax.bar(index, AM, bar_width,
                alpha=opacity, color='b', error_kw=error_config,
                label='Attention Model')

rects2 = ax.bar(index + 1*bar_width, GPN, bar_width,
                alpha=opacity, color='g', error_kw=error_config,
                label='Graph Pointer Net')

rects3 = ax.bar(index + 2*bar_width, PN, bar_width,
                alpha=opacity, color='r', error_kw=error_config,
                label='Pointer Net')

rects4 = ax.bar(index + 3*bar_width, DQN, bar_width,
                alpha=opacity, color='y', error_kw=error_config,
                label='s2v-DQN')

rects5 = ax.bar(index + 4*bar_width, OPT2, bar_width,
                alpha=opacity, color='m', error_kw=error_config,
                label='2-opt')

rects6 = ax.bar(index + 5*bar_width, RI, bar_width,
                alpha=opacity, color='c', error_kw=error_config,
                label='Random Insertion')

rects7 = ax.bar(index + 6*bar_width, Christo, bar_width,
                alpha=opacity, color='k', error_kw=error_config,
                label='Christofides')

'''
rects8= ax.bar(index + 7*bar_width, NN, bar_width,
                alpha=opacity, color='k', error_kw=error_config,
                label='Nearest Neighbor')
'''


# ax.set_xlabel('Group')
ax.set_ylabel('Approximate gap to optimal %')
# ax.set_title('')
ax.set_xticks(index + 3*bar_width)
ax.set_xticklabels(('TSP20', 'TSP50'))
ax.legend()

fig.tight_layout()
plt.show()
