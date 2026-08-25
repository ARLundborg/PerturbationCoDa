import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mpltern
import numpy as np

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 16})
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text.latex', preamble=r'\usepackage{amsfonts,amssymb,amsthm,amsmath}')

fig = plt.figure(figsize=(15, 5))
fig.subplots_adjust(wspace=0.4)


z1 = np.array([0.72, 0.21, 0])
z1[2] = 1 - np.sum(z1)

z2 = np.array([1/5, 1/5-1/10, 0])
z2[2] = 1 - np.sum(z2)

z_c = np.ones(3)/3
e_1 = np.array([1,0,0])
a_1 = np.array([0, z1[0]+z1[1], z1[2]])
a_2 = np.array([0, z2[0]+z2[1], z2[2]])

clr = lambda z: np.log(z) - np.mean(np.log(z))
inv_clr = lambda x: np.exp(x)/np.sum(np.exp(x))

def unit_speed(E, z, step=0.01):
        delta = np.sum(np.abs(E-z))
        gammas = np.arange(0, delta, step)[:, np.newaxis]
        return z + gammas*((E-z)/delta)

def multiplicative_speed(A, B, z, step=0.01):
        z_A = np.sum(z[A])
        z_B = np.sum(z[B])
        A_z = z.copy()
        A_z[A] = 0
        A_z[B] = z[B]/z_B*(z_B + z_A)
        delta = np.sum(np.abs(A_z-z))
        speed = 2*z_A*z_B/(z_A+z_B)**2
        gammas = np.arange(0, delta/speed, step)[:, np.newaxis]
        return z + gammas*speed*(A_z-z)/delta

def gini_speed(z, step=0.01):
        d = np.shape(z)[0]
        z_c = np.ones_like(z)/d
        v = (z_c - z)/np.sum(np.abs(z_c-z))
        speed = 2*d/(np.abs(np.subtract.outer(v, v)).sum())
        delta = np.sum(np.abs(z_c-z))
        gammas = np.arange(0, delta/speed, step)[:, np.newaxis]
        return z + gammas*speed*v

def clr_diversity(z, step=0.01):
        delta = np.linalg.norm(clr(z), ord=2)
        gammas = np.arange(0, delta, step)[:, np.newaxis]
        return  inv_clr(clr(z) - gammas*clr(z)/delta)

color1 = "C0"
color2 = "C2"
color3 = "C3"

### Unit speed:
ax = fig.add_subplot(141, projection="ternary")
ax.set_title(r"Unit speed", y=1.3)
marker_size = 10
mark_every = 25
marker_shift = 10
line_width = 2.5
ax.plot(*unit_speed(z_c, z1).T, c=color1, markevery=(marker_shift,mark_every), marker='.', markerfacecolor='w', markeredgecolor="none", markersize=marker_size, linewidth=line_width)
ax.plot(*unit_speed(z_c, z2).T, c=color1, markevery=(marker_shift,mark_every), marker='.', markerfacecolor='w', markeredgecolor="none", markersize=marker_size, linewidth=line_width)
ax.plot(*unit_speed(e_1, z1).T, c=color2, markevery=(marker_shift,mark_every), marker='.', markerfacecolor='w', markeredgecolor="none", markersize=marker_size, linewidth=line_width)
ax.plot(*unit_speed(e_1, z2).T, c=color2, markevery=(marker_shift,mark_every), marker='.', markerfacecolor='w', markeredgecolor="none", markersize=marker_size, linewidth=line_width)
ax.plot(*unit_speed(a_1, z1).T, c=color3, markevery=(marker_shift,mark_every), marker='.', markerfacecolor='w', markeredgecolor="none", markersize=marker_size, linewidth=line_width)
ax.plot(*unit_speed(a_2, z2).T, c=color3, markevery=(marker_shift,mark_every), marker='.', markerfacecolor='w', markeredgecolor="none", markersize=marker_size, linewidth=line_width)
point_size=35
ax.scatter(*z1, c="black", marker="s", s=0.8*point_size, zorder=3)
ax.scatter(*z2, c="black", marker="^", s=point_size, zorder=3)
ax.scatter(*z_c, c=color1, s=point_size, zorder=3)
ax.scatter(*(e_1-(0.02,-0.01, -0.01)), c=color2, s=point_size, zorder=3)
ax.scatter(*a_1, c=color3, s=point_size, zorder=3)
ax.scatter(*a_2, c=color3, s=point_size, zorder=3)

ax.set_tlabel(r'$z^1$')
ax.set_llabel(r'$z^2$')
ax.set_rlabel(r'$z^3$')
ax.taxis.set_label_position("tick1")
ax.laxis.set_label_position("tick1")
ax.raxis.set_label_position("tick1")
ax.tick_params(axis='both', which='major', labelsize=12)

## Alternative speeds
ax = fig.add_subplot(142, projection="ternary")
ax.set_title(r"Alternative speeds", y=1.3)
marker_shift = 10
ax.plot(*gini_speed(z1).T, c=color1, markevery=(marker_shift,mark_every), marker='.', markerfacecolor='w', markeredgecolor="none", markersize=marker_size, linewidth=line_width)
ax.plot(*gini_speed(z2).T, c=color1, markevery=(marker_shift,mark_every), marker='.', markerfacecolor='w', markeredgecolor="none", markersize=marker_size, linewidth=line_width)
ax.plot(*multiplicative_speed([1, 2], [0], z1).T, c=color2, markevery=(marker_shift,mark_every), marker='.', markerfacecolor='w', markeredgecolor="none", markersize=marker_size, linewidth=line_width)
ax.plot(*multiplicative_speed([1, 2], [0], z2).T, c=color2, markevery=(marker_shift,mark_every), marker='.', markerfacecolor='w', markeredgecolor="none", markersize=marker_size, linewidth=line_width)
ax.plot(*multiplicative_speed([0], [1], z1).T, c=color3, markevery=(marker_shift,mark_every), marker='.', markerfacecolor='w', markeredgecolor="none", markersize=marker_size, linewidth=line_width)
ax.plot(*multiplicative_speed([0], [1], z2).T, c=color3, markevery=(marker_shift,mark_every), marker='.', markerfacecolor='w', markeredgecolor="none", markersize=marker_size, linewidth=line_width)
ax.scatter(*z1, c="black", marker="s", s=0.8*point_size, zorder=3)
ax.scatter(*z2, c="black", marker="^", s=point_size, zorder=3)
ax.scatter(*z_c, c=color1, s=point_size, zorder=3)
ax.scatter(*(e_1-(0.02,-0.01, -0.01)), c=color2, s=point_size, zorder=3)
ax.scatter(*a_1, c=color3, s=point_size, zorder=3)
ax.scatter(*a_2, c=color3, s=point_size, zorder=3)


ax.set_tlabel(r'$z^1$')
ax.set_llabel(r'$z^2$')
ax.set_rlabel(r'$z^3$')
ax.taxis.set_label_position("tick1")
ax.laxis.set_label_position("tick1")
ax.raxis.set_label_position("tick1")
ax.tick_params(axis='both', which='major', labelsize=12)

## Alternative speeds
ax = fig.add_subplot(143, projection="ternary")
ax.set_title(r"Log-ratio perturbations", y=1.3)
marker_shift = 10
ax.plot(*clr_diversity(z1).T, c=color1, markevery=(marker_shift,mark_every), marker='.', markerfacecolor='w', markeredgecolor="none", markersize=marker_size, linewidth=line_width)
ax.plot(*clr_diversity(z2).T, c=color1, markevery=(marker_shift,mark_every), marker='.', markerfacecolor='w', markeredgecolor="none", markersize=marker_size, linewidth=line_width)
ax.plot(*multiplicative_speed([1, 2], [0], z1).T, c=color2, markevery=(marker_shift,mark_every), marker='.', markerfacecolor='w', markeredgecolor="none", markersize=marker_size, linewidth=line_width)
ax.plot(*multiplicative_speed([1, 2], [0], z2).T, c=color2, markevery=(marker_shift,mark_every), marker='.', markerfacecolor='w', markeredgecolor="none", markersize=marker_size, linewidth=line_width)
ax.scatter(*z1, c="black", marker="s", s=0.8*point_size, zorder=3)
ax.scatter(*z2, c="black", marker="^", s=point_size, zorder=3)
ax.scatter(*z_c, c=color1, s=point_size, zorder=3)
ax.scatter(*(e_1-(0.02,-0.01, -0.01)), c=color2, s=point_size, zorder=3)


ax.set_tlabel(r'$z^1$')
ax.set_llabel(r'$z^2$')
ax.set_rlabel(r'$z^3$')
ax.taxis.set_label_position("tick1")
ax.laxis.set_label_position("tick1")
ax.raxis.set_label_position("tick1")
ax.tick_params(axis='both', which='major', labelsize=12)

## Legend

ax = fig.add_subplot(144)
ax.axis("off")

legend_elements = [Line2D([0], [0], marker="s", color="white", markersize=10, markerfacecolor="black"),
        Line2D([0], [0], marker="^", color="white", markersize=10, markerfacecolor="black"),
        Line2D([0], [0], color=color1, linestyle="solid", linewidth=line_width),
        Line2D([0], [0], color=color2, linestyle="solid", linewidth=line_width),
        Line2D([0], [0], color=color3, linestyle="solid", linewidth=line_width),
        Line2D([0], [0], marker="o", color="white", markersize=10, markerfacecolor=color1),
        Line2D([0], [0], marker="o", color="white", markersize=10, markerfacecolor=color2),
        Line2D([0], [0], marker="o", color="white", markersize=10, markerfacecolor=color3)
        ]
ax.legend(legend_elements, [r"$z_1$", r"$z_2$", "diversity", r"$\{2,3\} \to \{1\}$", r"$\{1\} \to \{2\}$", r"$z_{\mathrm{cen}}$",r"$e_1$", r"$\mathsf{A}_{\{1\} \to \{2\}}(z) $"], loc="center")

fig.savefig("plots/perturbations.pdf", bbox_inches="tight")