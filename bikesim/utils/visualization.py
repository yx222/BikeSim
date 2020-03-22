import logging
import matplotlib.pyplot as plt
from matplotlib import animation
from typing import Optional

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

logger = logging.getLogger(__name__)


def create_kinematic_animation(system, solutions, file_name):
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes()
    ax.axis('equal')
    num_frame = len(solutions)

    def animate(i):
        system.set_states(solutions[i])
        ax.clear()
        ax.axis('equal')
        ax.set_xlim([-1, 1.5])
        ax.set_ylim([-0.5, 1.5])
        return system.plot(ax)

    t_total = 3
    t_interval = t_total*1000/num_frame
    anim = animation.FuncAnimation(fig, animate, frames=num_frame,
                                   interval=t_interval,
                                   blit=False, repeat=False)

    anim.save(file_name)
