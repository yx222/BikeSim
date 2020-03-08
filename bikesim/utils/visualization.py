import matplotlib.pyplot as plt
from matplotlib import animation
from typing import Optional

from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")


def create_kinematic_animation(system, solutions, file_name):
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes()
    num_frame = len(solutions)

    def animate(i):
        system.set_states(solutions[i])
        ax.clear()
        return system.plot(ax)

    t_total = 3
    t_interval = t_total*1000/num_frame
    anim = animation.FuncAnimation(fig, animate, frames=num_frame,
                                   interval=t_interval,
                                   blit=False, repeat=False)

    anim.save(file_name)
