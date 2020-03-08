import logging
from bikesim.models.kinematics import Kinematics
from bikesim.models.multibody import MultiBodySystem
from bikesim.utils.visualization import create_kinematic_animation
import numpy as np


def simulate_damper_sweep(l_damper=0.21, system_file='geometries/5010.json',
                          create_animation: bool = False):
    """
    Sweep the eye-to-eye damper length and solve for geometry.
    """
    system = MultiBodySystem.from_json(system_file)
    bike_kinematics = Kinematics(system=system)

    # Solve the problem
    x = bike_kinematics.get_init_guess()

    # sweep through damper length, make it iterable if it's scalar
    if np.isscalar(l_damper):
        l_damper = [l_damper]

    x_array = np.zeros((len(l_damper), bike_kinematics.n_dec))

    rear_axle_position_list = []
    for ii, l in enumerate(l_damper):
        nlp = bike_kinematics.construct_nlp(l_damper=l)
        x, info = nlp.solve(x)
        x_array[ii] = x

        # record rear axle
        system.set_states(x)

        rear_axle_position_list.append(system.find_point(
            'rear_triangle', 'rear_axle').get_position())

    if create_animation:
        create_kinematic_animation(system, x_array, '5010_animation.mp4')

    xz = np.vstack(rear_axle_position_list)

    return xz[:, 0], xz[:, 1]


if __name__ == "__main__":
    simulate_damper_sweep(l_damper=0.21)
