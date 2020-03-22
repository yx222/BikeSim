import logging
from bikesim.models.kinematics import Kinematics
from bikesim.models.multibody import MultiBodySystem
from bikesim.utils.visualization import create_kinematic_animation
import numpy as np

logger = logging.getLogger(__name__)


def simulate_damper_sweep(sag, system: MultiBodySystem,
                          create_animation: bool = False):
    """
    Sweep the eye-to-eye damper length and solve for geometry.
    """
    # TODO: figure out where these data should go.
    # They should probably be bike design parameters
    damper_eye_to_eye = 0.21
    damper_travel = 0.05
    fork_travel = 0.13

    l_damper = damper_eye_to_eye - damper_travel*sag
    l_fork = fork_travel * (1-sag)

    bike_kinematics = Kinematics(system=system)

    # Solve the problem
    x = bike_kinematics.get_init_guess()
    x_array = np.zeros((len(sag), bike_kinematics.n_dec))

    rear_axle_position_list = []
    for ii in range(len(l_damper)):
        nlp = bike_kinematics.construct_nlp(
            l_fork=l_fork[ii], l_damper=l_damper[ii])
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
