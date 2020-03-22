import logging
from bikesim.models.kinematics import Kinematics
from bikesim.models.multibody import MultiBodySystem
from bikesim.utils.visualization import create_kinematic_animation
import numpy as np

logger = logging.getLogger(__name__)


def simulate_damper_sweep(sag, system: MultiBodySystem,
                          damper_stroke=0.05, create_animation: bool = False):
    """
    Sweep the eye-to-eye damper length and solve for geometry.
    """
    # TODO: figure out where these data should go.
    # They should probably be bike design parametersu
    damper_eye_to_eye = 0.21
    fork_travel = 0.13

    l_damper = damper_eye_to_eye - damper_stroke*sag
    l_fork = fork_travel * (1-sag)

    bike_kinematics = Kinematics(system=system)

    # Solve the problem
    x = bike_kinematics.get_init_guess()
    x_array = np.zeros((len(sag), bike_kinematics.n_dec))

    # TODO: make getting positions easier
    travel_list = []
    for ii in range(len(l_damper)):
        nlp = bike_kinematics.construct_nlp(
            l_fork=l_fork[ii], l_damper=l_damper[ii])
        x, info = nlp.solve(x)
        x_array[ii] = x

        # record rear axle
        system.set_states(x)

        rear_axle_position = system.find_point(
            'rear_triangle', 'rear_axle').get_position()
        bb_position = system.find_point(
            'front_triangle', 'bottom_bracket').get_position()
        relative_travel = rear_axle_position - bb_position
        logger.debug(f'rear axle: {rear_axle_position}')
        logger.debug(f'bottom bracket: {bb_position}')
        logger.debug(f'wheel travel: {relative_travel[-1]}')

        travel_list.append(relative_travel)

    if create_animation:
        create_kinematic_animation(system, x_array, '5010_animation.mp4')

    xz = np.vstack(travel_list)

    return xz[:, 1]
