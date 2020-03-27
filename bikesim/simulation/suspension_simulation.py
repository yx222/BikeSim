import logging
from bikesim.models.kinematics import BikeKinematics
from bikesim.models.multibody import MultiBodySystem
from bikesim.utils.visualization import create_kinematic_animation
import numpy as np

logger = logging.getLogger(__name__)


def simulate_damper_sweep(sag_array,  bike: BikeKinematics,
                          damper_stroke=0.05, create_animation: bool = False):
    """
    Sweep sag solve for geometry. Assuming same sag front and rear
    """

    # Solve the problem
    x = bike.get_init_guess()
    x_array = np.zeros((len(sag_array), bike.n_dec))

    # TODO: make getting positions easier
    travel_list = []
    for ii, sag in enumerate(sag_array):
        x, info = bike.solve(sag_front=sag, init_guess=x)
        x_array[ii] = x

        # record rear axle
        bike.system.set_states(x)

        rear_axle_position = bike.system.find_point(
            'rear_triangle', 'rear_axle').get_position()
        bb_position = bike.system.find_point(
            'front_triangle', 'bottom_bracket').get_position()
        relative_travel = rear_axle_position - bb_position
        logger.debug(f'rear axle: {rear_axle_position}')
        logger.debug(f'bottom bracket: {bb_position}')
        logger.debug(f'wheel travel: {relative_travel[-1]}')

        travel_list.append(relative_travel)

    if create_animation:
        create_kinematic_animation(
            bike.system, x_array, '5010_animation.mp4')

    xz = np.vstack(travel_list)

    return xz[:, 1]
