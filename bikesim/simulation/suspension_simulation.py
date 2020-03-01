import logging
from bikesim.models.kinematics import Kinematics, geometry_from_json
import numpy as np

def simulate_damper_sweep(l_damper=0.21, geometry_file='geometries/5010.json'):
    """
    Sweep the eye-to-eye damper length and solve for geometry.
    """
    bike_kinematics = Kinematics(geometry=geometry_from_json(geometry_file))

    # Solve the problem
    x = bike_kinematics.get_init_guess()

    # sweep through damper length, make it iterable if it's scalar
    if np.isscalar(l_damper):
        l_damper = [l_damper]

    x_array = np.zeros((len(l_damper), bike_kinematics.n_dec))
    
    for ii, l in enumerate(l_damper):
        nlp = bike_kinematics.construct_nlp(l_damper=l)
        x, info = nlp.solve(x)
        x_array[ii] = x

    return x_array[:, bike_kinematics.idx['ax']], x_array[:, bike_kinematics.idx['az']]

if __name__ == "__main__":
    simulate_damper_sweep(l_damper=0.21)