import unittest
import logging
import os
import numpy as np
from parameterized import parameterized
from bikesim.simulation.suspension_simulation import simulate_damper_sweep
from bikesim.models.multibody import MultiBodySystem
from bikesim.models.kinematics import BikeKinematics

logger = logging.getLogger(__name__)


class TestKinematicSimulation(unittest.TestCase):
    """
    Test simulations we run for kinematics
    """

    def setUp(self):
        self.geometry_dir = os.path.join(os.getcwd(), "geometries")

    @parameterized.expand([
        ["VPP_High", '5010_bike.json']
    ])
    def test_damper_sweep(self, name, file_name):
        bike_file = os.path.join(self.geometry_dir, file_name)
        my_bike = BikeKinematics.from_json(bike_file)

        sag_array = np.linspace(0, 1, 21)
        rel_wheel_travel = simulate_damper_sweep(sag_array, bike=my_bike)


if __name__ == '__main__':
    unittest.main()
