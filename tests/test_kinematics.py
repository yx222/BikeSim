import unittest
import logging
import os
import numpy as np
from parameterized import parameterized
from bikesim.simulation.suspension_simulation import simulate_damper_sweep
from bikesim.models.multibody import MultiBodySystem
from bikesim.models.kinematics import BikeKinematics

logger = logging.getLogger(__name__)


class TestBikeKinematics(unittest.TestCase):
    """
    Test a bunch of different arrangements of kinematics.
    eg: VPP high pivot, low pivot, horst link, etc
    """

    def setUp(self):
        self.geometry_dir = os.path.join(os.getcwd(), "geometries")

    @parameterized.expand([
        ["VPP_High", '5010_bike.json']
    ])
    def test_load_from_json(self, name, file_name):
        bike_file = os.path.join(self.geometry_dir, file_name)
        my_bike = BikeKinematics.from_json(bike_file)

    @parameterized.expand([
        ["VPP_High", '5010_bike.json']
    ])
    def test_sag_solve(self, name, file_name):
        bike_file = os.path.join(self.geometry_dir, file_name)
        my_bike = BikeKinematics.from_json(bike_file)

        x, info = my_bike.solve(sag_front=0.3, sag_rear=0.3)


if __name__ == '__main__':
    unittest.main()
