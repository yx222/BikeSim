import unittest
import logging
import os
import numpy as np
from parameterized import parameterized
from bikesim.simulation.suspension_simulation import simulate_damper_sweep
from bikesim.models.multibody import MultiBodySystem

logging.getLogger().setLevel(logging.WARN)


class TestKinematics(unittest.TestCase):
    """
    Test a bunch of different arrangements of kinematics.
    eg: VPP high pivot, low pivot, horst link, etc
    """

    def setUp(self):
        self.geometry_dir = os.path.join(os.getcwd(), "geometries")

    @parameterized.expand([
        ["VPP_High", '5010.json']
    ])
    def test_damper_sweep(self, name, file_name):
        system_file = os.path.join(self.geometry_dir, file_name)
        logging.info(f'simulating {name} geometry from: {system_file}')
        x, z = simulate_damper_sweep(
            sag=np.linspace(0, 1, 21),
            system=MultiBodySystem.from_json(system_file))


if __name__ == '__main__':
    unittest.main()
