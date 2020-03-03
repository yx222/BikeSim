import unittest
import logging
import os
from parameterized import parameterized
from bikesim.simulation.suspension_simulation import simulate_damper_sweep

logging.getLogger().setLevel(logging.WARN)


class TestKinematics(unittest.TestCase):
    """
    Test a bunch of different arrangements of kinematics.
    eg: VPP high pivot, low pivot, horst link, etc
    """
    def setUp(self):
        self.geometry_table = {"VPP_High": '/l5010.json',
                    "VPP_Low": "Bronson.json"}
        self.geometry_dir = os.path.join(os.getcwd(), "geometries")

    @parameterized.expand([
        ["VPP_High", 'legacy/5010.json'],
        ["VPP_Low", 'legacy/Bronson.json']
    ])
    def test_damper_sweep(self, name, file_name):
        geometry_file = os.path.join(self.geometry_dir, file_name)
        logging.info(f'simulating {name} geometry from: {geometry_file}')
        x, z = simulate_damper_sweep(l_damper=0.21, geometry_file=geometry_file)
         
if __name__ == '__main__':
    unittest.main()