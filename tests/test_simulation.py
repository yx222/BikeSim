import unittest
import logging
import os
from bikesim.simulation.suspension_simulation import simulate_damper_sweep

logging.getLogger().setLevel(logging.INFO)


class TestKinematics(unittest.TestCase):
    """
    Test a bunch of different arrangements of kinematics.
    eg: VPP high pivot, low pivot, horst link, etc
    """
    def setUp(self):
        self.geometry_table = {"VPP_High": '5010.json',
                    "VPP_Low": "Bronson.json"}
        self.geometry_dir = os.path.join(os.getcwd(), "geometries")


    def test_damper_sweep(self):
        for key, file_name in self.geometry_table.items():
            geometry_file = os.path.join(self.geometry_dir, file_name)
            logging.info(f'simulating {key} geometry from: {geometry_file}')
            x, z = simulate_damper_sweep(l_damper=0.21, geometry_file=geometry_file)
         
if __name__ == '__main__':
    unittest.main()