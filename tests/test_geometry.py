import unittest
import logging
import os
from functools import wraps
from bikesim.models import kinematics

logging.getLogger().setLevel(logging.WARN)


class TestKinematics(unittest.TestCase):
    """
    Test a bunch of different arrangements of kinematics.
    eg: VPP high pivot, low pivot, horst link, etc
    """
    def setUp(self):
        self.geometry_table = {"VPP_High": '5010.json',
                    "VPP_Low": "Bronson.json"}
        self.geometry_dir = os.path.join(os.getcwd(), "geometries")


    def test_load_geometry(self):
        for key, file_name in self.geometry_table.items():
            geometry_file = os.path.join(self.geometry_dir, file_name)
            logging.info(f'loading {key} type geometry from: {geometry_file}')
            geo = kinematics.geometry_from_json(geometry_file)
         
if __name__ == '__main__':
    unittest.main()