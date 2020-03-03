import unittest
import logging
import os
import tempfile
import json
import filecmp
from parameterized import parameterized
from bikesim.models.multibody import MultiBodySystem

logging.getLogger().setLevel(logging.INFO)


class TestSystem(unittest.TestCase):
    def setUp(self):
        self.geometry_dir = os.path.join(os.getcwd(), "geometries")

    @parameterized.expand([
        ['VPP_High', '5010.json'],
        ['VPP_Low', 'Bronson.json']
        ])
    def test_load_system(self, name, file_name):
        """
        Test loading a multibody system from a json file.
        """
        json_file = os.path.join(self.geometry_dir, file_name)
        system = MultiBodySystem.from_json(json_file)

    @parameterized.expand([
        ['VPP_High', '5010.json'],
        ['VPP_Low', 'Bronson.json']
        ])
    def test_save_system(self, name, file_name):
        """
        Test write a system to file
        """
        json_file = os.path.join(self.geometry_dir, file_name)
        system = MultiBodySystem.from_json(json_file)  

        with tempfile.NamedTemporaryFile() as f:
            system.save(f.name)
            assert(filecmp.cmp(json_file, f.name, shallow=False))


    def test_constraints(self):
        """

        """
        pass
         
if __name__ == '__main__':
    unittest.main()