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
        ['VPP_High', '5010.json']
    ])
    def test_load_system(self, name, file_name):
        """
        Test loading a multibody system from a json file.
        """
        json_file = os.path.join(self.geometry_dir, file_name)
        system = MultiBodySystem.from_json(json_file)

    @parameterized.expand([
        ['VPP_High', '5010.json']
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

    @parameterized.expand([
        ['VPP_High', '5010.json']
    ])
    def test_constraints(self, name, file_name):
        """

        """
        json_file = os.path.join(self.geometry_dir, file_name)
        system = MultiBodySystem.from_json(json_file)

        x = system.get_states()
        system.set_states(x)
        con = system.evaluate_constraints()
        logging.info(f'constraints has the shape of {con.shape}')

        assert(len(con.shape) == 1)


if __name__ == '__main__':
    unittest.main()
