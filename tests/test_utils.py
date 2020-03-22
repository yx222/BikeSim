import unittest
import logging
import os
import numpy as np
from matplotlib import pyplot as plt
from parameterized import parameterized

from bikesim.simulation.suspension_simulation import simulate_damper_sweep
from bikesim.models.multibody import MultiBodySystem

logger = logging.getLogger(__name__)


class TestVisualization(unittest.TestCase):
    """
    Test Visualization of animation, plots etc...
    """

    def setUp(self):
        self.geometry_dir = os.path.join(os.getcwd(), "geometries")

    @parameterized.expand([
        ["VPP_High", '5010.json']
    ])
    def test_plot(self, name, file_name):
        system = MultiBodySystem.from_json(
            os.path.join(self.geometry_dir, file_name))
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes()

        system.plot(ax)
        fig.savefig('5010_geometry.png')

    @parameterized.expand([
        ["VPP_High", '5010.json']
    ])
    def test_animation(self, name, file_name):
        system_file = os.path.join(self.geometry_dir, file_name)
        logging.info(f'simulating {name} geometry from: {system_file}')
        wheel_travel = simulate_damper_sweep(
            sag=np.linspace(0, 1, 51),
            system=MultiBodySystem.from_json(system_file),
            create_animation=True)


if __name__ == '__main__':
    unittest.main()
