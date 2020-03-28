import unittest
import logging
from webtest import TestApp

from apps.webserver import app

logger = logging.getLogger(__name__)


class TestWebServer(unittest.TestCase):
    """
    Test Visualization of animation, plots etc...
    """

    def setUp(self):
        self.app = app

    def test_web(self):
        test_app = TestApp(app)
        resp = test_app.get('/')
        print(resp)
        # TODO: this is not ideal. We should test the button click
        # But somehow resp.clickbutton(buttonid='plot) doesn't work
        # Problems with Jquery dynamically created button?
        resp = test_app.get('/plot.svg?damper_travel=0.05')
        print(resp)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    unittest.main()
