#!/opt/conda/bin/python
from webtest import TestApp
import logging
import numpy as np
from io import StringIO

from bottle import Bottle, route, request, response, run, template, static_file, abort
from bikesim.simulation.suspension_simulation import simulate_damper_sweep
from bikesim.models.kinematics import BikeKinematics
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

app = Bottle()


def simulate_and_plot(damper_travel):
    bike_file = 'geometries/5010_bike.json'
    n_point = 20
    sag_array = np.linspace(0, 1, n_point)

    bike = BikeKinematics.from_json(bike_file)
    bike.damper_travel = damper_travel

    wheel_travel = simulate_damper_sweep(sag_array=sag_array, bike=bike)

    plt.plot(sag_array*100, wheel_travel, '-*')
    plt.xlabel('suspension sag [%]')
    plt.ylabel('wheel travel [m]')
    buf = StringIO()
    plt.savefig(buf, format='svg')
    return buf.getvalue()


@app.route('/')
def index():
    return static_file('html/index.html', root="./")


@app.route('/plot.svg')
def index():
    damper_travel = request.query.damper_travel
    try:
        damper_travel = float(damper_travel)
        response.content_type = 'image/svg+xml'
        return simulate_and_plot(damper_travel)
    except ValueError:
        abort(404, "Invalid damper travel: " + damper_travel)


def main():
    run(app, host='0.0.0.0', port=8080)


if __name__ == '__main__':
    main()
