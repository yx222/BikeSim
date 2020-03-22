import logging
import numpy as np
from io import StringIO

from bottle import route, request, response, run, template, static_file, abort
from bikesim.simulation.suspension_simulation import simulate_damper_sweep
from bikesim.models.multibody import MultiBodySystem
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)


def simulate_and_plot(damper_stroke):
    system_file = 'geometries/5010.json'
    n_point = 20
    sag = np.linspace(0, 1, n_point)
    wheel_travel = simulate_damper_sweep(
        sag=sag, damper_stroke=damper_stroke,
        system=MultiBodySystem.from_json(system_file))

    plt.plot(sag*100, wheel_travel, '-*')
    plt.xlabel('damper sag [%]')
    plt.ylabel('wheel travel [m]')
    buf = StringIO()
    plt.savefig(buf, format='svg')
    return buf.getvalue()


@route('/')
def index():
    return static_file('index.html', root="./")


@route('/plot.svg')
def index():
    damper_stroke = request.query.damper_stroke
    try:
        damper_stroke = float(damper_stroke)
        response.content_type = 'image/svg+xml'
        return simulate_and_plot(damper_stroke)
    except ValueError:
        abort(404, "Invalid probability: " + damper_stroke)


run(host='0.0.0.0', port=8090)
