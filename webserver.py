import numpy as np
from io import StringIO

from bottle import route, request, response, run, template, static_file, abort
from bikesim.simulation.suspension_simulation import simulate_damper_sweep
import matplotlib.pyplot as plt

def simulate_and_plot(damper_stroke):
    n_point = 20
    damper_eye2eye = 0.21
    damper_travel = np.linspace(0, damper_stroke, n_point)
    l_damper = damper_eye2eye - damper_travel

    x_rear_axle, z_rear_axle = simulate_damper_sweep(l_damper=l_damper, system_file='geometries/5010.json')

    plt.plot(damper_travel, z_rear_axle, '-*')
    plt.xlabel('damper travel [m]')
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

run(host='0.0.0.0', port=8080)
