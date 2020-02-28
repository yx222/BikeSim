from bottle import route, request, response, run, template, static_file, abort
from kinematics import rx201


@route('/')
def index():
    return static_file('index.html', root="./")

@route('/plot.svg')
def index():
    damper_stroke = request.query.damper_stroke
    try:
        damper_stroke = float(damper_stroke)
        response.content_type = 'image/svg+xml'
        return rx201(damper_stroke)
    except ValueError:
        abort(404, "Invalid probability: " + damper_stroke)

# run(host='localhost', port=8080)
run(host='0.0.0.0', port=8080)
