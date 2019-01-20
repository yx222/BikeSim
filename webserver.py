from bottle import route, request, response, run, template, static_file, abort
from kinematics import rx201


@route('/')
def index():
    return static_file('index.html', root="./")

@route('/plot.svg')
def index():
    prob = request.query.probability
    try:
        probability = float(prob)
        response.content_type = 'image/svg+xml'
        return rx201(probability)
    except ValueError:
        abort(404, "Invalid probability: " + prob)

run(host='localhost', port=8080)
