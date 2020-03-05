"""
Things we want to do:

Rigid Body
- define rigid body: ref point (2d) + coordinate system (orientation)
- add reference points on a rigid body (defined as position in the body's coordinate system)
- add constraints between bodies using points. (eg, point A on rocker is coincidental as point B on the frame)

Point
- refer to points by name
- 2d points
- point belong to a rigid body (parent)

System
- a system that contains multiple rigid body, as well as connecitons between them
- has a global coordinate system where we define the position of the rigid bodies in the global system

"""
# forward reference to avoid type hint circular dependency
from __future__ import annotations

import autograd.numpy as np
from autograd import grad, jacobian
import os
import logging
import json
from enum import Enum

from matplotlib import pyplot as plt


logging.getLogger().setLevel(logging.INFO)


class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):  # This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class MultiBodySystem:
    def __init__(self, bodies=None):
        self.bodies = {}
        self.idx_dict = {}
        self.constraints = {}

        if bodies:
            for b in bodies:
                self.add_body(b)

    @property
    def num_body(self):
        return len(self.bodies)

    @property
    def num_dof(self):
        return self.num_body*3

    def add_body(self, rigid_body: RigidBody2D):
        # Here it only works for python version >3.6, where OrderedDict is used by default
        self.bodies[rigid_body.name] = rigid_body
        self.idx_dict[rigid_body.name] = len(self.bodies) - 1

    def set_states(self, states):
        """
        For a given vector, set all the D.O.F. of the system. 
        The D.O.F. are essentially the position and orientaiton of the rigid bodies.
        """

        assert(len(states) ==
               self.num_dof), f'Expecting {self.num_dof} D.O.F., got {len(x)}'

        states = np.reshape(states, (self.num_body, 3))
        for name, idx in self.idx_dict.items():
            pose = states[idx]
            self.bodies[name].pose = pose

    def get_states(self):
        states = np.zeros((self.num_body, 3))
        for name, idx in self.idx_dict.items():
            states[idx, :] = self.bodies[name].pose

        return states.flatten()

    def evaluate_constraints(self):
        con_list = []

        for con in self.constraints.values():
            con_list.append(con.evaluate())

        return np.hstack(con_list)

    def _add_constraint(self, constraint: Constraint):
        self.constraints[constraint.name] = constraint

    def connect_points(self, point_A: Point2D, point_B: Point2D):
        """
        create a constraint in the system, that point A and point B are coincidental
        """

        self._add_constraint(Coincident(point_A, point_B))

    def fix_point(self, point: Point2D, position):
        self._add_constraint(FixedPoint(point, position))

    def fix_orientation(self, body: RigidBody2D, orientation):
        self._add_constraint(FixedOrientation(body, orientation))

    def fix_distance(self, point_A: Point2D, point_B: Point2D, distance):
        self._add_constraint(FixedDistance(point_A, point_B, distance))

    def check_system_dof(self):
        """
        check if the system is over or underconstrained
        """
        num_dof = self.num_dof - np.sum([c.num_dof for c in self.constraints])

        if num_dof == 0:
            logging.info('System is fully constrained')
        elif num_dof < 0:
            logging.info(
                f'System is over-constrained by {-num_dof} constraints')
        else:
            logging.info(f'System is under-constrained with {num_dof} D.O.F.')

    def plot(self, ax):
        """
        Plot all bodies in the system
        """
        for name, body in self.bodies.items():
            body.plot(ax)

    def list_bodies(self):
        for b in self.bodies.values():
            print(b)

    def find_point(self, body_name, point_name):
        return self.bodies[body_name].points[point_name]

    def find_body(self, body_name):
        return self.bodies[body_name]

    def to_dict(self):

        out_dict = {}
        # rigid bodies
        out_dict['bodies'] = {name: body.to_dict()
                              for name, body in self.bodies.items()}

        # constraints - we don't serialize constraints since they're constructed on points, which we should only refer to by name
        out_dict['constraints'] = [c.to_dict()
                                   for c in self.constraints.values()]

        return out_dict

    @classmethod
    def from_dict(cls, input_dict):
        bodies_list = [RigidBody2D.from_dict(
            body_dict) for body_dict in input_dict['bodies'].values()]
        system = cls(bodies_list)

        # add constraints
        for c in input_dict['constraints']:
            logging.info(f'adding {c["type"]} constraint')
            con = ConstraintFactory.create_constraint(system, c)
            system._add_constraint(con)
        return system

    def save(self, json_file):
        with open(json_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)
        logging.info(f'finished writing multibody system to {json_file}')

    @classmethod
    def from_json(cls, json_file):
        with open(json_file, 'r') as f:
            return cls.from_dict(json.load(f))


class RigidBody2D:
    def __init__(self, name: str, pose=np.zeros(2)):
        """
        pose: 3d array [x, z, pitch]

        x fwd, z up, pitch is counter-clockwise
        """
        self.name = name
        self.pose = pose

        self.points = {}

    def add_point(self, name, rel_position):
        """
        add a new point onto the rigid body
        """
        self.points[name] = Point2D(name, self, rel_position)

    def list_points(self):
        for key, val in self.points.items():
            print(val)

    def plot(self, ax, **kwargs):
        """
        Plot a rigid body as a polygon.
        """
        xy = np.array([p.get_position() for p in self.points.values()])
        xy = np.vstack((xy, xy[0]))

        ax.plot(xy[:, 0], xy[:, 1],  linewidth=6)

    def __repr__(self):
        return f'RigidBody2D {self.name} at {self.pose}'

    def to_dict(self):
        """
        Convert data to a dictionary
        """
        out_dict = {'name': self.name,
                    'pose': self.pose,
                    'points': {name: point.to_dict() for name, point in self.points.items()}
                    }

        return out_dict

    @classmethod
    def from_dict(cls, input_dict):
        body = cls(input_dict['name'], input_dict['pose'])
        for name, point_dict in input_dict['points'].items():
            body.add_point(name, point_dict['rel_position'])
        return body


class Point2D:
    def __init__(self, name, parent: RigidBody2D, rel_position):
        self.parent = parent
        self.rel_position = rel_position
        self.name = name

    def get_position(self):
        pitch = self.parent.pose[2]
        c, s = np.cos(pitch), np.sin(pitch)
        rot_matrix = np.array(((c, -s), (s, c)))
        position = self.parent.pose[0:2] + \
            np.matmul(rot_matrix, self.rel_position)

        return position

    def __repr__(self):
        return f'Point2D {self.parent.name}.{self.name} at {self.get_position()}'

    def to_dict(self):
        return {'rel_position': self.rel_position}


class Constraint:
    name = None

    def __init__(self):
        pass

    def evaluate(self):
        pass

    def to_dict(self):
        pass

    def from_dict(self, system: MultiBodySystem, input_dict):
        pass


class ConstraintFactory:
    @staticmethod
    def create_constraint(system, input_dict):
        # We could use type directly instead of strings, but then it wouldn't be serializable.
        con_type = input_dict['type']
        data = input_dict['data']
        if con_type == 'Coincident':
            return Coincident.from_dict(system, data)
        elif con_type == 'FixedPoint':
            return FixedPoint.from_dict(system, data)
        elif con_type == 'FixedOrientation':
            return FixedOrientation.from_dict(system, data)
        elif con_type == 'FixedDistance':
            return FixedDistance.from_dict(system, data)
        else:
            raise NameError(f'{con_type} is not a valid constraint type')


class Coincident(Constraint):
    def __init__(self, A: Point2D, B: Point2D, name=None):
        """
        Point A and Point B are coincident. Meaning x, z are the same. 2 Outputs
        """
        self.num_dof = 2
        self.A = A
        self.B = B
        if not name:
            name = f'Coincident.{A.parent.name}.{A.name}.{B.parent.name}.{B.name}'
        self.name = name

    def evaluate(self):
        return self.A.get_position() - self.B.get_position()

    def to_dict(self):
        data = {'name': self.name, 'point_names': [
            (self.A.parent.name, self.A.name), (self.B.parent.name, self.B.name)]}
        return {'type': 'Coincident', 'data': data}

    @classmethod
    def from_dict(cls, system, input_dict):
        points = [system.find_point(*point_names)
                  for point_names in input_dict['point_names']]
        return cls(*points, name=input_dict['name'])


class FixedDistance(Constraint):
    def __init__(self, A: Point2D, B: Point2D, distance, name=None):
        """
        Point A and Point B are a fixed distance away. (but not on a rigid body)
        """
        self.num_dof = 1
        self.A = A
        self.B = B
        self.distance = distance
        if not name:
            name = f'Distance.{A.parent.name}.{A.name}.{B.parent.name}.{B.name}'
        self.name = name

    def evaluate(self):
        delta = self.A.get_position() - self.B.get_position()
        return np.dot(delta, delta) - self.distance**2

    def to_dict(self):
        data = {'name': self.name, 'point_names': [(self.A.parent.name, self.A.name), (self.B.parent.name, self.B.name)],
                'distance': self.distance}
        return {'type': 'FixedDistance', 'data': data}

    @classmethod
    def from_dict(cls, system, input_dict):
        points = [system.find_point(*names)
                  for names in input_dict['point_names']]
        return cls(*points, input_dict['distance'], name=input_dict['name'])


class FixedPoint(Constraint):
    def __init__(self, point: Point2D, position, name=None):
        """
        Fix a point at a global position
        """
        self.num_dof = 2
        self.point = point
        self.position = position
        if not name:
            name = f'FixedPoint.{point.parent.name}.{point.name}'
        self.name = name

    def evaluate(self):
        return self.point.get_position() - self.position

    def to_dict(self):
        data = {'name': self.name, 'point_name': (
            self.point.parent.name, self.point.name), 'position': self.position}
        return {'type': 'FixedPoint', 'data': data}

    @classmethod
    def from_dict(cls, system, input_dict):
        point = system.find_point(*input_dict['point_name'])
        return cls(point, input_dict['position'], name=input_dict['name'])


class FixedOrientation(Constraint):
    def __init__(self, body: RigidBody2D, orientation, name=None):
        self.num_dof = 1
        self.body = body
        self.orientation = orientation
        if not name:
            name = f'FixedOrientation.{body.name}'
        self.name = name

    def evaluate(self):
        return self.body.pose[2] - self.orientation

    def to_dict(self):
        data = {'name': self.name, 'body_name': self.body.name,
                'orientation': self.orientation}
        return {'type': 'FixedOrientation', 'data': data}

    @classmethod
    def from_dict(cls, system, input_dict):
        return cls(system.find_body(input_dict['body_name']), input_dict['orientation'], name=input_dict['name'])


def sample_usage():
    # Create system

    # Add front triangle
    #   Add lower link point
    #   Add rocker pivot point
    #   Add front axle (here we make a simplificiation of fixed fork length ==> in 2D front axle is rigidly attached on front triangle)

    # Add rear triangle
    #   Add rear axle point
    #   Add lower link point
    #   Add rocker point

    # Add rocker
    #   Add pivot point on frame
    #   Add damper point
    #   Add rear triangle point

    # Add lower link
    #   Add rear triangle point
    #   Add pivot point

    # Add damper
    #   Add pivot point
    #   Add rocker point

    # Create coincidental constraints

    # Create axle height constraint (Effective rolling radius above ground)
    return 0
