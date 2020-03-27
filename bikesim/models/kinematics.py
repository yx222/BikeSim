import autograd.numpy as np
from autograd import grad, jacobian
import ipopt
import logging
import json
from typing import Optional

from bikesim.models.multibody import MultiBodySystem

logger = logging.getLogger(__name__)


class NonlinearProblem():
    """
    Base class for nonlinear optimization problem required for ipopt
    """

    def __init__(self):
        pass

    def objective(self, x):
        pass

    def gradient(self, x):
        return grad(self.objective)(x)

    def constraints(self, x):
        pass

    def jacobian(self, x):
        return jacobian(self.constraints)(x)


class BikeKinematics(NonlinearProblem):
    """
    Store and manipulate MTB kinematics problems.

    Free variables managed are:
    - design varibles:
        - key geometry data, such as BB position, headtube angle, etc
        - component, such as the damper eye-to-eye, damper travel, etc
    - state variables:
        - current sag of fork and damper
        - potentially tyre deformation as well?

    We should be able to easily:
    - for a fixed design variable (given a bike), solve for current state.
        eg, what's the motion ratio for a range of sag sweep
    - for a fixed state, optimize some criteria to decide on design
        eg, what length should my lower link be if I want to maximize
        anti-squat, while keeping the motion ratio constant at 30% sag?

    """

    def __init__(self, system: MultiBodySystem, damper_eye_to_eye: float, damper_travel: float, fork_travel: float):
        self.system = system
        self.damper_eye_to_eye = damper_eye_to_eye
        self.damper_travel = damper_travel
        self.fork_travel = fork_travel
        self.nlp = None

        self.n_dec = system.num_dof
        # Call the constraints once to get number of cons
        self.n_con = len(system.evaluate_constraints())

    @classmethod
    def from_json(cls, json_file: str):
        # json data contains bodies, constraints as for a MultibodySystem, and components that is speicifc to a bike.
        with open(json_file, 'r') as f:
            data = json.load(f)
            components = data.pop('components')
            system = MultiBodySystem.from_dict(data)
            return cls(system, **components)

    def objective(self, x):
        #
        # The callback for calculating the objective
        #
        # We want a constant, but a constant will throw warning
        # (output independent of input)
        return np.sum(x)*0

    def constraints(self, x):
        # The callback for calculating the constraints
        self.system.set_states(x)

        return self.system.evaluate_constraints()

    def get_init_guess(self):
        return np.zeros(self.n_dec)

    def set_bounds(self, **kwargs):
        # TODO: better bound setting. This is ugly hard-code
        self.lb = -np.ones(self.n_dec)*np.pi
        self.ub = 2*np.ones(self.n_dec)

    def set_cons(self):
        # Only equality constraints for now.
        self.cl = np.zeros(self.n_con)
        self.cu = np.zeros(self.n_con)

    def construct_nlp(self, sag_front: float, sag_rear: Optional[float] = None):
        """Construct a NLP based on sags provided

        sag_front: front fork compression [0, 1], 1 for fully extended
        sag_rear: rear shock compression [0, 1], 1 for fully extended

        if no rear sag is provided, rear sag takes the same value as the front. 
        """
        if sag_rear is None:
            sag_rear = sag_front

        l_damper = self.damper_eye_to_eye - self.damper_travel*sag_rear
        l_fork = self.fork_travel * (1-sag_front)

        self.system.constraints['damper_length'].distance = l_damper
        self.system.constraints['fork_shaft_length'].distance = l_fork
        self.set_bounds()
        self.set_cons()

        nlp = ipopt.problem(
            n=self.n_dec,
            m=self.n_con,
            problem_obj=self,
            lb=self.lb,
            ub=self.ub,
            cl=self.cl,
            cu=self.cu
        )

        # Set solver options
        # nlp.addOption('derivative_test', 'second-order')
        nlp.addOption('mu_strategy', 'adaptive')
        nlp.addOption('tol', 1e-4)
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            print_level = 5
        else:
            print_level = 0
        nlp.addOption('print_level', print_level)

        # Scale the problem (Just for demonstration purposes)
        #
        nlp.setProblemScaling(
            obj_scaling=2,
            x_scaling=np.ones(self.n_dec)
        )
        nlp.addOption('nlp_scaling_method', 'user-scaling')

        self.nlp = nlp

    def solve(self, sag_front: float, sag_rear: Optional[float] = None, init_guess=None):
        self.construct_nlp(sag_front=sag_front, sag_rear=sag_rear)
        if init_guess is None:
            init_guess = self.get_init_guess()

        x, info = self.nlp.solve(init_guess)
        return x, info
