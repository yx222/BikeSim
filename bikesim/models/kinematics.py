import autograd.numpy as np
from autograd import grad, jacobian
import ipopt
import logging

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


class Kinematics(NonlinearProblem):
    def __init__(self, system: MultiBodySystem):
        # number of decision variables
        self.system = system
        self.n_dec = system.num_dof

        # Call the constraints once to get number of cons
        self.n_con = len(system.evaluate_constraints())

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
        self.lb = -np.ones(self.n_dec)*np.pi
        self.ub = 2*np.ones(self.n_dec)

        # WHY? constrain x position of rear axle and lower link pivot on
        # rear triangle to be -ve
        # self.ub[self.idx['ax']] = 0
        # self.ub[self.idx['bx']] = 0

        # for key, val in kwargs.items():
        #     if np.isscalar(val):
        #         # equality bounds can be passed in as a scalar
        #         val = [val, val]
        #     logging.debug(f'{key}: {val}')
        #     logging.debug(f'adding bounds:  {val[0]} <= {key} <= {val[1]}')
        #     self.lb[self.idx[key]] = val[0]
        #     self.ub[self.idx[key]] = val[1]

    def set_cons(self):
        self.cl = np.zeros(self.n_con)
        self.cu = np.zeros(self.n_con)

    def construct_nlp(self, l_fork, l_damper):

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
        #nlp.addOption('derivative_test', 'second-order')
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
        return nlp


class BikeKinematics(Kinematics):
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

    def __init__(self):
        pass
