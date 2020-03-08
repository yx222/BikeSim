import autograd.numpy as np
from autograd import grad, jacobian
import ipopt
import logging

from bikesim.models.multibody import MultiBodySystem

logging.getLogger().setLevel(logging.INFO)


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
        # We want a constant, but a constant will throw warning (output independent of input)
        return np.sum(x)*0

    def constraints(self, x):
        # The callback for calculating the constraints
        self.system.set_states(x)

        return self.system.evaluate_constraints()

    def get_init_guess(self):
        return np.zeros(self.n_dec)

    def set_bounds(self, **kwargs):
        self.lb = -np.ones(self.n_dec)
        self.ub = np.ones(self.n_dec)

        # WHY? constrain x position of rear axle and lower link pivot on rear triangle
        # to be -ve
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

    def construct_nlp(self, l_damper):

        self.system.constraints['damper_length'].distance = l_damper
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
        nlp.addOption('tol', 1e-7)
        nlp.addOption('print_level', 0)

        # Scale the problem (Just for demonstration purposes)
        #
        nlp.setProblemScaling(
            obj_scaling=2,
            x_scaling=np.ones(self.n_dec)
        )
        nlp.addOption('nlp_scaling_method', 'user-scaling')
        return nlp
