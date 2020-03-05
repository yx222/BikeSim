import autograd.numpy as np
from autograd import grad, jacobian
import ipopt
import json
import logging
from sys import platform as sys_pf

from bikesim.models.multibody import MultiBodySystem

if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

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
        pass

    def constraints(self, x):
        pass

    def jacobian(self, x):
        pass


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

    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
        grad_fun = grad(self.objective)
        return grad_fun(x)

    def constraints(self, x):
        # The callback for calculating the constraints
        self.system.set_states(x)

        return self.system.evaluate_constraints()

    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian
        #
        jac_fun = jacobian(self.constraints)
        return jac_fun(x)

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


def animate():
    import matplotlib.pyplot as plt
    n_point = 20
    damper_eye2eye = 0.21
    damper_stroke = 0.05
    damper_travel = np.linspace(0, damper_stroke, n_point)
    l_damper = damper_eye2eye - damper_travel

    nlprob = Kinematics(geometry=geometry_from_json(
        'geometries/legacy/5010.json'))
    # FIXME: silly hardcode
    n_dec = 9
    x = np.zeros((n_point, n_dec))

    # we don't use list comprehension because we want to warm start
    sol = np.zeros(n_dec)
    for ii in range(l_damper.size):
        idx, sol = solve(l_damper[ii], sol)
        print("damper length {0}: RA height {1} \n".format(
            l_damper[ii], sol[idx["az"]]))
        x[ii, :] = sol

    x = np.array(x)

    # Draw the bicycle kinematics
    # reassign data
    geometry = nlprob.geometry

    z_ra = x[:, idx['az']]
    p_d = x[:, [idx['dx'], idx['dz']]]
    p_c = x[:, [idx['cx'], idx['cz']]]
    p_b = x[:, [idx['bx'], idx['bz']]]
    p_rocker_pivot = np.array(
        [geometry.p_rocker_pivot for _ in range(np.size(x, 0))])
    p_damper_pivot = np.array(
        [geometry.p_damper_pivot for _ in range(np.size(x, 0))])
    p_lower_pivot = np.array(
        [geometry.p_lower_pivot for _ in range(np.size(x, 0))])
    p_front_frame_bb = np.array(
        [geometry.p_front_frame_bb for _ in range(np.size(x, 0))])
    p_front_frame_downtube_low = np.array(
        [geometry.p_front_frame_downtube_low for _ in range(np.size(x, 0))])
    p_front_frame_headtube_low = np.array(
        [geometry.p_front_frame_headtube_low for _ in range(np.size(x, 0))])
    p_front_frame_headtube_high = np.array(
        [geometry.p_front_frame_headtube_high for _ in range(np.size(x, 0))])
    p_front_frame_seatstay_low = np.array(
        [geometry.p_front_frame_seatstay_low for _ in range(np.size(x, 0))])
    p_front_frame_seatstay_high = np.array(
        [geometry.p_front_frame_seatstay_high for _ in range(np.size(x, 0))])
    p_rear_wheel_centre = x[:, [idx['ax'], idx['az']]]
    p_front_wheel_centre = np.array(
        [geometry.p_front_wheel_centre for _ in range(np.size(x, 0))])

    # plt.figure(1)
    # # plt.plot(damper_travel, z_ra - z_ra[0])
    # motion_ratio = np.diff(z_ra)/np.diff(damper_travel)
    # plt.plot(z_ra[0:-1] - z_ra[0], motion_ratio)
    # plt.xlabel("wheel vertical travel [m]")
    # plt.ylabel("motion ratio [-]")

    # animation ================================
    from matplotlib import animation

    im = plt.imread('img/geometry.png')
    im_height, im_width, _ = im.shape
    x_bb_im = 745.0
    z_bb_im = im_height - 1115.0
    # work out the extent of the image
    buffer = 0.4
    scale_im = 0.425/(745.0 - 81.5)
    x_min = (-x_bb_im)*scale_im
    x_max = (im_width - x_bb_im)*scale_im
    z_min = (-z_bb_im)*scale_im
    z_max = (im_height - z_bb_im)*scale_im

    # fig = plt.figure(3, figsize=(im_width/50, im_height/50))
    fig = plt.figure(figsize=(12, 8))
    ax = plt.axes(xlim=(x_min-buffer, x_max+buffer),
                  ylim=(z_min-buffer, z_max+buffer))
    plt.title('5010 rear kinematics')
    # plt.axis('equal')

    ax.imshow(im, extent=[x_min, x_max, z_min, z_max])

    frame_width = 2
    frame_color = 'purple'
    rim_width = 2
    link_width = 2
    main_frame = ax.plot(
        [], [], 'k', linewidth=frame_width, color=frame_color)[0]
    triangle1 = ax.plot([], [], 'b', linewidth=frame_width,
                        color=frame_color)[0]
    triangle2 = ax.plot([], [], 'b', linewidth=link_width, color='black')[0]
    lower_link = ax.plot([], [], 'g')[0]
    damper = ax.plot([], [], 'g')[0]
    rear_wheel = ax.plot([], [], 'k', linewidth=rim_width)[0]
    front_wheel = ax.plot([], [], 'k', linewidth=rim_width)[0]
    ha_list = (triangle1, triangle2, lower_link, damper,
               rear_wheel, main_frame, front_wheel)

    ax.scatter(*geometry.p_bottom_bracket, label='bb')
    ax.scatter(*geometry.p_lower_pivot, label='lower pivot')
    ax.scatter(*geometry.p_rocker_pivot, label='rocker pivot')
    ax.scatter(*geometry.p_damper_pivot, label='damper pivot')
    ax.scatter(*geometry.p_front_wheel_centre, label='front wheel centre')

    ax.plot(p_rear_wheel_centre[:, 0],
            p_rear_wheel_centre[:, 1], 'r--', label='rear axle')
    ax.plot(p_b[:, 0], p_b[:, 1], 'r--', label='triangle lower link')
    ax.plot(p_c[:, 0], p_c[:, 1], 'r--', label='triangle upper link')
    ax.plot(p_d[:, 0], p_d[:, 1], 'r--', label='rocker damper link')
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    # plt.legend()

    def init():
        [init_one(ha) for ha in ha_list]
        return ha_list

    def init_one(ha):
        ha.set_xdata(np.empty([1, 2]))
        ha.set_ydata(np.empty([1, 2]))
        pass

    def update_circle(circle, c, i, r=0.34925):
        theta = np.linspace(0, 2*np.pi, 201)
        datax = c[i, 0] + r*np.cos(theta)
        datay = c[i, 1] + r*np.sin(theta)
        circle.set_xdata(datax)
        circle.set_ydata(datay)
        return circle

    def update_connected_dots(object, data, i):
        object.set_xdata(data[i, :, 0])
        object.set_ydata(data[i, :, 1])
        return object

    def animate(i):
        t1 = update_connected_dots(triangle1, np.stack(
            (p_rear_wheel_centre, p_b, p_c, p_rear_wheel_centre), axis=1), i)
        t2 = update_connected_dots(triangle2, np.stack(
            (p_c, p_d, p_rocker_pivot, p_c), axis=1), i)
        l1 = update_connected_dots(
            lower_link, np.stack((p_b, p_lower_pivot), axis=1), i)
        l2 = update_connected_dots(damper, np.stack(
            (p_d, p_damper_pivot), axis=1), i)
        c1 = update_circle(rear_wheel, p_rear_wheel_centre, i)
        f1 = update_connected_dots(main_frame, np.stack((p_front_frame_seatstay_low, p_front_frame_bb, p_front_frame_downtube_low, p_front_frame_headtube_low,
                                                         p_front_frame_headtube_high, p_front_frame_seatstay_low, p_front_frame_seatstay_high, p_front_frame_seatstay_low), axis=1), i)
        c2 = update_circle(front_wheel, p_front_wheel_centre, i)

        return t1, t2, l1, l2, c1, f1, c2

    t_total = 3
    t_interval = t_total*1000/n_point
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=np.size(x, 0),
                                   interval=t_interval, blit=False, repeat=False)

    anim.save('5010_kinematics.mp4')

# animation end ==========================================

    plt.show()

    return anim


if __name__ == '__main__':
    main()
