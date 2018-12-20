import autograd.numpy as np
from autograd import grad, jacobian
import ipopt

def dist_con(a, b, l):
    delta = np.array(a)-np.array(b)
    val = np.dot(delta, delta) - l**2
    # use norm can encounter singularity, due to sqrt derivatives
    # val = np.linalg.norm(delta) - l
    return val

class kinematics(object):
    def __init__(self):
        # data for Santa Cruz 5010 2019 small
        self.p_bb = [0, 0]
        self.p_lower_pivot = np.add(self.p_bb, [0.0462, 0.0462])
        self.p_rocker_pivot = np.add(self.p_bb, [-0.03, 0.3188])
        self.p_damper_pivot = np.add(self.p_bb, [0.2079, 0.3904])

        self.l_rocker_damper = 0.0947
        self.l_rocker_triangle = 0.0878
        self.l_rocker_middle = 0.0323

        self.l_triangle_hor = 0.4066
        self.l_triangle_vert = 0.2402
        self.l_triangle_diag = 0.4943
        self.l_lower_link = 0.097

        self.idx = {'ax': 0, # RA
                    'az': 1,
                    'bx': 2, # triangle lower
                    'bz': 3,
                    'cx': 4, # triangle upper
                    'cz': 5,
                    'dx': 6, # rocker damper size
                    'dz': 7,
                    'l_damper': 8}

        return

    def get_val(self, x, name):
        return x[self.idx[name]]

    def objective(self, x):
        #
        # The callback for calculating the objective
        #
        return 0*np.dot(x, x)

    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
        grad_fun = grad(self.objective)
        return grad_fun(x)

    def constraints(self, x):
        #
        # The callback for calculating the constraints
        #
        p_d = [self.get_val(x, 'dx'), self.get_val(x, 'dz')]
        p_c = [self.get_val(x, 'cx'), self.get_val(x, 'cz')]
        p_a = [self.get_val(x, 'ax'), self.get_val(x, 'az')]
        p_b = [self.get_val(x, 'bx'), self.get_val(x, 'bz')]

        con1 = dist_con(self.p_damper_pivot, p_d, self.get_val(x, 'l_damper'))
        con2 = dist_con(self.p_rocker_pivot, p_d, self.l_rocker_damper)
        con3 = dist_con(self.p_rocker_pivot, p_c, self.l_rocker_triangle)
        con4 = dist_con(p_d, p_c, self.l_rocker_middle)

        con5 = dist_con(self.p_lower_pivot, p_b, self.l_lower_link)
        con6 = dist_con(p_a, p_b, self.l_triangle_hor)
        con7 = dist_con(p_a, p_c, self.l_triangle_diag)
        con8 = dist_con(p_b, p_c, self.l_triangle_vert)
        con9 = p_c[0] - p_d[0]

        return np.array((con1, con2, con3, con4, con5, con6, con7, con8, con9))

    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian
        #
        jac_fun = jacobian(self.constraints)
        return jac_fun(x)

    # def hessianstructure(self):
    #     #
    #     # The structure of the Hessian
    #     # Note:
    #     # The default hessian structure is of a lower triangular matrix. Therefore
    #     # this function is redundant. I include it as an example for structure
    #     # callback.
    #     #
    #
    #     return np.nonzero(np.tril(np.ones((4, 4))))

    # def hessian(self, x, lagrange, obj_factor):
    #     #
    #     # The callback for calculating the Hessian
    #     #
    #     H = obj_factor*np.array((
    #             (2*x[3], 0, 0, 0),
    #             (x[3],   0, 0, 0),
    #             (x[3],   0, 0, 0),
    #             (2*x[0]+x[1]+x[2], x[0], x[0], 0)))
    #
    #     H += lagrange[0]*np.array((
    #             (0, 0, 0, 0),
    #             (x[2]*x[3], 0, 0, 0),
    #             (x[1]*x[3], x[0]*x[3], 0, 0),
    #             (x[1]*x[2], x[0]*x[2], x[0]*x[1], 0)))
    #
    #     H += lagrange[1]*2*np.eye(4)
    #
    #     row, col = self.hessianstructure()
    #
    #     return H[row, col]

    def intermediate(
            self,
            alg_mod,
            iter_count,
            obj_value,
            inf_pr,
            inf_du,
            mu,
            d_norm,
            regularization_size,
            alpha_du,
            alpha_pr,
            ls_trials
            ):

        #
        # Example for the use of the intermediate callback.
        #
        print("Objective value at iteration #%d is - %g" % (iter_count, obj_value))


def solve(l_damper):
    #
    # Define the problem
    #
    nlprob = kinematics()

    n_dec = 9
    n_con = 9
    x0 = np.zeros(n_dec)

    lb = -np.ones(n_dec)
    ub = np.ones(n_dec)
    ub[nlprob.idx['ax']] = 0
    ub[nlprob.idx['bx']] = 0

    # fix damper length
    lb[-1] = l_damper
    ub[-1] = l_damper

    cl = np.zeros(n_con)
    cu = np.zeros(n_con)
    # last constraint is inequality on cx - dx <=0
    cl[n_con-1] = -1

    nlp = ipopt.problem(
                n=len(x0),
                m=len(cl),
                problem_obj=nlprob,
                lb=lb,
                ub=ub,
                cl=cl,
                cu=cu
                )

    #
    # Set solver options
    #
    #nlp.addOption('derivative_test', 'second-order')
    nlp.addOption('mu_strategy', 'adaptive')
    nlp.addOption('tol', 1e-7)

    #
    # Scale the problem (Just for demonstration purposes)
    #
    nlp.setProblemScaling(
        obj_scaling=2,
        x_scaling=np.ones(n_dec)
        )
    nlp.addOption('nlp_scaling_method', 'user-scaling')

    #
    # Solve the problem
    #
    x, info = nlp.solve(x0)

    print("Solution of the primal variables: x=%s\n" % repr(x))

    print("Solution of the dual variables: lambda=%s\n" % repr(info['mult_g']))

    print("Objective=%s\n" % repr(info['obj_val']))

    return nlprob.idx, x

def main():
    import matplotlib.pyplot as plt
    n_point = 51
    damper_eye2eye = 0.21
    damper_stroke = 0.05
    damper_travel = np.linspace(0, damper_stroke, n_point)
    l_damper = damper_eye2eye - damper_travel

    nlprob = kinematics()
    _, x = zip(*[solve(l) for l in l_damper])
    idx = nlprob.idx

    x = np.array(x)

    # Draw the bicycle kinematics



    plt.figure(2)
    z_ra = x[:, idx['az']]
    # plt.plot(damper_travel, z_ra - z_ra[0])
    motion_ratio = np.diff(z_ra)/np.diff(damper_travel)
    plt.plot(z_ra[0:-1] - z_ra[0], motion_ratio)
    plt.xlabel("wheel vertical travel [m]")
    plt.ylabel("motion ratio [-]")

# animation ================================
    from matplotlib import animation
    fig = plt.figure(3, figsize=(10,6))

    ax = plt.axes(xlim=(-1, 1), ylim=(-1, 1))
    triangle1 = ax.plot([], [], 'b')[0]
    triangle2 = ax.plot([], [], 'r')[0]
    lowerlink = ax.plot([], [], 'r')[0]
    damper = ax.plot([], [], 'g')[0]

    ax.scatter(*nlprob.p_bb, label='bb')
    ax.scatter(*nlprob.p_lower_pivot, label='lower pivot')
    ax.scatter(*nlprob.p_rocker_pivot, label='rocker pivot')
    ax.scatter(*nlprob.p_damper_pivot, label='damper pivot')

    ax.scatter(x[:, idx['ax']], x[:, idx['az']], label='rear axle')
    ax.scatter(x[:, idx['bx']], x[:, idx['bz']], label='triangle lower link')
    ax.scatter(x[:, idx['cx']], x[:, idx['cz']], label='triangle upper link')
    ax.scatter(x[:, idx['dx']], x[:, idx['dz']], label='rocker damper link')
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    plt.axis('equal')
    plt.legend()


    data_triangle1 = np.array(([x[:, idx['ax']], x[:, idx['bx']], x[:, idx['cx']]], [x[:, idx['az']], x[:, idx['bz']], x[:, idx['cz']]]))
    fixed_point = np.array([nlprob.p_rocker_pivot for _ in range(np.size(x,0))])
    data_triangle2 = np.array(([x[:, idx['cx']], x[:, idx['dx']], fixed_point[:,0]], [x[:, idx['cz']], x[:, idx['dz']], fixed_point[:, 1]]))
    p_damper_pivot = np.array([nlprob.p_damper_pivot for _ in range(np.size(x,0))])
    p_d = x[:, [idx['dx'], idx['dz']]]
    p_lower_pivot = np.array([nlprob.p_lower_pivot for _ in range(np.size(x,0))])
    p_b = x[:, [idx['bx'], idx['bz']]]


    def init():
        triangle1.set_xdata(np.empty([1, 2]))
        triangle1.set_ydata(np.empty([1, 2]))
        triangle2.set_xdata(np.empty([1, 2]))
        triangle2.set_ydata(np.empty([1, 2]))
        lowerlink.set_xdata(np.empty([1, 2]))
        lowerlink.set_ydata(np.empty([1, 2]))
        damper.set_xdata(np.empty([1, 2]))
        damper.set_ydata(np.empty([1, 2]))

        return triangle1, triangle2, lowerlink, damper

    def update_triangle(triangle, data, i):
        datax = np.array([data[0, 0, i], data[0, 1, i], data[0, 2, i], data[0, 0, i]])
        datay = np.array([data[1, 0, i], data[1, 1, i], data[1, 2, i], data[1, 0, i]])
        triangle.set_xdata(datax)
        triangle.set_ydata(datay)
        return triangle

    def update_line(line, p1, p2, i):
        datax = np.array((p1[i, 0], p2[i, 0]))
        datay = np.array((p1[i, 1], p2[i, 1]))
        line.set_xdata(datax)
        line.set_ydata(datay)
        return line

    def animate(i):
        t1 = update_triangle(triangle1, data_triangle1, i)
        t2 = update_triangle(triangle2, data_triangle2, i)
        l1 = update_line(lowerlink, p_b, p_lower_pivot, i)
        l2 = update_line(damper, p_d, p_damper_pivot, i)

        # datax = np.array([x[i, idx['ax']], x[i, idx['bx']], x[i, idx['cx']], x[i, idx['ax']]])
        # datay = np.array([x[i, idx['az']], x[i, idx['bz']], x[i, idx['cz']], x[i, idx['az']]])
        # triangle1.set_xdata(datax)
        # triangle1.set_ydata(datay)
        return t1, t2, l1, l2

    t_total = 3
    t_interval = t_total*1000/n_point
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=np.size(x, 0),
                                   interval=t_interval, blit=False, repeat=False)

    anim.save('5010_kinematics.mp4')

# animation end ==========================================


    plt.show()

    return


if __name__ == '__main__':
    main()
