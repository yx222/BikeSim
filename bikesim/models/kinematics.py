import autograd.numpy as np
from autograd import grad, jacobian
import ipopt
from io import StringIO
import json
from sys import platform as sys_pf
if sys_pf == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")

def dist_con(a, b, l):
    delta = np.array(a)-np.array(b)
    val = np.dot(delta, delta) - l**2
    # use norm can encounter singularity, due to sqrt derivatives
    # val = np.linalg.norm(delta) - l
    return val

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
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def geometry_from_json(json_file):

        # data for Santa Cruz 5010 2019 small
        with open(json_file, 'r') as f:
            geometry_dict = json.load(f)

        for key, value in geometry_dict.items():
            if value is list:
                geometry_dict[key] = np.asarray(value)

        return Geometry(geometry_dict)        

class Geometry:
    def __init__(self, geometry_dict):
        # TODO: convert data format to json
        # TODO: generliaze to 3d, x fwd, y to the left, and z up
        # TODO: add checks to ensure geometry file is valid

        self.__dict__ = geometry_dict

        return

    def save(self, file_name):
        with open(file_name, 'w') as f:
            json.dump(self.__dict__, f, cls=NumpyEncoder, indent=1)

class Kinematics:
    def __init__(self, geometry):
        self.idx = {'ax': 0, # rear axle point
                    'az': 1,
                    'bx': 2, # triangle lower pivot point (on lower link)
                    'bz': 3,
                    'cx': 4, # triangle upper pivot point (on rocker)
                    'cz': 5,
                    'dx': 6, # damper point on rocker
                    'dz': 7,
                    'l_damper': 8}

        self.geometry = geometry

        return

    def get_val(self, x, name):
        return x[self.idx[name]]

    def objective(self, x):
        #
        # The callback for calculating the objective
        #
        return 0.0

    def gradient(self, x):
        #
        # The callback for calculating the gradient
        #
        grad_fun = grad(self.objective)
        return grad_fun(x)

    def constraints(self, x):
        # The callback for calculating the constraints
        geometry = self.geometry

        p_d = [self.get_val(x, 'dx'), self.get_val(x, 'dz')]
        p_c = [self.get_val(x, 'cx'), self.get_val(x, 'cz')]
        p_a = [self.get_val(x, 'ax'), self.get_val(x, 'az')]
        p_b = [self.get_val(x, 'bx'), self.get_val(x, 'bz')]

        con1 = dist_con(geometry.p_damper_pivot, p_d, self.get_val(x, 'l_damper'))
        con2 = dist_con(geometry.p_rocker_pivot, p_d, geometry.l_rocker_damper)
        con3 = dist_con(geometry.p_rocker_pivot, p_c, geometry.l_rocker_triangle)
        con4 = dist_con(p_d, p_c, geometry.l_rocker_middle)

        con5 = dist_con(geometry.p_lower_pivot, p_b, geometry.l_lower_link)
        con6 = dist_con(p_a, p_b, geometry.l_triangle_hor)
        con7 = dist_con(p_a, p_c, geometry.l_triangle_diag)
        con8 = dist_con(p_b, p_c, geometry.l_triangle_vert)

        # this constraint is required to ensure that we find the correct solution (there are multiple feasible points)
        con9 = p_c[0] - p_d[0]

        return np.array((con1, con2, con3, con4, con5, con6, con7, con8, con9))


    def jacobian(self, x):
        #
        # The callback for calculating the Jacobian
        #
        jac_fun = jacobian(self.constraints)
        return jac_fun(x)


def solve(l_damper, x0):
    #
    # Define the problem
    #
    nlprob = Kinematics(geometry=geometry_from_json('geometries/5010_large.json'))

    n_dec = 9
    n_con = 9

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

    # Set solver options
    #nlp.addOption('derivative_test', 'second-order')
    nlp.addOption('mu_strategy', 'adaptive')
    nlp.addOption('tol', 1e-7)
    nlp.addOption('print_level', 0)

    # Scale the problem (Just for demonstration purposes)
    #
    nlp.setProblemScaling(
        obj_scaling=2,
        x_scaling=np.ones(n_dec)
        )
    nlp.addOption('nlp_scaling_method', 'user-scaling')

    # Solve the problem
    x, info = nlp.solve(x0)

    print("Solution of the primal variables: x=%s\n" % repr(x))

    print("Solution of the dual variables: lambda=%s\n" % repr(info['mult_g']))

    print("Objective=%s\n" % repr(info['obj_val']))

    return nlprob.idx, x

def rx201(dl_stroke=0.05):
    import matplotlib.pyplot as plt
    n_point = 20
    damper_eye2eye = 0.21
    damper_stroke = dl_stroke
    damper_travel = np.linspace(0, damper_stroke, n_point)
    l_damper = damper_eye2eye - damper_travel

    nlprob = Kinematics(geometry=geometry_from_json('geometries/5010_large.json'))
    #FIXME: silly hardcode
    n_dec = 9
    x = np.zeros((n_point, n_dec))

    # we don't use list comprehension because we want to warm start
    sol = np.zeros(n_dec)
    for ii in range(l_damper.size):
        idx, sol = solve(l_damper[ii], sol)
        print("damper length {0}: RA height {1} \n".format(l_damper[ii], sol[idx["az"]]))
        x[ii, :] = sol

    x = np.array(x)
    ax = x[:, idx['ax']]
    az = x[:, idx['az']]

    plt.plot(ax, az, '-*')
    buf = StringIO()
    plt.savefig(buf, format='svg')
    return buf.getvalue()

def main():
    import matplotlib.pyplot as plt
    n_point = 20
    damper_eye2eye = 0.21
    damper_stroke = 0.05
    damper_travel = np.linspace(0, damper_stroke, n_point)
    l_damper = damper_eye2eye - damper_travel

    nlprob = Kinematics(geometry=geometry_from_json('geometries/5010_large.json'))
    #FIXME: silly hardcode
    n_dec = 9
    x = np.zeros((n_point, n_dec))

    # we don't use list comprehension because we want to warm start
    sol = np.zeros(n_dec)
    for ii in range(l_damper.size):
        idx, sol = solve(l_damper[ii], sol)
        print("damper length {0}: RA height {1} \n".format(l_damper[ii], sol[idx["az"]]))
        x[ii, :] = sol

    x = np.array(x)

    # Draw the bicycle kinematics
    # reassign data
    geometry = nlprob.geometry

    z_ra = x[:, idx['az']]
    p_d = x[:, [idx['dx'], idx['dz']]]
    p_c = x[:, [idx['cx'], idx['cz']]]
    p_b = x[:, [idx['bx'], idx['bz']]]
    p_rocker_pivot = np.array([geometry.p_rocker_pivot for _ in range(np.size(x,0))])
    p_damper_pivot = np.array([geometry.p_damper_pivot for _ in range(np.size(x,0))])
    p_lower_pivot = np.array([geometry.p_lower_pivot for _ in range(np.size(x,0))])
    p_front_frame_bb = np.array([geometry.p_front_frame_bb for _ in range(np.size(x, 0))])
    p_front_frame_downtube_low = np.array([geometry.p_front_frame_downtube_low for _ in range(np.size(x, 0))])
    p_front_frame_headtube_low = np.array([geometry.p_front_frame_headtube_low for _ in range(np.size(x, 0))])
    p_front_frame_headtube_high = np.array([geometry.p_front_frame_headtube_high for _ in range(np.size(x, 0))])
    p_front_frame_seatstay_low = np.array([geometry.p_front_frame_seatstay_low for _ in range(np.size(x, 0))])
    p_front_frame_seatstay_high = np.array([geometry.p_front_frame_seatstay_high for _ in range(np.size(x, 0))])
    p_rear_wheel_centre = x[:, [idx['ax'], idx['az']]]
    p_front_wheel_centre = np.array([geometry.p_front_wheel_centre for _ in range(np.size(x, 0))])

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
    ax = plt.axes(xlim=(x_min-buffer, x_max+buffer), ylim=(z_min-buffer, z_max+buffer))
    plt.title('5010 rear kinematics')
    # plt.axis('equal')

    ax.imshow(im, extent=[x_min, x_max, z_min, z_max])

    frame_width = 2
    frame_color = 'purple'
    rim_width = 2
    link_width = 2
    main_frame = ax.plot([],[], 'k', linewidth=frame_width, color=frame_color)[0]
    triangle1 = ax.plot([], [], 'b', linewidth=frame_width, color=frame_color)[0]
    triangle2 = ax.plot([], [], 'b', linewidth=link_width, color='black')[0]
    lower_link = ax.plot([], [], 'g')[0]
    damper = ax.plot([], [], 'g')[0]
    rear_wheel = ax.plot([], [], 'k', linewidth=rim_width)[0]
    front_wheel = ax.plot([], [], 'k', linewidth=rim_width)[0]
    ha_list = (triangle1, triangle2, lower_link, damper, rear_wheel, main_frame, front_wheel)

    ax.scatter(*geometry.p_bottom_bracket, label='bb')
    ax.scatter(*geometry.p_lower_pivot, label='lower pivot')
    ax.scatter(*geometry.p_rocker_pivot, label='rocker pivot')
    ax.scatter(*geometry.p_damper_pivot, label='damper pivot')
    ax.scatter(*geometry.p_front_wheel_centre, label='front wheel centre')

    ax.plot(p_rear_wheel_centre[:, 0], p_rear_wheel_centre[:,1], 'r--', label='rear axle')
    ax.plot(p_b[:, 0], p_b[:,1], 'r--', label='triangle lower link')
    ax.plot(p_c[:, 0], p_c[:,1], 'r--', label='triangle upper link')
    ax.plot(p_d[:, 0], p_d[:,1], 'r--', label='rocker damper link')
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
        t1 = update_connected_dots(triangle1, np.stack((p_rear_wheel_centre, p_b, p_c, p_rear_wheel_centre), axis=1), i)
        t2 = update_connected_dots(triangle2, np.stack((p_c, p_d, p_rocker_pivot, p_c), axis=1), i)
        l1 = update_connected_dots(lower_link, np.stack((p_b, p_lower_pivot), axis=1), i)
        l2 = update_connected_dots(damper, np.stack((p_d, p_damper_pivot), axis=1), i)
        c1 = update_circle(rear_wheel, p_rear_wheel_centre, i)
        f1 = update_connected_dots(main_frame, np.stack((p_front_frame_seatstay_low, p_front_frame_bb, p_front_frame_downtube_low, p_front_frame_headtube_low, p_front_frame_headtube_high, p_front_frame_seatstay_low, p_front_frame_seatstay_high, p_front_frame_seatstay_low), axis=1), i)
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
