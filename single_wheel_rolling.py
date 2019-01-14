# import numpy as np
import autograd.numpy as np
from autograd import grad, jacobian
from scipy.integrate import odeint
from matplotlib import animation
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import optimize
# from sympy import *
from scipy import misc

# custom modules
import roadmodel


def get_force_from_field():

    return

def get_force_from_contact_surface(qb, qc, v, r, k):
    distance_bc = np.linalg.norm(qb - qc)
    compression = r - distance_bc
    if compression < 0:
        k = -k*1e-6  # a spring to help the wheel return to the ground when it's in the air

    spring_force = compression * k * (qb - qc) / distance_bc

    cd = 0.4
    drag_force = -0.5*cd*v*np.linalg.norm(v)

    force = spring_force + drag_force
    return (force, spring_force, drag_force)

def get_closest_point(r, road_func, qb):
    version = 1
    if version == 1:
        qc = get_closest_point_v1(r, road_func['f'], qb)
    elif version == 2:
        qc = get_closest_point_v2(r, road_func, qb)

    return qc

# TO BE OPTIMIZED
def get_closest_point_v1(r, z_func, qb):
    # TIME CONSUMING, SHOULD BE DONE FASTER]
    npoint = 100
    x_start = qb[0]
    xs = np.linspace(x_start-2*r, x_start+2*r, npoint)
    surface = np.array([xs, z_func(xs)])
    dist_2 = np.sum((surface - qb[:, np.newaxis]) ** 2, axis=0)
    qc_idx = np.argmin(dist_2)
    qc = surface[:, qc_idx]
    return qc

def get_closest_point_v2(r, road_func, qb):
    x_start = qb[0]

    # sol = optimize.root(func_to_solve, args=(z_func, qb), x0=x_start)
    # x_sol = sol.x

    window = 0.01
    npoint = 100

    # x_sol = optimize.bisect(func_to_solve, x_start, x_start+window, args=(z_func, qb))

    x_start = qb[0]
    xx = np.linspace(x_start-window, x_start+window, npoint)
    yy = [func_to_solve(x, road_func, qb) for x in xx]
    qc_idx = np.argmin(np.absolute(yy))
    x_sol = xx[qc_idx]

    qc = np.array((x_sol, road_func['f'](x_sol)))

    return qc

def func_to_solve(x, road_func, qb):
    fval = road_func['dfdx'](x) * (road_func['f'](x) - qb[1]) - qb[0] + x
    return fval

def single_wheel_ode(q_and_p, t, road_func, r, k, m, g):
    x = q_and_p[0]
    z = q_and_p[1]
    vx = q_and_p[2]
    vz = q_and_p[3]

    q = np.array([x, z])
    v = np.array([vx, vz])

    qc = get_closest_point(r, road_func, q)
    force, *__ = get_force_from_contact_surface(q, qc, v, r, k)

    dxdt = vx
    dzdt = vz
    dvxdt = force[0]/m
    dvzdt = force[1]/m - g

    print("running ode at x={:3f}".format(x))

    return [dxdt, dzdt, dvxdt, dvzdt]

def aug_single_wheel_ode(state, t, road_spline, r, k, m, g):
    x = state[0]
    z = state[1]
    vx = state[2]
    vz = state[3]
    s = state[4]
    n = state[5]

    fx, fz = roadmodel.get_force(s, n, road_spline, k)

    # get curvilinear road property from splines
    # zeta is the angle of the x-y coordinate w.r.t. curvilinear coordinate angle theta.
    # Since x-y is inertia, zeta = -theta
    zeta = -road_spline['theta'](s)
    kappa = road_spline['kappa'](s)

    dxdt = vx
    dzdt = vz
    dvxdt = fx/m
    dvzdt = fz/m - g
    dsdt = (vx*np.cos(zeta) - vz*np.sin(zeta))/(1 - n*kappa)
    dndt = vx*np.sin(zeta) + vz*np.cos(zeta)

    return [dxdt, dzdt, dvxdt, dvzdt, dsdt, dndt]

def generate_circle_data(c, r):
    theta = np.linspace(0, 2 * np.pi, 51)
    datax = c[0] + r*np.cos(theta)
    datay = c[1] + r*np.sin(theta)
    circle_data = np.array([datax, datay])
    return circle_data

def update_connected_dots(ha, data, i):
    if data.ndim == 2:
        data = np.expand_dims(data, axis=2)

    ha.set_xdata(data[i, 0, :])
    ha.set_ydata(data[i, 1, :])
    return

def animate_one_frame(i, h_list, list_of_data):
    for ha_idx, ha in enumerate(h_list):
        update_connected_dots(ha, list_of_data[ha_idx], i)
    return

def run_animation(fig, h_list, list_of_data, frame_number):
    t_total = 10
    t_interval = t_total*1000/frame_number # [ms]
    anim = animation.FuncAnimation(fig, animate_one_frame, fargs=(h_list, list_of_data), frames=frame_number,
                                   interval=t_interval, blit=False, repeat=False)
    return anim

# main function ================================================
def main():
    # constants
    R = 0.4  # [m]
    K = 3.5E4 # [N/m]
    M = 2   # total weight of the person and the wheel [kg]
    G = 9.8 # [m/s**2]

    # surface definition
    surface_offset = 1

    step_height = 0.15*2

    def z_func(x):
        # zs = np.linspace(0, 0, xs.size) + surface_offset
        # zs = np.cos(xs/5 * 2 * np.pi)
        return (np.tanh((x-2)*20)+1)*step_height/2 + surface_offset

    road_func = {'f': z_func,
                 'dfdx': grad(z_func),
                 'd2fdx2': grad(grad(z_func))}

    npoint = 100
    x_min = -1
    x_max = 5
    xs = np.linspace(x_min, x_max, npoint)
    zs = z_func(xs)
    surface = np.array([xs, zs])

    # initial status
    qb_0 = np.array([-1, R-M*G/K + surface_offset])
    vb_0 = np.array([5, 0])

    # time steps for the ODE solver
    t_steps_number = 100
    t = np.linspace(0, 2, t_steps_number)

    # solve ODE ========================
    mode = 2
    if mode == 1:
        state_0 = np.concatenate((qb_0, vb_0), axis=None)

        atol = np.array([1E-3, 1E-6, 1E-2, 1E-6])*1E-6
        rtol = 1E-9
        state = odeint(single_wheel_ode, state_0, t, args=(road_func, R, K, M, G), h0=0.001, hmax=0.001, rtol=rtol, atol=atol)

    else:
        road_spline = roadmodel.build_force_field(road_func['f'], x_min, x_max, R)

        s0 = 0
        n0 = -M*G/K
        x0, z0 = roadmodel.sn2xz(s0, n0, road_spline, x0=x_min, z0=z_func(x_min)+R)
        state_0 = np.concatenate((x0, z0, vb_0, s0, n0), axis=None)
        print(state_0.shape)

        atol = np.array([1E-3, 1E-6, 1E-2, 1E-6, 1E-3, 1E-3]) * 1E-6
        rtol = 1E-9
        state = odeint(aug_single_wheel_ode, state_0, t, args=(road_spline, R, K, M, G), h0=0.001, hmax=0.001, rtol=rtol,
                       atol=atol)

    # reorganize ode solution data structure
    qb = state[:, (0, 1)]
    vb = state[:, (2, 3)]

    # get the force and closest point at each time step ===================
    q_closest = np.array([get_closest_point(R, road_func, q) for q in qb])
    force, spring_force, drag_force = zip(*[get_force_from_contact_surface(qb, qc, vb, R, K) for qb, qc, vb in zip(qb, q_closest, vb)])
    force = np.array(force)
    spring_force = np.array(spring_force)
    drag_force = np.array(drag_force)


    # plots ==============================================
    rim_width = 6
    lim_buffer = 2
    xmin = np.min(surface, axis=1)[0] - lim_buffer
    xmax = np.max(surface, axis=1)[1] + lim_buffer
    ymin = np.min(force[:, 1]) - lim_buffer
    ymax = np.max(force[:, 1]) + lim_buffer

    fig = plt.figure(1)
    ax = plt.axes(xlim=(xmin, xmax), ylim=(ymin, ymax))
    wheel_center_traj = ax.plot(qb[:, 0], qb[:, 1], 'k:')[0]
    surface_line = ax.plot(xs, zs, 'k-')[0]
    plt.axis('equal')
    plt.title('wheel simulation')
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")

    # initiate plot objects for animation
    h_wheel = ax.plot([], [], 'k', linewidth=rim_width)[0]
    h_wheel_center = ax.plot([], [], 'r.')[0]
    h_closest_point = ax.plot([], [], 'r.')[0]
    h_force_vector = ax.plot([], [], 'k-')[0]
    # h_wheel_velocity = ax.plot([], [], 'b-')[0]
    h_drag_force_vector = ax.plot([], [], 'g-')[0]
    h_spring_force_vector = ax.plot([], [], 'b-')[0]


    # prepare data for animation function
    wheel_data = np.stack([generate_circle_data(c, R) for c in qb])  # wheel_data.shape returns (n_time_steps, x_and_y=2, number of points at each step)
    force_vector = np.stack((qb, force + qb), axis=2) # force_vector.shape returns (n_time_steps, x_and_y=2, number of points at each step)
    wheel_velocity_vector = np.stack((qb, vb + qb), axis=2)
    spring_force_vector = np.stack((qb, spring_force + qb), axis=2)
    drag_force_vector = np.stack((qb, drag_force + qb), axis=2)
    list_of_data = [wheel_data, qb, q_closest, force_vector, spring_force_vector] #, drag_force_vector]  # list of arrays with different dimensions
    h_list = [h_wheel, h_wheel_center, h_closest_point, h_force_vector, h_spring_force_vector] #, h_drag_force_vector]

    # run animation
    anim = run_animation(fig, h_list, list_of_data, t_steps_number)

    plt.show()
    # end plot =================================================

    anim.save('rolling_wheel.mp4')

    return



if __name__ == '__main__':
    main()
