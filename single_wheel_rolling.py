import numpy as np
# from autograd import grad, jacobian
# import ipopt
from scipy.integrate import odeint
from matplotlib import animation
import matplotlib.pyplot as plt
from scipy import interpolate


def get_force_from_contact_surface(qb, qc, v, r, k):
    distance_bc = np.linalg.norm(qb - qc)
    compression = r - distance_bc
    if compression < 0:
        k = -k*1e-6  # a spring to help the wheel return to the ground when it's in the airå

    spring_force = compression * k * (qb - qc) / distance_bc

    cd = 0.4
    drag_force = -0.5*cd*v*np.linalg.norm(v)

    force = spring_force + drag_force
    return force

# TO BE OPTIMIZED
def get_closest_point(x, r, z_func, qb):
    # TIME CONSUMING, SHOULD BE DONE FASTER]
    npoint = 100
    xs = np.linspace(x-2*r, x+2*r, npoint)
    surface = np.array([xs, z_func(xs)])
    dist_2 = np.sum((surface - qb[:, np.newaxis]) ** 2, axis=0)
    qc_idx = np.argmin(dist_2)
    qc = surface[:, qc_idx]
    return qc

def single_wheel_ode(q_and_p, t, z_func, r, k, m, g):
    x = q_and_p[0]
    z = q_and_p[1]
    vx = q_and_p[2]
    vz = q_and_p[3]

    q = np.array([x, z])
    v = np.array([vx, vz])

    qc = get_closest_point(x, r, z_func, q)
    force = get_force_from_contact_surface(q, qc, v, r, k)

    dxdt = vx
    dzdt = vz
    dvxdt = force[0]/m
    dvzdt = force[1]/m - g

    return [dxdt, dzdt, dvxdt, dvzdt]

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

    step_height = 0.1

    def z_func(x):
        # zs = np.linspace(0, 0, xs.size) + surface_offset
        # zs = np.cos(xs/5 * 2 * np.pi)
        return (np.tanh((x-2)*10)+1)*step_height + surface_offset

    npoint = 100
    xs = np.linspace(0, 4, npoint)
    zs = z_func(xs)
    surface = np.array([xs, zs])

    # initial status
    qb_0 = np.array([0, R-M*G/K + surface_offset])
    vb_0 = np.array([10, 0])

    max_deflection = np.sqrt((0.5*M*vb_0[1]**2 + G*qb_0[1]*M)*2/K)
    print(max_deflection)

    # time steps for the ODE solver
    t_steps_number = 100
    t = np.linspace(0, 5, t_steps_number)

    # solve ODE ========================
    q_and_p0 = np.concatenate((qb_0, vb_0), axis=None)

    atol = np.array([1E-3, 1E-6, 1E-2, 1E-6])*1E-6
    rtol = 1E-9
    q_and_p = odeint(single_wheel_ode, q_and_p0, t, args=(z_func, R, K, M, G), h0=0.001, hmax=0.001, rtol=rtol, atol=atol)

    # reorganize ode solution data structure
    qb = q_and_p[:, (0, 1)]
    vb = q_and_p[:, (2, 3)]

    # get the force and closest point at each time step ===================
    q_closest = np.array([get_closest_point(q[0], R, z_func, q) for q in qb])
    force = np.array([get_force_from_contact_surface(qb, qc, vb, R, K) for qb, qc, vb in zip(qb, q_closest, vb)])

    # check the closest point and the force ==============================
    print("initial velocity vx={0:9.2f}[m/s], vy={1:9.2f}[m/s]".format(*vb[0, :]))
    print("first closest point x={0:9.2f}[m], y={1:9.2f}[m]".format(*q_closest[0, :]))
    print("first force vector fx={0:9.2f}[N], fy={1:9.2f}[N]".format(*force[0, :]))
    # end check =====================================================

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
    h_force_vector = ax.plot([], [], 'b-')[0]
    h_wheel_velocity = ax.plot([], [], 'b-')[0]

    # prepare data for animation function
    wheel_data = np.stack([generate_circle_data(c, R) for c in qb])  # wheel_data.shape returns (n_time_steps, x_and_y=2, number of points at each step)
    force_vector = np.stack((q_closest, force + q_closest), axis=2) # force_vector.shape returns (n_time_steps, x_and_y=2, number of points at each step)
    wheel_velocity_vector = np.stack((qb, vb + qb), axis=2)
    list_of_data = [wheel_data, qb, q_closest, force_vector, wheel_velocity_vector]  # list of arrays with different dimensions
    h_list = [h_wheel, h_wheel_center, h_closest_point, h_force_vector, h_wheel_velocity]

    # run animation
    anim = run_animation(fig, h_list, list_of_data, t_steps_number)

    plt.show()
    # end plot =================================================

    anim.save('rolling_wheel.mp4')

    return



if __name__ == '__main__':
    main()
