import numpy as np
# from autograd import grad, jacobian
# import ipopt
from scipy.integrate import odeint
from matplotlib import animation
import matplotlib.pyplot as plt
from scipy import interpolate


def get_force_from_contact_surface(qb, qc, r, k):
    distance_bc = np.sqrt(np.sum((qb - qc) ** 2, axis=0))
    compression = r - distance_bc
    if compression < 0:
        k = -k*1e-6 # a spring to help the wheel return to the ground when it's in the airå

    force = compression * k * (qb - qc) / distance_bc

    return force

# TO BE OPTIMIZED
def get_closest_point(surface, qb):
    # TIME CONSUMING, SHOULD BE DONE FASTER
    tck = interpolate.splrep(surface[0,:], surface[1,:])
    x2 = np.linspace(surface[0,1], surface[0, -1], 2*surface.shape[1])
    y2 = interpolate.splev(x2, tck)
    surface2 = np.array([x2, y2])
    dist_2 = np.sum((surface2 - qb[:, np.newaxis]) ** 2, axis=0)
    qc_idx = np.argmin(dist_2)
    qc = surface2[:, qc_idx]
    return qc

def single_wheel_ode(q_and_p, t, surface, r, k, m, g):
    x = q_and_p[0]
    z = q_and_p[1]
    vx = q_and_p[2]
    vz = q_and_p[3]

    q = np.array([x, z])
    qc = get_closest_point(surface, q)
    force = get_force_from_contact_surface(q, qc, r, k)

    dxdt = vx
    dzdt = vz
    dvxdt = force[0]/m
    dvzdt = force[1]/m - g

    return [dxdt, dzdt, dvxdt, dvzdt]

def plot_circle(circle, c, r):
    theta = np.linspace(0, 2*np.pi, 201)
    datax = c[0] + r*np.cos(theta)
    datay = c[1] + r*np.sin(theta)
    circle.set_xdata(datax)
    circle.set_ydata(datay)
    return circle

def main():
    # constants
    R = 0.4  # [m]
    K = 3.5E4 # [N/m]
    M = 2   # total weight of the person and the wheel [kg]
    G = 9.8 # [m/s**2]

    # surface definition
    surface_offset = 1
    xs = np.linspace(0, 5, 100)
    zs = np.linspace(0, 0, xs.size) + surface_offset
    # zs = np.sin(xs/20 * 2 * np.pi)
    surface = np.array([xs, zs])

    # initial status
    qb_0 = np.array([1.2, 0.38+1 + surface_offset])
    vb_0 = np.array([1, 0])
    q_and_p0 = np.concatenate((qb_0, vb_0), axis=None)

    max_deflection = np.sqrt((0.5*M*vb_0[1]**2 + G*qb_0[1]*M)*2/K)

    print(max_deflection)

    # time steps for the ODE solver
    t = np.linspace(0, 6, 150)

    # check the closest point and the force ==============================
    qc = get_closest_point(surface, qb_0)
    print(qc)
    force = get_force_from_contact_surface(qb_0, qc, R, K)
    print(force)
    # end check =====================================================

    # solve ODE ========================
    atol = np.array([1E-3, 1E-6, 1E-2, 1E-6])*1E-6
    rtol = 1E-9
    q_and_p = odeint(single_wheel_ode, q_and_p0, t, args=(surface, R, K, M, G), h0=0.01, rtol=rtol, atol=atol)

    # plots ==============================================
    rim_width = 2

    fig = plt.figure(1)
    ax = plt.axes(xlim=(0, 5), ylim=(-3, 8))
    surface_line = ax.plot(xs, zs, 'k-')[0]
    wheel = ax.plot([], [], 'k', linewidth=rim_width)[0]
    plot_circle(wheel, qb_0, R)
    closet_point = ax.plot(qc[0], qc[1], 'r.')[0]
    force_vector = ax.plot([qc[0], force[0]], [qc[1], force[1]], 'b-')[0]
    wheel_center_traj = ax.plot(q_and_p[:, 0], q_and_p[:, 1], 'k-*')[0]

    plt.show()
    # end plot =================================================

    return



if __name__ == '__main__':
    main()
