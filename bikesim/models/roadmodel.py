import autograd.numpy as np
from autograd import grad, jacobian
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, PchipInterpolator


def get_cc_to_tangent(x, y, slope, radius):
    cc_x = np.zeros(np.size(x))
    cc_y = np.zeros(np.size(x))
    for idx, slopei in enumerate(slope):
        norm_vec = np.array((-slopei, 1))
        norm_vec = norm_vec / np.sqrt(np.sum(norm_vec ** 2))
        delta = norm_vec * radius
        cc_x[idx] = delta[0] + x[idx]
        cc_y[idx] = delta[1] + y[idx]

    return cc_x, cc_y


def is_local_minimum(x, z, z_func, r):
    # for a point at x, z, check if it is just a local minimum point w.r.t. the road
    # The check is performed by checking if there are points in the road surface with distance to (x,z)<radius
    #
    # the check should be valid if: 1) the grid point is fine enough and 2) the z_func is 1-1 mapping
    n_point = 100
    xs = np.linspace(x-r, x+r, n_point)
    zs = z_func(xs)

    dist = np.sqrt((xs-x)**2 + (zs-z)**2)

    return any(dist<r)


def get_distance(x, z):
    # compute the curvilinear coordinate quantities
    dx = np.diff(x)
    dz = np.diff(z)
    ds = np.sqrt(dx**2 + dz**2)
    s = np.cumsum(np.insert(ds, 0, 0))
    return s


def get_force(s, n, ndot, road_spline, k, c):
    # compute the force in cartesian coordinate, for a wheel at s, n location in the curvilinear force field
    # k is spring rate in N/m, and is +ve. So the force must be opposite to deflection.
    # c is the viscous term N*s/m, where F = -lambda*v

    # also for n > 0, the force should be zero! (no deflection, wheel in the air), but we put in a small term to
    # help the numerics (so it still sees the gradient in the n direciton , even when in the air!)

    if n > 0:
        k = k*1e-6
        c = c*1e-6

    theta = road_spline['theta'](s)

    f = -n*k - c*ndot
    fx = f*np.sin(-theta)
    fz = f*np.cos(-theta)

    return fx, fz


def sn2xz(s, n, road_spline):
    x0 = road_spline['x'](s)
    z0 = road_spline['z'](x0)

    theta = road_spline['theta'](s)
    x = x0 + np.sin(-theta)*n
    z = z0 + np.cos(-theta)*n

    return x, z


def build_force_field(z_func, x_min, x_max, r):
    # return the spline z = f(x) that represents the 0 force contour in the contact force field

    # TODO: constant? (pass from outside?)
    x_step = r/5
    # BUG: n_point must be an integer
    n_point = int(np.ceil((x_max - x_min)/x_step))

    # get gradient using autograd
    dzdx = grad(z_func)

    # get the tangent line of each surface point
    x_surf = np.linspace(x_min, x_max, n_point)

    slope = [dzdx(x) for x in x_surf]
    z_surf = z_func(x_surf)
    cc_x, cc_z = get_cc_to_tangent(x_surf, z_surf, slope, r)

    # identify the points that shouldn't be on the profile
    b_local_min = [is_local_minimum(x, z, z_func, r) for x, z in zip(cc_x, cc_z)]
    b_global_min = np.logical_not(b_local_min)

    # create cubic spline for the forcefield centerline
    x_f0 = cc_x[b_global_min]
    z_f0 = cc_z[b_global_min]

    # pchip is a lot smoother than cubic, which can be wiggly
    # TODO: find something that can be differentiated with autograd!
    zspline = PchipInterpolator(x_f0, z_f0)

    # build (s, kappa) spline

    # first resample uniformly from the spline
    xs = np.linspace(x_min, x_max, n_point)
    zs = zspline(xs)

    # compute the curvilinear coordinate quantities
    dx = np.diff(xs)
    dz = np.diff(zs)
    theta = np.arctan2(dz, dx)
    theta = np.append(theta, theta[-1])
    ds = np.sqrt(dx**2 + dz**2)
    s = np.cumsum(np.insert(ds, 0, 0))

    # plt.figure(101)
    # plt.plot(s, theta)
    # plt.show()

    # compute theta spline
    thetaspline = PchipInterpolator(s, theta)

    # x as a function of s
    xspline = PchipInterpolator(s, xs)

    # kappa is just the 1st derivative of theta w.r.t. s
    def kappaspline(ss):
        return thetaspline(ss, 1)

    # put everything in a dictinonary
    road_spline = {'z':     zspline,
                   'x':     xspline,
                   'theta': thetaspline,
                   'kappa': kappaspline}

    return road_spline

def get_road(x_min, x_max, r):
    # surface definition
    # manually assigned value, further work needs to be done
    surface_offset = 1
    step_height = 0.1
    def z_func(x):
        # zs = np.linspace(0, 0, xs.size) + surface_offset
        # zs = np.cos(x * 5 * np.pi)*0.3
        zs = (np.tanh((x-2)*20)+1)*step_height/2 + surface_offset
        return zs

    road_spline = build_force_field(z_func, x_min, x_max, r)
    return road_spline


def main():
    r = 0.4
    x_min = 1.5
    x_max = 2.5
    road_spline = get_road(x_min, x_max, r)

    n_point = 100
    x_surf = np.linspace(x_min, x_max, n_point)
    z_surf = road_spline['z'](x_surf)

    x_fine = np.linspace(x_min, x_max, n_point*10)
    z_fine = road_spline['z'](x_fine)

    s_fine = get_distance(x_fine, z_fine)

    # plot
    plt.figure(1)

    plt.plot(x_surf, z_surf, 'r')
    plt.plot(x_fine, z_fine, 'gx')

    plt.figure(2)
    plt.plot(s_fine, road_spline['theta'](s_fine))
    plt.plot(s_fine, road_spline['kappa'](s_fine), 'r')

    plt.show()


if __name__ == '__main__':
    main()









