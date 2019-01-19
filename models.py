"""
This module contains a collection of models of dynamical systems
"""
# FIXME: there is a mixture of array or multiple scarlar representation of vector quantities, unify!
from matplotlib import pyplot as plt
from matplotlib import animation

import autograd.numpy as np

# custom modules
import roadmodel

# Constants
g = 9.81

class RigidBike(object):
    def __init__(self, road_spline):
        # states: px, pz, vx, vz, a_pitch, n_pitch (+ve pitch is nose-up)
        #         and front and rear tyre position in curvlinear coordinate (of the road forcefield)
        self.state_names = ['px', 'pz', 'vx', 'vz', 'a_pitch', 'n_pitch',
                            's_front', 'n_front', 's_rear', 'n_rear']
        self.state_enum = get_enum(self.state_names)
        self.nx = len(self.state_names)

        # control: fx, fz at pedal, handlebar
        self.control_names = ['fx_bar', 'fz_bar', 'fx_pedal', 'fz_pedal']
        self.control_enum = get_enum(self.control_names)
        self.nu = len(self.control_names)

        # centre of gravity
        self.cog = np.array([(0.77718859-0.423)/2*0, 0.43])

        # geometry info required: relative position of the wheels w.r.t. the c.o.g when a_pitch = 0
        self.position = {'front axle': np.array([0.77718859, 0.02053558]) - self.cog,
                         'rear axle': np.array([-0.423, 0.0156]) - self.cog}

        # some other properties
        self.m = 70 # [kg], including rider and wheels
        self.Iyy = 0.5*self.m*0.8**2 # [kgm^2] just a guess
        self.k = 5E4
        self.c = 500*0
        self.R = 0.4

        self.road_spline = road_spline

        self.lines = {}

        return

    def get_state(self, x, name):
        return x[self.state_enum[name]]

    def get_control(self, x, name):
        return x[self.control_enum[name]]

    def get_force_moment(self, x, u, s_front, n_front, dndt_front, s_rear, n_rear, dndt_rear):
        # fx_front, fz_front, fx_rear, fz_rear = self.get_wheel_force(x, u)

        # The forces here are in inertia coordiate, like the states
        fx_front, fz_front = roadmodel.get_force(s_front, n_front, dndt_front, self.road_spline, self.k, self.c)
        fx_rear, fz_rear = roadmodel.get_force(s_rear, n_rear, dndt_rear, self.road_spline, self.k, self.c)

        p_front = self.position['front axle']
        p_rear = self.position['rear axle']

        a_pitch = self.get_state(x, 'a_pitch')

        fx = fx_front + fx_rear
        fz = fz_front + fz_rear - self.m*g
        my = fz_front*p_front[0] - fx_front*p_front[1] + fz_rear*p_rear[0] - fx_rear*p_rear[1]

        # print("fx_f:{:.1f} fz_f:{:.1f} fx_r:{:.1f} fz_r:{:.1f} ".format(- fx_front*p_front[1], fz_front*p_front[0], - fx_rear*p_rear[1], fz_rear*p_rear[0]))

        # print("fx_f:{:.1f} fz_f:{:.1f} fx_r:{:.1f} fz_r:{:.1f} ".format(fx_front, fz_front, fx_rear, fz_rear))

        return fx, fz, my

    def get_mech_energy(self, x):
        pz = self.get_state(x, 'pz')
        vx = self.get_state(x, 'vx')
        vz = self.get_state(x, 'vz')
        n_pitch = self.get_state(x, 'n_pitch')

        mech_energy = 0.5*self.m*(vx**2 + vz**2) + 0.5*self.Iyy*n_pitch**2 + self.m*g*pz

        return mech_energy

    def call_model(self, x, u):
        xdot = np.zeros(self.nx)

        # extract variables
        px = self.get_state(x, 'px')
        pz = self.get_state(x, 'pz')
        vx = self.get_state(x, 'vx')
        vz = self.get_state(x, 'vz')
        a_pitch = self.get_state(x, 'a_pitch')
        n_pitch = self.get_state(x, 'n_pitch')

        s_front = self.get_state(x, 's_front')
        n_front = self.get_state(x, 'n_front')
        s_rear = self.get_state(x, 's_rear')
        n_rear = self.get_state(x, 'n_rear')

        # dxdt = vx
        xdot[self.state_enum['px']] = vx

        # dzdt = vz
        xdot[self.state_enum['pz']] = vz

        # da_pitch/dt = n_pitch
        xdot[self.state_enum['a_pitch']] = n_pitch

        # wheel curvilinear derivatives
        R_matrix = np.array([[np.cos(a_pitch), -np.sin(a_pitch)],
                     [np.sin(a_pitch),  np.cos(a_pitch)]])
        p_front = np.dot(R_matrix, self.position['front axle'])
        p_rear = np.dot(R_matrix, self.position['rear axle'])
        v_front = get_rigid_velocity(p_front, np.array((vx, vz)), n_pitch)
        v_rear = get_rigid_velocity(p_rear, np.array((vx, vz)), n_pitch)

        # kinematic derivatives
        dsdt_front, dndt_front = vxy2vsn(v_front[0], v_front[1], s_front, n_front, self.road_spline)
        dsdt_rear, dndt_rear = vxy2vsn(v_rear[0], v_rear[1], s_rear, n_rear, self.road_spline)

        # calculate the forces and moments on cog
        fx, fz, my = self.get_force_moment(x, u, s_front, n_front, dndt_front, s_rear, n_rear, dndt_rear)

        # dvx/dt = fx/m
        xdot[self.state_enum['vx']] = fx/self.m

        # dvz/dt = fz/m
        xdot[self.state_enum['vz']] = fz/self.m

        # dnpitch/dt = my/Iyy
        xdot[self.state_enum['n_pitch']] = my/self.Iyy

        xdot[self.state_enum['s_front']] = dsdt_front
        xdot[self.state_enum['n_front']] = dndt_front
        xdot[self.state_enum['s_rear']] = dsdt_rear
        xdot[self.state_enum['n_rear']] = dndt_rear

        # debug
        mech_energy = self.get_mech_energy(x)
        # print('mechanical energy is {:3f}kJ'.format(mech_energy))

        return xdot

    def render(self, x):
        if len(self.lines.keys()) == 0:
            # create the first axes
            # plt.plot returns a list of line object, even if the length is 1
            self.lines['front wheel'] = plt.plot([], [], 'k')[0]
            self.lines['rear wheel'] = plt.plot([], [], 'k')[0]
            self.lines['frame'] = plt.plot([], [], 'k')[0]
            self.lines['axis'] = plt.plot([], [], 'k')[0]

        # extract variables
        px = self.get_state(x, 'px')
        pz = self.get_state(x, 'pz')
        a_pitch = self.get_state(x, 'a_pitch')

        s_front = self.get_state(x, 's_front')
        n_front = self.get_state(x, 'n_front')
        s_rear = self.get_state(x, 's_rear')
        n_rear = self.get_state(x, 'n_rear')

        x_front, z_front = roadmodel.sn2xz(s_front, n_front, self.road_spline)
        x_rear, z_rear = roadmodel.sn2xz(s_rear, n_rear, self.road_spline)

        p_front = np.array((x_front, z_front))
        p_rear = np.array((x_rear, z_rear))
        p_cog = np.array((px, pz))
        front_wheel_data = generate_circle_data(p_front, self.R)
        rear_wheel_data = generate_circle_data(p_rear, self.R)
        frame_data = np.stack(([x_front, z_front], [x_rear, z_rear], p_cog, [x_front, z_front]), axis=1)
        half_axis = np.array((np.cos(a_pitch), np.sin(a_pitch)))
        axis_data = np.stack([p_cog - half_axis, p_cog + half_axis], axis=1)
        self.lines['front wheel'].set_data(front_wheel_data)
        self.lines['rear wheel'].set_data(rear_wheel_data)
        self.lines['frame'].set_data(frame_data)
        self.lines['axis'].set_data(axis_data)
        return


# =========================================== Helper Functions =========================================================
def generate_circle_data(c, r):
    theta = np.linspace(0, 2 * np.pi, 51)
    x = c[0] + r*np.cos(theta)
    y = c[1] + r*np.sin(theta)
    return np.array((x, y))

def get_enum(names):
    return dict(zip(names, range(len(names))))

def get_rigid_velocity(r, va, omega):
    # get velocity of position B, where the velocity of position A is v, and angular velcoity omega.
    # r is displacement from A to B

    return va + np.array((-r[1]*omega, r[0]*omega))

def vxy2vsn(vx, vz, s, n, road_spline):
    # converting cartesian velocity to curvilinear velocity
    zeta = -road_spline['theta'](s)
    kappa = road_spline['kappa'](s)
    dsdt = (vx * np.cos(zeta) - vz * np.sin(zeta)) / (1 - n * kappa)
    dndt = vx * np.sin(zeta) + vz * np.cos(zeta)
    return dsdt, dndt

# ======================================================================================================================

def run_fwd():
    from scipy.integrate import odeint
    import matplotlib.pyplot as plt

    # some constants
    R = 0.4  # [m]

    # surface definition
    surface_offset = 0
    step_height = R*0.5*4
    x_min = -1
    x_max = 20

    def z_func(x):
        # zs = np.linspace(0, 0, xs.size) + surface_offset
        # zs = np.sin(x/10 * 2 * np.pi)* step_height*2
        zs = (np.tanh((x-3)*1)+1)*step_height/2 + surface_offset
        return zs

    road_spline = roadmodel.build_force_field(z_func, x_min, x_max, R)

    # initialize bike model
    rigid_bike = RigidBike(road_spline)

    # set initial condition -- note that the number of states is greater than D.O.F, meaning we must set the states
    # carefuly to not violate any constraints (of rigid body). In essense, s_front, n_front, s_rear and n_rear are
    # redundant! But we use them in the ODE to avoid haivng to calculate them from x,y, which is hard (the other way is
    # easier)

    # FIXME: here we create a road that is flat at x = 0, and we know s = x-xmin, n = z, a_pitch = 0
    s_rear = 1
    n_rear = -rigid_bike.m*g/rigid_bike.k/2 + 0.0
    delta = rigid_bike.position['front axle'] - rigid_bike.position['rear axle']
    s_front = s_rear + delta[0]
    n_front = n_rear + delta[1]

    px = road_spline['x'](s_rear) - rigid_bike.position['rear axle'][0]
    pz = surface_offset + n_rear + R - rigid_bike.position['rear axle'][1]
    vx = 10
    vz = 0
    a_pitch = 0
    n_pitch = 0
    x0 = np.array((px, pz, vx, vz, a_pitch, n_pitch, s_front, n_front, s_rear, n_rear))
    u0 = np.zeros(rigid_bike.nu)

    xdot = rigid_bike.call_model(x0, u0)
    print(xdot)

    # call the ODE solver
    # time steps for the ODE solver
    t_steps_number = 100
    t = np.linspace(0, 2, t_steps_number)
    rtol = 1E-9

    def ode_fcn(xx, tt):
        return rigid_bike.call_model(xx, u0)

    x = odeint(ode_fcn, x0, t, h0=0.001, hmax=0.001, rtol=rtol)


    mech_energy = np.array([rigid_bike.get_mech_energy(xx) for xx in x])

    plt.figure(101)
    n_col = 2
    n_row = np.ceil((rigid_bike.nx+1)/n_col)

    for ii in range(rigid_bike.nx):
        plt.subplot(n_row, n_col, ii+1)
        plt.plot(t, x[:, ii])
        plt.ylabel(rigid_bike.state_names[ii])

    plt.subplot(n_row, n_col, rigid_bike.nx+2)
    plt.plot(t, mech_energy/1e3)
    plt.ylabel('ME [kJ]')

    plt.show()


    # prepare data for animation function
    fig = plt.figure(1)
    ax = plt.axes(xlim=(-5, 10), ylim=(0, 2))
    wheel_center_traj = ax.plot(x[:, rigid_bike.state_enum['px']], x[:, rigid_bike.state_enum['pz']], 'k:')[0]
    xs = np.linspace(x_min, x_max, 101)
    zs = z_func(xs)
    surface_line = ax.plot(xs, zs, 'k-')[0]
    plt.axis('equal')
    plt.title('rigid bike simulation')
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")


    # run animation
    frame_number = 100
    t_total = 10
    t_interval = t_total*1000/frame_number # [ms]

    def anim_func(i, x):
        rigid_bike.render(x[i, :])

    anim = animation.FuncAnimation(fig, anim_func, fargs=(x, ), frames=frame_number,
                                   interval=t_interval, blit=False, repeat=False)

    plt.show()
    anim.save('rigid_bike.mp4')
    # end plot =================================================


    return




if __name__ == '__main__':
    run_fwd()

