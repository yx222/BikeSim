# Examples showing how users should use the code. For deisgn guidance


# Create a bike object
sc_5010 = BikeKinematics.from_json('some_json_file')

# Sweep through travel, get bike properties
# Questions:
# 1) which properties are just rear travel related?
# 2) which properties require front travel?
# 3) if we sweep both together, what's the sensible way of coordinating front and rear travel?
properties = sc_5010.sweep_travel()
properties.anti_squat.plot()
properties.motion_ratio.plot()
properties.plot()

# Inside travel_sweep, it might look like below,
# where front and rear travel are just properties of the bike

# For a fixed design, the bike only has 2 D.O.F.
# And we can express them as front and rear travel.
for damper_travel, fork_travel in zip(damper_travel_list, fork_travel_list):
    states = sc_5010.solve_states(damper_travel=damper_travel,
                                  fork_travel=fork_travel)

# Change the design of a bike (just change, no solve)
# Increment (in body coordinate)
sc_5010.move_point(body='front_triangle', point='damper', np.array([0.005, -0.001]))

# Move relative position (in the body's coordinate)
sc_5010.move_point_to(body='front_triangle', point='damper', np.array([0.305, 0.21]))

# If we describe bodies by the lenght of ther frame members, then we can also do:
# Increment
sc_5010.change_length(body='front_triangle', member='top_tube', -0.1)

# absolute
sc_5010.change_length_to(body='front_triangle', member='top_tube', 0.45)

# Even for the member length representaiton, we still need the relative position of at least one point in the rigid body's coordinate

# If I want to adjust the rocker arm position on damper to achieve:
# - at 30% front and rear sag
# - BB height = 0.28m
# - allow rocker arm point to move within [-0.1, 0.1] in both directions (in front triangle body frame)
sc_5010.solve_design(free_point=('rocker', 'damper'),
                     free_range=np.array([[-0.1, 0.1], [-0.1, 0.1]]),
                     target_point=('front_triangle', 'bottom_bracket')
                     target_position=np.array([0.12, 0.28]),
                     front_travel=0.3, rear_travel=0.3)

# Or we could optimize things
# eg: maximize antisquat
# - move rocker damper arm within a certain range
sc_5010.maximize(objective='anti-squat',
                 front_travel=0.3, rear_travel=0.3,
                 free_point=('rocker', 'damper'),
                 free_range=np.array([[-0.1, 0.1], [-0.1, 0.1]]))
