
# Sim Types

## Geometry Sim
casing the kinematics as a multi body problem, solved as an nonlinear system. 
The output should be kinematics data as well as polynomials that represent the kinematics
 for fast execution.


## Wheel Sim
A wheel rolling down an uneven ground, should contain fundamental physics such as:

* gravity (and coordinate frame change)
* contact force modelling (probably the hardest problem in the entire project)

It will give a taste on the challenges and potential solutions of the following problems:

* ODEs with events (wheel bouncing and landing), potential multi-phase, and stiff during one of the phase
 (the landing phase)
* the cost and merit of contact force modelling with continuous contact point on the ground and non-continuous ones



## DH Sim
A natural extension of wheel sim, but now with:
* the bike on top (the easy bit)
* rider actuatation (pedal and handlebar force with simple constraints -- not too hard)
* rider actuation with rider body and joints also modeld (more realtistic actuation constraints and geometry
 constraints, but more difficult) 

## Bunny Hop Sim
like DH sim, but on an even ground, the objective is to get the bike as high as possible.
Should be a good test on the optimal control framework, without the need of a thorough ODE scheme that handles
the multi-phase robustly and without the correct contact force modelling (simple spring will do)



# Key components
## Models

### Kinematics model
* bike suspension kinematics

### Rider model
* dynamic model (the body) with kinematic and dynamic constraints (body geometry, joint force/power limits)

### Dynamic model
* a single wheel (only for dev/test/debug)
* bike with actuation only
* bike with rider

## Algorithm/Simulation
### ODE solver 

### Optimal Control Parser -- discretise and parse an optimal control problem, and solve as an NLP

### Ride Simulation -- 3 post replay?  
