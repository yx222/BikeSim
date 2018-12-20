# BikeSim -- MTB geometry and dyanmics simulation

* Choice of Automatic-differentiation tools:
    * [Autograd](https://github.com/HIPS/autograd)
        * pro: straight forward syntax, functions can be reused
        * con: efficiency? ('only scalar valued functions')
    * [ad](https://pythonhosted.org/ad/)
    * [Casadi](https://web.casadi.org/docs/#document-symbolic)
        * pro: well integrated, lots of functionality for optimal control, ode, optimization, etc
        * con: not very flexible, syntax is counter-intuitive
    * [PyTorch](https://pytorch.org) 
 
 * Most likely candidate solution: [Autograd](https://github.com/HIPS/autograd) + [cyipopt](https://github.com/matthias-k/cyipopt)  