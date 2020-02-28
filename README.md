# BikeSim -- MTB geometry and dyanmics simulation

* Choice of Automatic-differentiation tools:
    * [Autograd](https://github.com/HIPS/autograd)
        * pro: straight forward syntax, functions can be reused (same syntax as numpy)
        * con: efficiency? ('only scalar valued functions')
    * [ad](https://pythonhosted.org/ad/)
    * [Casadi](https://web.casadi.org/docs/#document-symbolic)
        * pro: well integrated, can be code-gen'd! lots of functionality for optimal control, ode, optimization, etc
        * con: not very flexible, syntax is a little bit confusing
    * [PyTorch](https://pytorch.org) 
 
 * Most likely candidate solution: [Autograd](https://github.com/HIPS/autograd) + [cyipopt](https://github.com/matthias-k/cyipopt)  
 
* Running the docker container:
   * clone BikeSim into ~/workspace/BikeSim
   * cd into docker folder
   * run make
   * run run_docker.sh
   * visit the website at localhost:8080
