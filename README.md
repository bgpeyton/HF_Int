Integrals are evaluated using the functions within ints.py. Most functions currently only work for s-type Gaussians. NOTE: This code is pretty messy, but it works!

The Hartree Fock procedure is laid out in hf.py. Functions exist for calculating a density matrix, doing a single HF iteration, and a complete HF function. The complete HF function `full_hf()` needs only a molecule (numpy array of coordinates) and a basis set (numpy array of Gaussian exponents for each atom). Convergence criteria are optional, otherwise it is set at 10E-12 AU. 

The optimization procedure is found in opt.py. For now, it is not implemented as a function. Istead, there is a starting bond distance r, and Newton-Raphson steps are taken based on an approximate derivative analysis until the geometry is converged (the absolute value of the first derivative is < 10E-6).

Code depends on math, numpy, and scipy. 
