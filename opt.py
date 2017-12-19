from hf import *
# import chain includes ints.py, math, numpy as np, scipy.special as sps, and scipy.linalg as spl


basis = np.array([[5.447178, 0.824547, 0.183192000],[5.447178, 0.824547, 0.183192000]])


r = 1.0
d = 0.001

molecule = np.array([[0,0,0],[r,0,0]])
molecule1 = np.array([[0,0,0],[r+d,0,0]])
molecule2 = np.array([[0,0,0],[r-d,0,0]])

E = full_hf(molecule, basis)
E1 = full_hf(molecule1, basis)
E2 = full_hf(molecule2, basis)
dE = (E1 - E2) / 2. / d
ddE = (E1 + E2 - 2*E) / d / d 
print("For r = {}, E = {}, dE = {}, ddE = {}\n".format(r,E,dE,ddE))

while math.fabs(dE) > 10E-6:
    r = r - dE / ddE
    molecule = np.array([[0,0,0],[r,0,0]])
    molecule1 = np.array([[0,0,0],[r+d,0,0]])
    molecule2 = np.array([[0,0,0],[r-d,0,0]])
    E = full_hf(molecule, basis)
    E1 = full_hf(molecule1, basis)
    E2 = full_hf(molecule2, basis)
    dE = (E1 - E2) / 2. / d
    ddE = (E1 + E2 - 2*E) / d / d 
    print("For r = {}, E = {}, dE = {}, ddE = {}\n".format(r,E,dE,ddE))

