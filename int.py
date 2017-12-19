import math
import numpy as np # for norm
import scipy.special as sps # for binom
import scipy.linalg as spl # for inverse sqrt of matrix
np.set_printoptions(precision=6)

def Pn(a, An, b, Bn):
    """Returns equation 9, takes individual coords of A and B"""
    if (a + b == 0):
        return 0 
    else:
        return (a*An + b*Bn)/(a+b)


def In(nn, a, An, mn, b, Bn):
    """Returns equation 8"""
    val = 0
    for i in range(0, nn+1):
        for j in range(0, mn+1):
            val += (Pn(a, An, b, Bn) - An) ** (nn - i) * sps.binom(nn,i) * (Pn(a, An, b, Bn) - Bn) ** (mn - j) * sps.binom(mn,j) * 0.5 * (1 + (-1) ** (i+j)) * (a + b) ** ((-i - j - 1)/2.) * math.gamma((1 + i + j)/2.) 
    return val


def gen_S(n, a, A, m, b, B):
    """Returns equation 7"""
    return math.exp(-a*b*np.linalg.norm(A-B)**2/(a+b))*In(n[0],a,A[0],m[0],b,B[0])*In(n[1],a,A[1],m[1],b,B[1])*In(n[2],a,A[2],m[2],b,B[2]) 


def normalize(n, a, A):
    """Returns a normalization constant from vector A and basis a"""
    val = gen_S(n, a, A, n, a, A)
    return (1/gen_S(n, a, A, n, a, A)**0.5)


def gen_K(a, A, b, B):
    """Returns equation 12: specifically for S gaussians!"""
    val = 0
    n = np.array([0,0,0])
    m = np.array([0,0,0])
    mx = np.array([2,0,0])
    my = np.array([0,2,0])
    mz = np.array([0,0,2])
    val = -2*b*gen_S(n, a, A, m, b, B) + 4*b**2*gen_S(n, a, A, mx, b, B)
    val += -2*b*gen_S(n, a, A, m, b, B) + 4*b**2*gen_S(n, a, A, my, b, B)
    val += -2*b*gen_S(n, a, A, m, b, B) + 4*b**2*gen_S(n, a, A, mz, b, B)
    val = -0.5*val
    return val


def gen_V(a, A, b, B, molecule):
    """Returns equation 14: specifically for S gaussians!"""
    val1 = 0
    val2 = 0
    val1 = -2*math.pi*math.exp(-a*b*np.linalg.norm(A-B)**2/(a+b))/(a+b)

    Px = Pn(a, A[0], b, B[0])
    Py = Pn(a, A[1], b, B[1])
    Pz = Pn(a, A[2], b, B[2])
    P = np.array([Px, Py, Pz])

    C = molecule[0] # Hard-coding two atoms for now
    if np.array_equal(C, P):
        val2 += 1
    else:
        val2 += math.pi**0.5*math.erf((a+b)**0.5*np.linalg.norm(P-C))/(2*(a+b)**0.5*np.linalg.norm(P-C))

    C = molecule[1] # Hard-coding two atoms for now
    if np.array_equal(C, P):
        val2 += 1
    else:
        val2 += math.pi**0.5*math.erf((a+b)**0.5*np.linalg.norm(P-C))/(2*(a+b)**0.5*np.linalg.norm(P-C)) 

    return val1*val2


def gen_eri(a1, A1, b1, B1, a2, A2, b2, B2):
    """Returns equation 15""" 

    Px1 = Pn(a1, A1[0], b1, B1[0])
    Py1 = Pn(a1, A1[1], b1, B1[1])
    Pz1 = Pn(a1, A1[2], b1, B1[2])
    P1 = np.array([Px1, Py1, Pz1])

    Px2 = Pn(a2, A2[0], b2, B2[0])
    Py2 = Pn(a2, A2[1], b2, B2[1])
    Pz2 = Pn(a2, A2[2], b2, B2[2])
    P2 = np.array([Px2, Py2, Pz2])

    K1 = math.exp(-a1*b1*np.linalg.norm(A1-B1)**2/(a1+b1))
    K2 = math.exp(-a2*b2*np.linalg.norm(A2-B2)**2/(a2+b2))

    T = np.linalg.norm(P1-P2)**2*(a1 + b1)*(a2+b2)/((a1+b1)+(a2+b2))

    if T == 0:
        F = 1
    else:
        F = math.pi**0.5/2./(T**0.5)*math.erf(T**0.5)  

    val = 2*math.pi**2.5*K1*K2/((a1+b1)*(a2+b2)*((a1+b1)+(a2+b2))**0.5)*F
    return val


def S_mat(molecule, basis):
    """Returns overlap matrix S"""
    S = np.zeros([molecule.shape[0]*basis.shape[1],molecule.shape[0]*basis.shape[1]])
    n = np.array([0,0,0])
    m = np.array([0,0,0])
    natom = molecule.shape[0]
    nbasis = basis.shape[1]

    for i in range(0,natom): # Atom 1 loop
        for j in range(0,natom): # Atom 2 loop
            for p in range(0,nbasis): # Basis function 1 loop
                for q in range(0,nbasis): # Basis function 2 loop
                    S[p+3*i,q+3*j] = gen_S(n, basis[i,p], molecule[i], m, basis[j,q], molecule[j]) * normalize(n, basis[i,p], molecule[i]) * normalize(m, basis[j,q], molecule[j])

    return S


def K_mat(molecule, basis):
    """Returns kinetic energy matrix K"""
    K = np.zeros([molecule.shape[0]*basis.shape[1],molecule.shape[0]*basis.shape[1]])
    n = np.array([0,0,0])
    m = np.array([0,0,0])
    natom = molecule.shape[0]
    nbasis = basis.shape[1]

    for i in range(0,natom): # Atom 1 loop
        for j in range(0,natom): # Atom 2 loop
            for p in range(0,nbasis): # Basis function 1 loop
                for q in range(0,nbasis): # Basis function 2 loop
                    K[p+3*i,q+3*j] = gen_K(basis[i,p], molecule[i], basis[j,q], molecule[j]) * normalize(n, basis[i,p], molecule[i]) * normalize(m, basis[j,q], molecule[j])

    return K


def V_mat(molecule, basis):
    """Returns kinetic energy matrix K"""
    V = np.zeros([molecule.shape[0]*basis.shape[1],molecule.shape[0]*basis.shape[1]])
    n = np.array([0,0,0])
    m = np.array([0,0,0])
    natom = molecule.shape[0]
    nbasis = basis.shape[1]

    for i in range(0,natom): # Atom 1 loop
        for j in range(0,natom): # Atom 2 loop
            for p in range(0,nbasis): # Basis function 1 loop
                for q in range(0,nbasis): # Basis function 2 loop
                    V[p+3*i,q+3*j] = gen_V(basis[i,p], molecule[i], basis[j,q], molecule[j], molecule) * normalize(n, basis[i,p], molecule[i]) * normalize(m, basis[j,q], molecule[j])

    return V


def gen_H_core(molecule, basis):
    H_core = K_mat(molecule, basis) + V_mat(molecule, basis)
    return H_core


def eri_mat(molecule, basis, P):
#    mol = np.array([molecule[0],molecule[0],molecule[0],molecule[1],molecule[1],molecule[1]])
#    print("molecule = {}".format(molecule))
#    print("mol = {}".format(mol))
#    print("mol[1] = {}".format(mol[1]))
#    print("mol[1,1] = {}".format(mol[1,1]))
#    bas = basis.flatten()
#    print("bas = {}".format(bas))
#    print("bas[1] = {}".format(bas[1]))


    na = np.array([0,0,0])
    ma = np.array([0,0,0])

#    J = np.zeros([molecule.shape[0]*basis.shape[1],molecule.shape[0]*basis.shape[1]])
#    K = np.zeros([molecule.shape[0]*basis.shape[1],molecule.shape[0]*basis.shape[1]])
#    G = np.zeros([molecule.shape[0]*basis.shape[1],molecule.shape[0]*basis.shape[1]])
#
#    for i in range(0,6):
#        for j in range(0,6):
#            for m in range(0,6):
#                for n in range(0,6):
#                    J[i,j] += P[m,n]*gen_eri(bas[i],mol[i],bas[j],mol[j],bas[m],mol[m],bas[n],mol[n]) * normalize(na, bas[i], mol[i]) * normalize(na, bas[j], mol[j]) * normalize(na, bas[m], mol[m]) * normalize(na, bas[n], mol[n]) 
#                    K[i,j] += P[m,n]*gen_eri(bas[i],mol[i],bas[n],mol[n],bas[m],mol[m],bas[j],mol[j]) * normalize(na, bas[i], mol[i]) * normalize(na, bas[j], mol[j]) * normalize(na, bas[m], mol[m]) * normalize(na, bas[n], mol[n]) 
#                    G[i,j] = 2*J[i,j] - K[i,j]
#            print("\nJ:\n{}".format(J))
#            print("\nK:\n{}".format(K))


    natom = molecule.shape[0] # this is 2 for us
    nbasis = basis.shape[1] # this is 3 for us
    G = np.zeros([molecule.shape[0]*basis.shape[1],molecule.shape[0]*basis.shape[1]])
    J = np.zeros([molecule.shape[0]*basis.shape[1],molecule.shape[0]*basis.shape[1]])
    K = np.zeros([molecule.shape[0]*basis.shape[1],molecule.shape[0]*basis.shape[1]])
#    print("\nJ at beginning of function:\n{}".format(J)) 
    for i in range(0,natom): 
        for j in range(0,natom):
            for ii in range(0,nbasis):
                for jj in range(0,nbasis):
                    for m in range(0,natom):
                        for n in range(0,natom):
                            for mm in range(0,nbasis):
                                for nn in range(0,nbasis):
                                    G[ii+3*i,jj+3*j] += P[mm+3*m,nn+3*n]*(2*gen_eri(basis[i,ii],molecule[i],basis[j,jj],molecule[j],basis[m,mm],molecule[m],basis[n,nn],molecule[n]) - gen_eri(basis[i,ii],molecule[i],basis[n,nn],molecule[n],basis[m,mm],molecule[m],basis[j,jj],molecule[j])) * normalize(na, basis[i,ii], molecule[i]) * normalize(na, basis[j,jj],molecule[j]) * normalize(na, basis[m,mm],molecule[m]) * normalize(na, basis[n,nn],molecule[n])


                                    J[ii+3*i,jj+3*j] += P[mm+3*m,nn+3*n]*gen_eri(basis[i,ii],molecule[i],basis[j,jj],molecule[j],basis[m,mm],molecule[m],basis[n,nn],molecule[n]) * normalize(na, basis[i,ii], molecule[i]) * normalize(na, basis[j,jj],molecule[j]) * normalize(na, basis[m,mm],molecule[m]) * normalize(na, basis[n,nn],molecule[n])

                                    K[ii+3*i,jj+3*j] += P[mm+3*m,nn+3*n]*gen_eri(basis[i,ii],molecule[i],basis[m,mm],molecule[m],basis[n,nn],molecule[n],basis[j,jj],molecule[j]) * normalize(na, basis[i,ii], molecule[i]) * normalize(na, basis[j,jj],molecule[j]) * normalize(na, basis[m,mm],molecule[m]) * normalize(na, basis[n,nn],molecule[n])
   
#            print("\nJ:\n{}".format(J)) 
#            print("\nK:\n{}".format(K)) 
#    print("\nJ at end of function:\n{}".format(J)) 
    return G


molecule = np.array([[0,0,0],[1.4,0,0]])
basis = np.array([[5.447178, 0.824547, 0.183192000],[5.447178, 0.824547, 0.183192000]])

print("\nS matrix:\n{}".format(S_mat(molecule, basis)))
print("\nK matrix:\n{}".format(K_mat(molecule, basis)))
print("\nV matrix:\n{}".format(V_mat(molecule, basis)))
print("\nCore Hamiltonian:\n{}".format(gen_H_core(molecule, basis)))
H = gen_H_core(molecule, basis)
S = S_mat(molecule, basis)
orth = spl.sqrtm(spl.inv(S))
print("\nOrthoganalization matrix:\n{}".format(orth))
#F = np.multiply(np.multiply(orth.T,H),orth)
#F = np.dot(np.dot(orth.T,H),orth)
#print("\nInitial Fock matrix in AO basis:\n{}".format(F))
#eps, C0 = np.linalg.eig(F)
#print("\nEigenvalues:\n{}\nEigenvectors:\n{}".format(eps,C0))
#idx = eps.argsort()[::-1]   
#eps = eps[idx]
#C0 = C0[:,idx]
#C = np.dot(orth,C0)
#print("\nC:\n{}".format(C))
#Cocc = C[:, :1]
#print("\nCocc:\n{}".format(Cocc))
#D = np.dot(Cocc, Cocc.T) 
#print("\nD:\n{}".format(D))
#E = np.sum(np.dot(D, (H + F)))
#print("\nE = {}".format(E))



# Let's try making the P matrix Valeev's way
eps, c = np.linalg.eigh(np.dot(np.dot(orth.T,H),orth))
print("\nEigenvectors:\n{}".format(c))
c_norm = np.dot(orth,c)
print("\nNormalized Eigenvectors:\n{}".format(c_norm))
P = np.zeros([molecule.shape[0]*basis.shape[1],molecule.shape[0]*basis.shape[1]])
for i in range (0,molecule.shape[0]*basis.shape[1]):
    for j in range (0,molecule.shape[0]*basis.shape[1]):
        for m in range (0,1):
            P[i,j] += c_norm[i,m]*c_norm[j,m]
print("\nP:\n{}".format(P))

G = eri_mat(molecule,basis,P)
print("\nG matrix:\n{}".format(G))

#F = np.zeros([molecule.shape[0]*basis.shape[1],molecule.shape[0]*basis.shape[1]])
#for i in range (0,molecule.shape[0]*basis.shape[1]):
#    for j in range (0,molecule.shape[0]*basis.shape[1]):
#        G = 0
#        for m in range (0,molecule.shape[0]*basis.shape[1]):
#            for n in range (0,molecule.shape[0]*basis.shape[1]):
#                print("i={},j={},m={},n={}".format(i,j,m,n))
#                G += P[m,n]*(2*gen_eri(basis[i,i],molecule[i],basis[m,m],molecule[m],basis[j,j],molecule[j],basis[n,n],molecule[n])-gen_eri(basis[i,i],molecule[i],basis[m,m],molecule[m],basis[n,n],molecule[n],basis[j,j],molecule[j])) 
#        F[i,j] = H[i,j] + G


#n = np.array([0,0,0])
#m = np.array([0,0,0])
#a1 = 5.447178
#norm1 = 2.5411995
#a2 = 0.824547
#norm2 = 0.6166967
#a3 = 0.183192000
#norm3 = 0.1995676
#A = np.array([0,0,0])
#B = np.array([1.4,0,0])
#nc1 = 2.5411995

#print(gen_S(n, a1, A, m, a1, A)*norm1*norm1)
#print(K(a2, A, a2, B)*norm2*norm2)
#print(V(a1, B, a1, B)*norm1*norm1)
#print(V(a1, A, a1, A))
#print("Test eri value:{}".format(gen_eri(a1, A, a1, A, a1, A, a1, A)))
#print(gen_eri(a2, A, a1, A, a3, A, a2, B)*normalize(n,a2,A)*normalize(n,a1,A)*normalize(n,a3,A)*normalize(n,a2,B))
#print(normalize(n, a1, A))
