import math
import numpy as np # for norm
import scipy.special as sps # for binom

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


def S(n, a, A, m, b, B):
    """Returns equation 7"""
    return math.exp(-a*b*np.linalg.norm(A-B)**2/(a+b))*In(n[0],a,A[0],m[0],b,B[0])*In(n[1],a,A[1],m[1],b,B[1])*In(n[2],a,A[2],m[2],b,B[2]) 


def K(a, A, b, B):
    """Returns equation 12: specifically for S gaussians!"""
    val = 0
    n = np.array([0,0,0])
    m = np.array([0,0,0])
    mx = np.array([2,0,0])
    my = np.array([0,2,0])
    mz = np.array([0,0,2])
    val = -2*b*S(n, a, A, m, b, B) + 4*b**2*S(n, a, A, mx, b, B)
    print("Val = {}".format(val))
    print("Overlap1 = {} and Overlap2 = {}".format(S(n, a, A, m, b, B),S(n, a, A, mx, b, B)))
    val += -2*b*S(n, a, A, m, b, B) + 4*b**2*S(n, a, A, my, b, B)
    val += -2*b*S(n, a, A, m, b, B) + 4*b**2*S(n, a, A, mz, b, B)
    val = -0.5*val
    return val


def V(a, A, b, B):
    """Returns equation 14: specifically for S gaussians!"""
    val1 = 0
    val2 = 0
    val1 = -2*math.pi*math.exp(-a*b*np.linalg.norm(A-B)**2/(a+b))/(a+b)

    Px = Pn(a, A[0], b, B[0])
    Py = Pn(a, A[1], b, B[1])
    Pz = Pn(a, A[2], b, B[2])
    P = np.array([Px, Py, Pz])
#    print("P = {}".format(P))

    C = A # Hard-coding two atoms for now
#    if C.all() == P.all():
    if np.array_equal(C, P):
        val2 += 1
#        print ("Caught the first if")
    else:
        val2 += math.pi**0.5*math.erf((a+b)**0.5*np.linalg.norm(P-C))/(2*(a+b)**0.5*np.linalg.norm(P-C))
#        print("Caught the first else")

    C = B # Hard-coding two atoms for now
#    if C.all() == P.all():
    if np.array_equal(C, P):
        val2 += 1
#        print ("Caught the second if")
    else:
        val2 += math.pi**0.5*math.erf((a+b)**0.5*np.linalg.norm(P-C))/(2*(a+b)**0.5*np.linalg.norm(P-C)) 
#        print ("Caught the second else")

#    print("val1 = {}".format(val1))
#    print("val2 = {}".format(val2))
    return val1*val2


def eri(a1, A1, b1, B1, a2, A2, b2, B2):
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

    if np.linalg.norm(P1-P2)**2*(a1 + b1)*(a2+b2)/((a1+b1)+(a2+b2)) == 0:
        F = 1
    else:
        F = math.pi**0.5/2./(np.linalg.norm(P1-P2)**2*(a1 + b1)(a2+b2)/((a1+b1)+(a2+b2)))**0.5*math.erf(np.linalg.norm(P1-P2)**2*(a1 + b1)(a2+b2)/((a1+b1)+(a2+b2)))  

    val = 2*math.pi**2.5*K1*K2/((a1+b1)*(a2+b2)*((a1+b1)+(a2+b2))**0.5)*F
    return val


n = np.array([2,0,0])
m = np.array([0,0,0])
a1 = 5.447178
norm1 = 2.5411995
a2 = 0.824547
norm2 = 0.6166967
a3 = 0.183192000
norm3 = 0.1995676
A = np.array([0,0,0])
B = np.array([1.4,0,0])
nc1 = 2.5411995

#print(S(n, a, A, m, a, B))
#print(K(a2, A, a2, B)*norm2*norm2)
#print(V(a1, B, a1, B)*norm1*norm1)
#print(V(a1, A, a1, A))
print(eri(a1, A, a1, B, a1, A, a1, B)*norm1*norm1*norm1*norm1)
