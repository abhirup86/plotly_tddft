import numpy as np

import psi4
from ab_products import AB_product_factory
from psi4.core import Matrix

# mol = psi4.geometry("""
# O     0.0000000000    0.0000000000    2.3054658725
# H     1.7822044879    0.0000000000   -1.0289558751
# H    -1.7822044879    0.0000000000   -1.0289558751
# C     0.0000000000    0.0000000000    0.0000000000
# symmetry c1
# """)

mol = psi4.geometry("""
O 0.000000000000  -0.143225816552   0.000000000000
H 1.638036840407   1.136548822547  -0.000000000000
H -1.638036840407   1.136548822547  -0.000000000000
units bohr
symmetry c1
""")

psi4.set_options({"SAVE_JK": True, 'scf_type': 'pk', 'd_convergence': 10})
e, wfn = psi4.energy("HF/sto-3g", return_wfn=True)
prod = AB_product_factory(wfn)

F_ao = prod.mFa.to_array()
C = prod.mCa.to_array()
Fmo = (C.T).dot(F_ao).dot(C)
F_ij = Fmo[prod.o, prod.o]
F_ab = Fmo[prod.v, prod.v]
Fa = np.einsum("ab,ij->iajb", F_ab, np.eye(prod.nocc))
Fa -= np.einsum("ij,ab->iajb", F_ij, np.eye(prod.nvir))
Fa = Fa.reshape((prod.nrot, prod.nrot))

fake_guess = np.eye(prod.nrot)
# P = Fa.copy()
# M = Fa.copy()
P = np.zeros((prod.nrot, prod.nrot))
P2 = np.zeros((prod.nrot, prod.nrot))
M = np.zeros((prod.nrot, prod.nrot))
M2 = np.zeros((prod.nrot, prod.nrot))
for jb in range(prod.nrot):
    Fx, Jx, Kx = prod.jk_ab_parts(fake_guess[:, jb])
    Jx = Jx.to_array()
    Kx = Kx.to_array()
    Px = Fx.to_array().reshape(prod.nrot)
    Mx = Fx.to_array().reshape(prod.nrot)
    J_mo = (C[:, prod.o].T).dot(Jx).dot(C[:, prod.v])
    K_mo = (C[:, prod.o].T).dot(Kx).dot(C[:, prod.v])
    Kt_mo = (C[:, prod.o].T).dot(Kx.transpose()).dot(C[:, prod.v])
    Px += (4.0 * J_mo - K_mo - Kt_mo).reshape(prod.nrot)
    Mx += (Kt_mo - K_mo).reshape(prod.nrot)
    P[:, jb] = Px
    M[:, jb] = Mx
    Px_2, Mx_2 = prod.jk_RHF(fake_guess[:, jb])
    P2[:, jb] = Px_2
    M2[:, jb] = Mx_2

NH = np.einsum("ij,jk->ik", M, P)
NH2 = np.einsum("ij,jk->ik", M2, P2)
w, X = np.linalg.eig(NH)
w_NH = np.sqrt(w[w.argsort()])
w, X = np.linalg.eig(NH2)
w_NH2 = np.sqrt(w[w.argsort()])

Mhalf2 = Matrix.from_array(M2)
Mhalf = Matrix.from_array(M)
Mhalf.power(0.5, 1.0e-16)
Mhalf2.power(0.5, 1.0e-16)
Mhalf = Mhalf.to_array()
Mhalf2 = Mhalf2.to_array()
H = np.einsum("ij,jk,km->im", Mhalf, P, Mhalf)
H2 = np.einsum("ij,jk,km->im", Mhalf2, P2, Mhalf2)
w, X = np.linalg.eigh(H)
w_H = np.sqrt(w[w.argsort()])
w, X = np.linalg.eigh(H2)
w_H2 = np.sqrt(w[w.argsort()])

wm, X = np.linalg.eig(M)
wp, X = np.linalg.eig(P)

w_check = [
    0.3547782530, 0.4153174946, 0.5001011401, 0.5513718846, 0.6502707118, 0.8734253708, 1.2832053178, 1.3237421886,
    20.0109471551, 20.0504919449
]

print("{:^5}  {:^20}  {:^20}  {:^10}".format("#", "(A-B)(A+B)", "Project 12", "Match?"))
print("{:-^5}  {:-^20}  {:-^20}  {:-^10}".format("-", "-", '-', "-"))
for i, wi in enumerate(w_NH2):
    if abs(wi - w_check[i]) < 1.0e-8:
        m = "YES"
    else:
        m = "X NO X"
    print("{:>5}  {:>20.12f}  {:>20.12f}  {:^10}".format(i, wi, w_check[i], m))

print("{:^5}  {:^20}  {:^20}  {:^10}".format("#", "(A-B)^1/2(A+B)(A-B)^1/2", "Project 12", "Match?"))
print("{:-^5}  {:-^20}  {:-^20}  {:-^10}".format("-", "-", '-', "-"))
for i, wi in enumerate(w_H2):
    if abs(wi - w_check[i]) < 1.0e-8:
        m = "YES"
    else:
        m = "X NO X"
    print("{:>5}  {:>20.12f}  {:>20.12f}  {:^10}".format(i, wi, w_check[i], m))

print("{:^5}  {:^20}  {:^20}  {:^10}".format("#", "eigval (A+B)", "eigval (A-B)", "Match?"))
print("{:-^5}  {:-^20}  {:-^20}  {:-^10}".format("-", "-", '-', "-"))
for i, wi in enumerate(wm):
    m = "N/A"
    print("{:>5}  {:>20.12f}  {:>20.12f}  {:^10}".format(i, wi, wp[i], m))
