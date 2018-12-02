import numpy as np

import psi4
from ab_products import AB_product_factory
from psi4.core import Matrix

mol = psi4.geometry("""
O 0.000000000000  -0.143225816552   0.000000000000
H 1.638036840407   1.136548822547  -0.000000000000
H -1.638036840407   1.136548822547  -0.000000000000
units bohr
symmetry c1
""")


class fake_jk(object):
    def __init__(self, Int):
        self.I = Int.copy()

    def J(self, Dlt):
        return np.einsum("unlt,lt->un", self.I, Dlt)

    def K(self, Dlt):
        return np.einsum("ulnt,lt->un", self.I, Dlt)


psi4.set_options({"SAVE_JK": True, 'scf_type': 'pk', 'd_convergence': 10})
e, wfn = psi4.energy("HF/sto-3g", return_wfn=True)
prod = AB_product_factory(wfn)

F_ao = prod.mFa.to_array()
C = prod.mCa.to_array()
Fmo = (C.T).dot(F_ao).dot(C)
F_ij = Fmo[prod.o, prod.o]
F_ab = Fmo[prod.v, prod.v]
FockA = np.einsum("ab,ij->iajb", F_ab, np.eye(prod.nocc))
FockA -= np.einsum("ij,ab->iajb", F_ij, np.eye(prod.nvir))
fake_guess = np.eye(prod.nrot)
Iuvlt = prod.mints.ao_eri(prod.mints.basisset(), prod.mints.basisset(), prod.mints.basisset(),
                          prod.mints.basisset()).to_array()
jk_bldr = fake_jk(Iuvlt)
A_jk = np.zeros((prod.nrot, prod.nrot))
A_jk2 = np.zeros((prod.nrot, prod.nrot))
for ia in range(prod.nrot):
    x = fake_guess[:, ia]
    x = x.reshape(prod.nocc, prod.nvir)
    T = np.einsum("lj,tb,jb->lt", C[:, prod.o], C[:, prod.v], x)
    Jmn = jk_bldr.J(T)
    Kmn = jk_bldr.K(T)
    Jia = (C[:, prod.o].T).dot(Jmn).dot(C[:, prod.v])
    Kia = (C[:, prod.o].T).dot(Kmn).dot(C[:, prod.v])
    Ax = 2 * Jia - Kia
    Ax += np.einsum("iajb,jb->ia", FockA, x)
    A_jk[:, ia] = Ax.reshape(prod.nrot)

w, X = np.linalg.eig(A_jk)
w_jk = w[w.argsort()]

for ia in range(prod.nrot):
    x = fake_guess[:, ia]
    Fx, Jx, Kx = prod.jk_ab_parts(x)
    #Ax = np.einsum("IJ,J->I", FockA.reshape(prod.nrot, prod.nrot), x)
    Ax = Fx.to_array().reshape(prod.nrot)
    J_mo = 2 * (C[:, prod.o].T).dot(Jx).dot(C[:, prod.v])
    K_mo = (C[:, prod.o].T).dot(Kx).dot(C[:, prod.v])
    Ax += J_mo.reshape(prod.nrot)
    Ax -= K_mo.reshape(prod.nrot)
    A_jk2[:, ia] = Ax

w, X = np.linalg.eigh(A_jk2)
w_jk2 = w[w.argsort()]

Viajb = np.einsum("uvlt,ui,va,lj,tb->iajb", Iuvlt, C[:, prod.o], C[:, prod.v], C[:, prod.o], C[:, prod.v])
Vabij = np.einsum("uvlt,ua,vb,li,tj->abij", Iuvlt, C[:, prod.v], C[:, prod.v], C[:, prod.o], C[:, prod.o])
A_mo = 2 * Viajb - np.einsum("abij->iajb", Vabij)
A_mo += FockA
A_mo = A_mo.reshape((prod.nrot, prod.nrot))
w, X = np.linalg.eig(A_mo)
w_mo = w[w.argsort()]

w_check = [
    #0.2872554996,
    #0.2872554996,
    #0.2872554996,
    #0.3444249963,
    #0.3444249963,
    #0.3444249963,
    0.3564617587,
    #0.3659889948,
    #0.3659889948,
    #0.3659889948,
    #0.3945137992,
    #0.3945137992,
    #0.3945137992,
    0.4160717386,
    0.5056282877,
    #0.5142899971,
    #0.5142899971,
    #0.5142899971,
    0.5551918860,
    #0.5630557635,
    #0.5630557635,
    #0.5630557635,
    0.6553184485,
    0.9101216891,
    #1.1087709658,
    #1.1087709658,
    #1.1087709658,
    #1.2000961331,
    #1.2000961331,
    #1.2000961331,
    1.3007851948,
    1.3257620652,
    #19.9585264123,
    #19.9585264123,
    #19.9585264123,
    20.0109794203,
    #20.0113420895,
    #20.0113420895,
    #20.0113420895,
    20.0505319444
]

print("{:^5}  {:^20}  {:^20}  {:^10}".format("#", "Direct MO A", "Project 12", "Match?"))
print("{:-^5}  {:-^20}  {:-^20}  {:-^10}".format("-", "-", '-', "-"))
for i, wi in enumerate(w_mo):
    if abs(wi - w_check[i]) < 1.0e-8:
        m = "YES"
    else:
        m = "X NO X"
    print("{:>5}  {:>20.12f}  {:>20.12f}  {:^10}".format(i, wi, w_check[i], m))

print("{:^5}  {:^20}  {:^20}  {:^10}".format("#", "MyJK A(AO)", "Project 12", "Match?"))
print("{:-^5}  {:-^20}  {:-^20}  {:-^10}".format("-", "-", '-', "-"))
for i, wi in enumerate(w_jk):
    if abs(wi - w_check[i]) < 1.0e-8:
        m = "YES"
    else:
        m = "X NO X"
    print("{:>5}  {:>20.12f}  {:>20.12f}  {:^10}".format(i, wi, w_check[i], m))

print("{:^5}  {:^20}  {:^20}  {:^10}".format("#", "Real_JK A(AO)", "Project 12", "Match?"))
print("{:-^5}  {:-^20}  {:-^20}  {:-^10}".format("-", "-", '-', "-"))
for i, wi in enumerate(w_jk2):
    if abs(wi - w_check[i]) < 1.0e-8:
        m = "YES"
    else:
        m = "X NO X"
    print("{:>5}  {:>20.12f}  {:>20.12f}  {:^10}".format(i, wi, w_check[i], m))
