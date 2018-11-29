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

F_ako = prod.mFa.to_array()
C = prod.mCa.to_array()
Fmo = (C.T).dot(F_ao).dot(C)
F_ij = Fmo[prod.o, prod.o]
F_ab = Fmo[prod.v, prod.v]
print("F_ij: ", " x ".join(str(x) for x in F_ij.shape))
print("")
print("F_ab: ", " x ".join(str(x) for x in F_ab.shape))
Iiajb = prod.mints.mo_eri(prod.mCa_occ, prod.mCa_virt, prod.mCa_occ,
                          prod.mCa_virt).to_array()
Iabji = prod.mints.mo_eri(prod.mCa_virt, prod.mCa_virt, prod.mCa_occ,
                          prod.mCa_occ).to_array()
A = np.einsum("ab,ij->iajb", F_ab, np.eye(prod.nocc))
A -= np.einsum("ij,ab->iajb", F_ij, np.eye(prod.nvir))
A += 2 * Iiajb - np.einsum("abji->iajb", Iabji)
B = 2 * Iiajb - Iiajb.swapaxes(0, 2)
A = A.reshape((prod.nrot, prod.nrot))
B = B.reshape((prod.nrot, prod.nrot))
H1 = np.hstack((A, B))
H2 = np.hstack((-1.0 * B, -1.0 * A))
H = np.vstack((H1, H2))
w, X = np.linalg.eig(H)
w = w[np.abs(w) == w]
w_full = w[w.argsort()]

AmB = A - B
ApB = A + B
Red = np.dot(AmB, ApB)
w2, X = np.linalg.eig(Red)
w_red = np.sqrt(w2[w2.argsort()])

wm, X = np.linalg.eig(A - B)

w_check = [
    #0.2851637170,
    #0.2851637170,
    #0.2851637170,
    #0.2997434467,
    #0.2997434467,
    #0.2997434467,
    #0.3526266606,
    #0.3526266606,
    #0.3526266606,
    0.3547782530,
    #0.3651313107,
    #0.3651313107,
    #0.3651313107,
    0.4153174946,
    0.5001011401,
    #0.5106610509,
    #0.5106610509,
    #0.5106610509,
    #0.5460719086,
    #0.5460719086,
    #0.5460719086,
    0.5513718846,
    0.6502707118,
    0.8734253708,
    #1.1038187957,
    #1.1038187957,
    #1.1038187957,
    #1.1957870714,
    #1.1957870714,
    #1.1957870714,
    1.2832053178,
    1.3237421886,
    #19.9585040647,
    #19.9585040647,
    #19.9585040647,
    20.0109471551,
    #20.0113074586,
    #20.0113074586,
    #20.0113074586,
    20.0504919449
]
print("{:^5}  {:^20}  {:^20}  {:^10}".format("#", "MeFULL", "Project 12",
                                             "Match?"))
print("{:-^5}  {:-^20}  {:-^20}  {:-^10}".format("-", "-", '-', "-"))
for i, wi in enumerate(w_full):
    if abs(wi - w_check[i]) < 1.0e-8:
        m = "YES"
    else:
        m = "X NO X"
    print("{:>5}  {:>20.12f}  {:>20.12f}  {:^10}".format(i, wi, w_check[i], m))

print("{:^5}  {:^20}  {:^20}  {:^10}".format("#", "MeRed", "Project 12",
                                             "Match?"))
print("{:-^5}  {:-^20}  {:-^20}  {:-^10}".format("-", "-", '-', "-"))
for i, wi in enumerate(w_red):
    if abs(wi - w_check[i]) < 1.0e-8:
        m = "YES"
    else:
        m = "X NO X"
    print("{:>5}  {:>20.12f}  {:>20.12f}  {:^10}".format(i, wi, w_check[i], m))

print("{:^5}  {:^20}  {:^20}  {:^10}".format("#", "eigval (A-B)", "sqrt",
                                             "Match?"))
print("{:-^5}  {:-^20}  {:-^20}  {:-^10}".format("-", "-", '-', "-"))
for i, wi in enumerate(wm):
    m = "N/A"
    print("{:>5}  {:>20.12f}  {:>20.12f}  {:^10}".format(
        i, wi, np.sqrt(wi), m))
