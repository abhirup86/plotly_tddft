import sys
import time
from pathlib import Path
import numpy as np

try:
    import psi4
except ImportError:
    sys.path.append(str(Path.home() / "Workspace/git/psi4/build/stage/lib"))
    import psi4

CH2O_str = """
O
C  1 1.22
H  2 1.08 1 120
H  2 1.08 1 120 3 180
symmetry c1
"""


def get_HF_wfn(molstring, basis, **options):
    psi4.core.clean()
    psi4.set_memory("8 GB")
    psi4.set_num_threads(8)
    psi4.set_output_file("psilog.dat")

    mol = psi4.geometry(molstring)
    psi4.set_options(options)
    e, wfn = psi4.energy("HF/{}".format(basis), return_wfn=True, molecule=mol)
    return wfn


def do_direct_diag(wfn):
    "Build A,B in core, direct diag A ->RPA, direct diag AB//-B-A -> TDA"
    mints = psi4.core.MintsHelper(wfn.basisset())
    nocc = wfn.nalphapi().sum()
    nmo = wfn.nmopi().sum()
    nvir = nmo - nocc

    mCa_occ = wfn.Ca_subset("AO", "OCC")
    mCa_vir = wfn.Ca_subset("AO", "VIR")
    mFa = wfn.Fa()
    F_ij = psi4.core.Matrix.triplet(mCa_occ, mFa, mCa_occ, True, False, False).to_array()
    F_ab = psi4.core.Matrix.triplet(mCa_vir, mFa, mCa_vir, True, False, False).to_array()
    Iiajb = mints.mo_eri(mCa_occ, mCa_vir, mCa_occ, mCa_vir).to_array()
    Iabji = mints.mo_eri(mCa_vir, mCa_vir, mCa_occ, mCa_occ).to_array()
    A = np.einsum("ab,ij->iajb", F_ab, np.eye(nocc))
    A -= np.einsum("ij,ab->iajb", F_ij, np.eye(nvir))
    A += 2 * Iiajb - np.einsum("abji->iajb", Iabji)
    B = 2 * Iiajb - Iiajb.swapaxes(0, 2)
    A = A.reshape(nocc * nvir, nocc * nvir)
    B = B.reshape(nocc * nvir, nocc * nvir)

    val, TDA_vec = np.linalg.eig(A)
    idx = np.argsort(val)
    TDA_vec = TDA_vec[:, idx]
    TDA_val = val[idx]

    H1 = np.hstack((A, B))
    H2 = np.hstack((-B, -A))
    H = np.vstack((H1, H2))
    RPA_val, RPA_vec = np.linalg.eig(H)
    # select positive roots
    pos_idx = RPA_val[RPA_val > 1.0e-6]
    RPA_vec = RPA_vec[:, pos_idx]
    RPA_val = RPA_val[pos_idx]
    # sort
    sort_idx = np.argsort(RPA_val)
    RPA_vec = RPA_vec[:, sort_idx]
    RPA_val = RPA_val[sort_idx]
    return (TDA_val, TDA_vec), (RPA_val, RPA_vec)


def check_same(aval, a_label, b_val, b_label):
    print("{:^5}  {:^20}  {:^20}  {:^10}".format("#", a_label, b_label, "Match?"))
    print("{:-^5}  {:-^20}  {:-^20}  {:-^10}".format("-", "-", '-', "-", "-", "-"))
    all_same = True
    for i in range(len(aval)):
        av = aval[i]
        bv = bval[i]
        diff = abs(av - bv)
        if abs(diff) < tol:
            m = "YES"
        else:
            m = "NO"
            all_same = False
        print("{:>5}  {:>20.12f}  {:>20.12f}  {:^10}".format(i, av, bv, m))
    return all_same


# dav_vals = np.sqrt(dav_vals)
# for i in range(len(dav_stats['iters'])):
#     evals = dav_stats['iters'][i]['e']
#     dav_stats['iters'][i]['e'] = [np.sqrt(ei) for ei in evals]

# compare("Davidson RPA", dav_vals, wRPA_check[:nroot])
# compare("Symp RPA", symp_vals, wRPA_check[:nroot])

# root_info = "  {:.7f}  "
# title = "Iteration (nvec)  | " + " | ".join(["{}".format(n+1).center(12) for n in range(nroot)])

# iter_lbl = "{:<3}({:<3})" + (" " * (len(title.split("|")[0].strip())-8)) +" | "
# print("Davidson Diagonalize".center(len(title)))
# print(title)
# comp_per_iter_dav = []
# for i, st in enumerate(dav_stats['iters']):
#     d_ev = 0
#     for n in range(nroot):
#         d_ev += abs(st['e'][n] - dav_vals[n]) * psi4.constants.hartree2ev
#     d_ev = d_ev / nroot
#     comp_per_iter_dav.append(d_ev)
#     line = iter_lbl.format(i+1, st['nvecs']) + " | ".join([root_info.format(abs(st['e'][n] - dav_vals[n]) *
#         psi4.constants.hartree2ev) for n in range(nroot)])
#     print(line)

# print("")

# print("Symplectic Diagonalize".center(len(title)))
# print(title)
# conv_per_iter_symp = []
# for i, st in enumerate(symp_stats['iters']):
#     d_ev = 0
#     for n in range(nroot):
#         d_ev += abs(st['e'][n] - symp_vals[n]) * psi4.constants.hartree2ev
#     d_ev = d_ev / nroot
#     conv_per_iter_symp.append(d_ev)
#     line = iter_lbl.format(i+1, st['nvecs']) + " | ".join([root_info.format(abs(st['e'][n] - symp_vals[n]) *
#         psi4.constants.hartree2ev) for n in
#         range(nroot)])
#     print(line)

# # plot conv
# dav_dat = np.array(comp_per_iter_dav)
# symp_dat = np.zeros_like(dav_dat)
# symp_dat[:len(conv_per_iter_symp)] = np.array(conv_per_iter_symp)

# plt.plot(symp_dat, 'bo', dav_dat, 'gx')
# plt.yscale('log')
