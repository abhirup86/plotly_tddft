"""
   engines for TDSCF

   isort:skip_file
"""
import sys
from pathlib import Path

import numpy as np

try:
    import psi4
    from psi4 import core
    from psi4.driver.p4util import solvers
    from psi4.driver.p4util.exceptions import *
except ImportError:
    sys.path.append(str(Path.home() / "Workspace/git/psi4/build/stage/lib"))
    import psi4
    from psi4 import core
    from psi4.driver.p4util import solvers
    from psi4.driver.p4util.exceptions import *


class TDSCF_JK_Engine(object):
    def __init__(self, wfn, blocks=False, precondition='denom', tda=False, triplet=False):
        blocks = False
        if not wfn.same_a_b_orbs():
            raise ValidationError("TDSCF is RHF only")
        self.wfn = wfn
        self.jk = wfn.jk()
        self.C = wfn.Ca()
        self.Co = wfn.Ca_subset("SO", "OCC")
        self.Cv = wfn.Ca_subset("SO", "VIR")
        self.noccpi = wfn.nalphapi()
        self.nmopi = wfn.nmopi()
        self.nirrep = wfn.nirrep()
        self.nvirpi = core.Dimension([mo - occ for mo, occ in zip(self.noccpi.to_tuple(), self.nmopi.to_tuple())])
        self.nocc = wfn.nalphapi().sum()
        self.nmo = wfn.nmopi().sum()
        self.nvir = self.nmo - self.nocc
        self.nov = self.nocc * self.nvir
        self.Fab = core.Matrix.triplet(self.Cv, self.wfn.Fa(), self.Cv, True, False, False)
        self.Fij = core.Matrix.triplet(self.Co, self.wfn.Fb(), self.Co, True, False, False)
        self.e_ia = np.zeros((self.nocc, self.nvir))
        for i in range(self.nocc):
            for a in range(self.nvir):
                self.e_ia[i, a] = self.Fab.np[a, a] - self.Fij.np[i, i]
        self.blocks = blocks
        self.triplet = triplet

    def precond_denom(self, res, i, w):
        denom = np.zeros((self.nocc, self.nvir))
        for i in range(self.nocc):
            for a in range(self.nvir):
                denom[i, a] = 1.0 / (w - self.Fab.np[a, a] - self.Fij.np[i, i])
        return res / self.e_ia.ravel()

    def precond_none(self, res, i, w):
        return res

    def Fx_product(self, X):
        """Computes the Fock contribution to A

        A_{ia,jb}X_{jb} <-- X_{jb} F_ab - F_{ij} X_{jb}

        Parameters
        ----------
        Parameters
        ----------
        X: :py:class:`psi4.core.Matrix`
           A trial vector (occ x vir)
        """
        Fx_i = X.to_array() * self.e_ia
        return core.Matrix.from_array(Fx_i)

    def TDA_product(self, X):
        """Compute the product of a trial vector with the CIS Hamiltonian A

        Ax_{ia} = A_{ia,jb}X_{jb}
                = X_{jb} F_ab - F_{ij} X_{jb} + C^{occ}_pi C^{vir}_{qa} (2*J[P]_pq - K[P}_pq])

        Parameters
        ----------
        Parameters
        ----------
        X: :py:class:`numpy.ndarray` or :py:class:`psi4.core.Matrix`
           A trial vector (occ x vir)
        """
        triplet = False
        ret = self.wfn.Ax_JK_products([core.Matrix.from_array(X.reshape(self.nocc, self.nvir))], triplet)
        Fx = X * self.e_ia.ravel()

        return ret[0].to_array().ravel() + Fx.ravel()

    def RPA_product_pair(self, X):
        """Compute A+B and A-B products

        (A+B)_{ia,jb}X_{jb} =
                X_{jb} * [(F_{ab} - F_{ij}) + C^{occ}_{pi} C^{vir}_{qa}(4 * J[P]_pq - K[P]_{pq} - K[P]_qp)]

        (A-B)_{ia,jb}X_{jb} =
                X_{jb} * [(F_{ab} - F_{ij}) + C^{occ}_{pi} C^{vir}_{qa}(K[P]_pq - K[P]_qp)]

        Parameters
        ----------
        X : list of :py:class:`psi4.core.Matrix`
           A trial vector (noccpi x nvirpi)

        Returns
        -------
        l : list of tuples (P, M)

        P : :py:class:`psi4.core.Matrix` (noccpi x nvirpi)
          A+B x trial vector
        M : :py:class:`psi4.core.Matrix` (noccpi x nvirpi)
          A-B x trial vector
        """
        ret = self.wfn.ABx_JK_products(X, False)
        for i, xi in enumerate(X):
            Fxi = self.Fx_product(xi)
            ret[i][0].axpy(1.0, Fxi)
            ret[i][1].axpy(1.0, Fxi)
        return ret

    def RPA_product_single(self, X):
        "Compute (A-B)(A+B)X in one go"

        first = self.wfn.ABx_JK_products(X, False)
        next_x = []
        for i, xi in enumerate(X):
            Fxi = self.Fx_product(xi)
            P = first[i][0]
            P.axpy(1.0, Fxi)
            next_x.append(P)
        sec = self.wfn.ABx_JK_products(next_x, False)
        ret = []
        for i, Pxi in enumerate(next_x):
            Fxi = self.Fx_product(Pxi)
            M = sec[i][1]
            M.axpy(1.0, Fxi)
            ret.append(M)
        return ret

    def davidson_RPA_func(self, X):
        ret = self.RPA_product_single([core.Matrix.from_array(X.reshape(self.nocc, self.nvir))])[0]
        return ret.to_array().flatten()

    def symp_RPA_func(self, X):
        ret = self.RPA_product_pair([core.Matrix.from_array(X.reshape(self.nocc, self.nvir))])[0]
        return (ret[0].to_array().flatten(), ret[1].to_array().flatten())
