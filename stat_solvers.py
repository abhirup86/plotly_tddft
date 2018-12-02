import math
import time

import numpy as np

ConvergenceError = Exception


def davidson_solver(ax_function,
                    preconditioner,
                    guess,
                    e_conv=1.0E-8,
                    r_conv=None,
                    no_eigs=1,
                    max_vecs_per_root=20,
                    maxiter=100):
    """
    Solves for the lowest few eigenvalues and eigenvectors of the given real symmetric matrix

    Parameters
    -----------
    ax_function : function
        Takes in a guess vector and returns the Matrix-vector product.
    preconditioner : function
        Takes in a list of :py:class:`~psi4.core.Matrix` objects and a mask of active indices. Returns the preconditioned value.
    guess : list of :py:class:`~psi4.core.Matrix`
        Starting vectors, required
    e_conv : float
        Convergence tolerance for eigenvalues
    r_conv : float
        Convergence tolerance for residual vectors
    no_eigs : int
        Number of eigenvalues needed
    maxiter : int
        The maximum number of iterations this function will take.

    Returns
    -----------

    Notes
    -----------

    Examples
    -----------

    """
    if r_conv == None:
        r_conv = e_conv / 100

    # using the shape of the guess vectors to set the dimension of the matrix
    N = guess.shape[0]

    #sanity check, guess subspace must be at least equal to number of eigenvalues
    nli = guess.shape[1]
    if nli < no_eigs:
        raise ValueError("Not enough guess vectors provided!")

    stats = {}
    stats['Nroot'] = no_eigs
    stats['Nov'] = N
    stats['iters'] = []
    stats['solver_name'] = 'Davidson'

    nl = nli
    converged = False
    count = 0
    A_w_old = np.ones(no_eigs)
    max_ss_size = no_eigs * max_vecs_per_root
    B = guess

    ### begin loop
    conv_roots = [False] * no_eigs
    while count < maxiter:
        istat = {}
        istat['time'] = time.time()
        # active_mask = [True for x in range(nl)] # never used later
        # Apply QR decomposition on B to orthogonalize the new vectors wrto all other subspace vectors
        ## orthogonalize preconditioned residuals against all other vectors in the search subspace
        B, r = np.linalg.qr(B)
        nl = B.shape[1]
        istat['nvecs'] = nl
        istat['nprod'] = 0

        print("Davidson: Iter={:<3} nl = {:<4}".format(count, nl))
        # compute sigma vectors corresponding to the new vectors sigma_i = A B_i
        sigma = np.zeros_like(B)
        for i in range(nl):
            sigma[:, i] = ax_function(B[:, i])
            istat['nprod'] += 1

        # compute subspace matrix A_b = Btranspose sigma
        A_b = np.dot(B.T, sigma)

        # solve eigenvalue problem for subspace matrix; choose n lowest eigenvalue eigpairs
        A_w, A_v = np.linalg.eig(A_b)

        # sorting eigenvalues and corresponding eigenvectors
        idx = A_w.argsort()[:no_eigs]
        A_v_imag = A_v[:, idx].imag
        A_w_imag = A_w[idx].imag
        A_v = A_v[:, idx].real
        A_w = A_w[idx].real
        istat['e'] = A_w

        # Print warning if complex parts are too large
        for i, w_i in enumerate(A_w_imag):
            if abs(w_i) > 1.0e-10:
                print("*** WARNING***")
                print("    Root {:>5}: |Imag[A_w]| > 1.0E-10!".format(i))
                print("              : |Imag[A_v[{}]]| = {:.2E}".format(
                    i, np.linalg.norm(A_v_imag[:, i])))

        # here, check if no residuals > max no residuals, if so, collapse subspace
        if nl >= max_ss_size:
            print("Subspace too big. Collapsing.\n")
            B = np.dot(B, A_v)
            continue
        # else, build residual vectors
        ## residual_i = sum(j) sigma_j* eigvec_i - eigval_i * B_j * eigvec_i
        norm = np.zeros(no_eigs)
        istat['rnorm'] = np.zeros(no_eigs)
        istat['de'] = np.zeros(no_eigs)
        new_Bs = []
        # Only need a residual for each desired root, not one for each guess
        force_collapse = False
        for i in range(no_eigs):
            if conv_roots[i]:
                print("    ROOT {:<3}: CONVERGED!".format(i))
                continue
            residual = np.dot(sigma, A_v[:, i]) - A_w[i] * np.dot(B, A_v[:, i])

            # check for convergence by norm of residuals
            norm[i] = np.linalg.norm(residual)
            istat['rnorm'][i] = norm[i]
            # apply the preconditioner
            precon_resid = preconditioner(residual, i, A_w[i])
            de = abs(A_w_old[i] - A_w[i])
            istat['de'][i] = de
            print("    ROOT {:<3} de = {:<10.6f} |r| = {:<10.6f}".format(
                i, de, norm[i]))
            conv_roots[i] = (de < e_conv) and (norm[i] < r_conv)
            if conv_roots[i]:
                force_collapse = True
            else:
                new_Bs.append(precon_resid)

        # check for convergence by diff of eigvals and residual norms
        # r_norm = np.linalg.norm(norm)
        # eig_norm = np.linalg.norm(A_w - A_w_old)
        A_w_old = A_w
        #if( r_norm < r_conv) and (eig_norm < e_conv):
        istat['time'] = time.time() - istat['time']
        stats['iters'].append(istat)
        if all(conv_roots):
            converged = True
            print("Davidson converged at iteration number {}".format(count))
            print("{:<3}|{:^20}|".format('count', 'eigenvalues'))
            print("{:<3}|{:^20}|".format('=' * 3, '=' * 20))
            retvecs = np.dot(B, A_v)
            for i, val in enumerate(A_w):
                print("{:<3}|{:<20.12f}".format(i, val))
            stats['final_vals'] = np.sqrt(A_w)
            for i in range(len(stats['iters'])):
                stats['iters'][i]['e'] = np.sqrt(stats['iters'][i]['e'])
            return A_w, retvecs, stats
        else:
            if force_collapse:
                B = np.dot(B, A_v)
            n_left_to_converge = np.count_nonzero(np.logical_not(conv_roots))
            n_converged = np.count_nonzero(conv_roots)
            max_ss_size = n_converged + (
                n_left_to_converge * max_vecs_per_root)
            B = np.column_stack(
                tuple(B[:, i] for i in range(B.shape[-1])) + tuple(new_Bs))
        count += 1

    if not converged:
        print("Davidson did not converge. Max iterations exceeded.")
        return None, None, stats


def symp_solver(pm_function,
                preconditioner,
                guess,
                e_conv=1.0E-8,
                r_conv=None,
                no_eigs=1,
                max_vecs_per_root=20,
                maxiter=100):
    """
    Given the problem:
     [A B] [X] = w[1  0] [X]
     [B A] [Y]    [0 -1] [Y]


    Parameters:
    -----------
    pm_function: function(l * array(N)) -> 2*l * array(N)
       Computes: (A+B) bi, (A-B) bi
    preconditioner: function(array(N), index, w)
       The function applied to residual vectors before using them to augment the subspace
    guess: array(N, l)
       A matrix with guess vectors in the columns. There should be N rows, and l columns where l is at least equal to k.
    e_conv: float {default 10^-8}
       The maximal difference allowed between successive approximate eigenvalues, before they are considered converged
       Note, This is not actually used to check convergence
    r_conv: float {default e_conv * 19^-2}
       The maximal value for the norm of a residual vector before the corresponding root is considered converged.
    no_roots: int (<0, >=N)
       The number of roots (k) that are desired
    max_vecs_per_root: int
       Collapse will be triggered if subspace expands beyond `no_eigs` * this
    maxiter: int {default 100}
       If convergence is not achieved after this number of iterations an error will be raised.
    """

    if r_conv is None:
        r_conv = e_conv / 100

    # number of guess vectors
    nl = guess.shape[-1]
    # number of desired roots
    nk = no_eigs
    # Dimensionality of full problem
    N = guess.shape[0]
    B = guess

    stats = {}
    stats['iters'] = []
    #    stats['Nguess'] = nl
    stats['Nov'] = N
    stats['Nroot'] = nk
    stats['solver_name'] = 'Symplectic'

    # require nl > nk
    assert nl >= nk, "The number of guesses must be at least the number of roots"

    iter_info = {
        'count': 0,
        '|WL|': [None] * nk,
        '|WR|': [None] * nk,
        'dw': [None] * nk,
        'w': [0.0] * nk,
        'nl': nl,
        'conv': [False] * nk
    }
    ss_max = max_vecs_per_root * nk

    def stop(info):
        return False

    def display_iter_info(info):
        ieconv = abs(int(math.log10(e_conv))) + 2
        vwidth = ieconv + 2
        irconv = abs(int(math.log10(r_conv))) + 2
        rwidth = irconv + 2
        val_fmt = "<{}.{}".format(vwidth, ieconv)
        res_fmt = "<{}.{}".format(rwidth, irconv)
        root_info_title = "          {{: ^5}} | {{: ^{vwidth}}} | {{: ^{vwidth}}} | {{: ^{rwidth}}} | {{: ^{rwidth}}}".format(
            rwidth=rwidth, vwidth=vwidth)
        root_info_fmt = "     ROOT {{n:<5}} | {{w: {val_fmt}}} | {{dw: {val_fmt}}} | {{normR: {res_fmt}}} | {{normL: {res_fmt}}} |".format(
            val_fmt=val_fmt, res_fmt=res_fmt)
        print("Iteration: {:<5}       subspace dimension: {:<5}".format(
            info['count'], info['nl']))
        print(root_info_title.format(" ", "E (au)", "dE (au)", "|WR|", "|WL|"))
        print(root_info_title.format(" ", "------", "-------", "----", "----"))
        for i in range(nk):
            if info['conv'][i]:
                print(
                    root_info_fmt.format(
                        n=i,
                        w=info['w'][i],
                        dw=info['dw'][i],
                        normR=info['|WR|'][i],
                        normL=info['|WL|'][i]) + "  CONVERGED!")
            else:
                print(
                    root_info_fmt.format(
                        n=i,
                        w=info['w'][i],
                        dw=info['dw'][i],
                        normR=info['|WR|'][i],
                        normL=info['|WL|'][i]))

    while iter_info['count'] < maxiter:
        print("Sympsolve iter {}".format(iter_info['count']))
        istat = {}
        istat['time'] = time.time()
        # orthogonalized guess
        B, r = np.linalg.qr(B)
        nl = B.shape[-1]
        istat['nvecs'] = nl
        istat['nprod'] = 0

        if nl >= ss_max:
            collapse = True
        else:
            collapse = False

        iter_info['nl'] = nl
        # Step 2: compute (P)*bi and (M)*bi
        Pb = np.zeros_like(B)
        Mb = np.zeros_like(B)
        for i in range(nl):
            Px_i, Mx_i = pm_function(B[:, i])
            Pb[:, i] = Px_i
            Mb[:, i] = Mx_i
            istat['nprod'] += 2

        # Step 3: form Pss, Mss. The P/M matrices in the subspace
        Pss = np.dot(B.T, Pb)
        Mss = np.dot(B.T, Mb)
        # Step 4: Hermitian Product (Subspace analog of M^{1/2} P M^{1/2})
        Mss_val, Mss_vec = np.linalg.eigh(Mss)
        Mss_half = np.einsum('ij,j,kj->ik', Mss_vec, np.sqrt(Mss_val), Mss_vec)
        Hss = np.einsum('ij,jk,km->im', Mss_half, Pss, Mss_half)
        # Step 5: diagonalize Hss -> w^2, Tss
        w, Tss = np.linalg.eigh(Hss)
        w = np.sqrt(w)
        idx = np.argsort(w)
        w = w[idx]
        Tss = Tss[:, idx]

        #Step 6a: extract Rss = M^{1/2}Tss
        Rss = np.dot(Mss_half, Tss)
        #Step 6b: extract Lss = Pss Rss * w^-1
        Lss = np.dot(Pss, Rss)
        Lss = np.einsum('ij,j->ij', Lss, np.divide(1.0, w))

        WL = np.einsum("Ni,ik->Nk", Pb, Rss) - np.einsum(
            "Ni,ik,k->Nk", B, Lss, w)
        WR = np.einsum("Ni,ik->Nk", Mb, Lss) - np.einsum(
            "Ni,ik,k->Nk", B, Rss, w)
        new_space = []
        istat['rnorm'] = np.zeros(nk)
        istat['de'] = np.zeros(nk)
        istat['e'] = w[:nk]
        for k in range(nk):
            eps_k = np.sum(WR[:, k]**2) + np.sum(WL[:, k]**2)
            iter_info['|WR|'][k] = eps_k
            iter_info['|WL|'][k] = eps_k
            istat['rnorm'][k] = iter_info['|WR|'][k]
            istat['de'][k] = iter_info['dw'][k]
            iter_info['dw'][k] = np.abs(iter_info['w'][k] - w[k])
            iter_info['w'][k] = w[k]
            iter_info['conv'][k] = eps_k < r_conv
            if iter_info['conv'][k]:
                continue
            else:
                p = preconditioner(WL[:, k], i, w[k])
                q = preconditioner(WR[:, k], i, w[k])
                # p = p / np.sqrt(np.abs(p.dot(q)))
                # q = (q * np.sign(p.dot(q))) / np.sqrt(np.abs(p.dot(q)))
                new_space.append(p)
                new_space.append(q)

        istat['time'] = time.time() - istat['time']
        display_iter_info(iter_info)
        stats['iters'].append(istat)
        if all(iter_info['conv']):
            print("conv catch")
            final_R = np.dot(B, Rss[:, :nk])
            final_L = np.dot(B, Lss[:, :nk])
            stats['final_vals'] = w[:nk]
            return w[:nk], final_L, final_R, stats
        else:
            print("expand subspace")
            #collapse = False
            if collapse:
                print("Collapse")
                R = np.dot(B, Rss[:, :nk])
                L = np.dot(B, Lss[:, :nk])
                new_space = tuple(R.T) + tuple(L.T)
            else:
                new_space = tuple(new_space) + tuple(B.T)

        B = np.column_stack(new_space)
        iter_info['count'] += 1
    raise ConvergenceError("Non-Hermitian Davidson solver failed to converge")
