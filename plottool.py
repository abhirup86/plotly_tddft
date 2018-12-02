import numpy as np
import pandas as pd

import colorlover as cl
import plotly.graph_objs as go


def _extract_iterstat(iters, colname):
    return np.array([iter_stat[colname] for iter_stat in iters])


def split_cols(array, axis, key_fmt):
    dim = len(array.shape)
    assert axis >= 0, "invalid axis < 0"
    assert axis < dim, "invalid axis >= dim"
    cols = {}
    iter_limit = array.shape[axis]
    for i in range(iter_limit):
        col_key = key_fmt.format(i)
        slice_tup = (i if i == axis else slice(None) for i in range(dim))
        cols[col_key] = array[slice_tup]
    return cols


def collect_data(final_vals, Nov, Nroot, iters, solver_name, system_label):
    data = {}
    niter = len(iters)

    conv_vals = np.array(final_vals)
    vals = _extract_iterstat(iters, 'e')
    # abs_de_pi_pr = np.array(
    #     np.abs(vals[it, :] - conv_vals) for it in range(niter))
    abs_de_pi_pr = np.abs(vals - conv_vals)
    data['avg_abs_de'] = np.average(abs_de_pi_pr, axis=1)
    data['max_abs_de'] = np.max(abs_de_pi_pr, axis=1)
    data['min_abs_de'] = np.min(abs_de_pi_pr, axis=1)

    rel_de_pi_pr = _extract_iterstat(iters, 'de')
    data['avg_rel_de'] = np.average(rel_de_pi_pr, axis=1)
    data['max_rel_de'] = np.max(rel_de_pi_pr, axis=1)
    data['min_rel_de'] = np.min(rel_de_pi_pr, axis=1)

    res_norm_pi_pr = _extract_iterstat(iters, 'rnorm')
    #data.updata(split_cols(res_norm_pi_pr, axis=1, key_fmt="r{}_res_norm"))
    data['avg_res_norm'] = np.average(res_norm_pi_pr, axis=1)
    data['max_res_norm'] = np.max(rel_de_pi_pr, axis=1)
    data['min_res_norm'] = np.min(rel_de_pi_pr, axis=1)

    nvec_pi = _extract_iterstat(iters, 'nvecs')
    bytes_per_ss_vec = Nov * 8
    data['mem_bytes'] = nvec_pi * bytes_per_ss_vec

    data['n_product_evals'] = np.cumsum(_extract_iterstat(iters, 'nprod'))
    data['solver'] = [solver_name] * niter
    data['system'] = [system_label] * niter
    data['niter'] = np.arange(niter)

    return pd.DataFrame(data)


def make_conv_plot(df, threshold, col_name='res_norm'):
    sys_2_color = {s: cl.scales['6']['qual']['Dark2'][i] for i, s in enumerate(df.system.unique())}
    solver_2_line = {'Davidson': 'dot', 'Symplectic': 'solid'}
    traces = []
    xmax = df.niter.max()
    for sysname, sysf in df.groupby('system'):
        color = sys_2_color[sysname]
        for solvname, syssolf in sysf.groupby('solver'):
            dash = solver_2_line[solvname]
            traces.append(
                go.Scatter(
                    x=syssolf.niter,
                    y=syssolf['max_' + col_name],
                    name=sysname + ' ' + solvname,
                    line=dict(color=color, dash=dash)))
    layout = go.Layout(
        yaxis=dict(type='log', exponentformat='e'),
        xaxis=dict(gridwidth=10),
        shapes=[dict(type='line', x0=0, x1=xmax, y0=threshold, y1=threshold, line=dict(width=4.0, color='red'))])

    return go.Figure(data=traces, layout=layout)
