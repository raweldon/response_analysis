''' Uses python 3.7'''

import lmfit
import numpy as np
import glob
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import symfit
from symfit import exp
import time
import random
from scipy.special import lpmv

start_time = time.time()
cwd = os.getcwd()

def cartesian_to_spherical(x, y, z):
    # definition as used in lo_analysis
    theta = np.arccos(z) 
    phi = np.arctan2(y, x) # arctan2 ensures the correct qudrant of (x, y)
    # correct rounding errors
    phi[abs(phi) < 0.00001] = 0.
    # correct 360 deg symmetry
    phi[phi < -3.14159] = 3.141593
    return theta, phi

def order_by_theta_phi(df, theta, phi):
    data = np.array([df.param.values, theta, phi, df.x.values, df.y.values, df.z.values, df.E_p.values])
    data = data.T
    df = pd.DataFrame(data, columns=['param', 'theta', 'phi', 'x', 'y', 'z', 'E_p'])
    # group all duplicated values of theta, phi 
    df = df.sort_values(by=['x', 'y', 'z'])
    #print( df.to_string())
    return df

def fit(E_p, a, b, c, d):
    return a*E_p - b*(1-np.exp(-c*E_p**d))

def individual_fit(sorted_dfs):
    ''' fit lo vs ep individually for all directions '''

    exp_model = lmfit.Model(fit)
    params = exp_model.make_params(a=0.76, b=2.8, c=0.25, d=0.98)
    params['b'].max = 20
    params['a'].vary = False
    params['d'].vary = False

    colors = cm.viridis(np.linspace(-1, 1, len(sorted_dfs)))
    fit_vals, xs, ys, zs, bad_sign_dfs = [], [], [], [], []
    for idx, sorted_df in enumerate(sorted_dfs):

        xs.append(sorted_df.iloc[0]['x'])
        ys.append(sorted_df.iloc[0]['y'])
        zs.append(sorted_df.iloc[0]['z'])

        res = exp_model.fit(sorted_df.param.values, params, E_p=sorted_df.E_p.values)
        fit_vals.append((res.params['a'].value, res.params['b'].value, res.params['c'].value, res.params['d'].value))

        # if res.params['b'].value > 7.99:
        #     print( lmfit.fit_report(res, show_correl=True))
        #     print( '\n', res.message)
        #     print(sorted_df.to_string())

        # plot fits
        xvals = np.linspace(0, 10, 100)
        plt.figure(0)
        line, = plt.plot(xvals, fit(xvals, res.params['a'].value, res.params['b'].value, res.params['c'].value, res.params['d'].value), 
                         linestyle='--', color=colors[idx])
        plt.plot(sorted_df.E_p.values, sorted_df.param.values, 'o', color=colors[idx])

    # for val in sorted(fit_vals, key=lambda x: x[1]):
    #     print( val)

    a, b, c, d = np.array(fit_vals).T
    print( '\n%7s %7s %7s %7s %7s %7s %7s %7s' % ('a_mean', 'a_std', 'b_mean', 'b_std', 'c_mean', 'c_std', 'd_mean', 'd_std'))
    print( '%7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f %7.4f' % (np.mean(a), np.std(a), np.mean(b), np.std(b), np.mean(c), np.std(c), np.mean(d), np.std(d)))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(xs, ys, zs, c=a)
    ax.set_title('a')
    fig.colorbar(p)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(xs, ys, zs, c=b)
    ax.set_title('b')
    fig.colorbar(p)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(xs, ys, zs, c=c)
    ax.set_title('c')
    fig.colorbar(p)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(xs, ys, zs, c=d)
    ax.set_title('d')
    fig.colorbar(p)

def symfit_global_fit(sorted_dfs, E_p):
    ''' attempted global fit with symfit
        not working...
    '''
    dfs = sorted_dfs
    sorted_dfs = []
    for i in range(10):
        sorted_dfs.append(random.choice(dfs))   

    print('\nPerforming global fit on %i directions with symfit\n' % len(sorted_dfs))

    eps = symfit.variables(', '.join('ep_{}'.format(i) for i in range(len(sorted_dfs))))
    los = symfit.variables(', '.join('lo_{}'.format(i) for i in range(len(sorted_dfs))))
    a, d = symfit.parameters('a, d')
    a.value = 0.74
    a.max = 1
    a.min = 0.5
    d.value = 0.98
    d.max = 1.1
    d.min = 0.9
    bs = symfit.parameters(', '.join('b_{}'.format(i) for i in range(len(sorted_dfs))))
    cs = symfit.parameters(', '.join('c_{}'.format(i) for i in range(len(sorted_dfs))))

    model_dict = {
        lo: a*ep - b*(1 - exp(-c*ep**d))
            for lo, ep, b, c in zip(los, eps, bs, cs)
    }

    x_data = [E_p]*len(sorted_dfs)
    x_data = {'ep_'+str(i):x for i, x in enumerate(x_data)}

    y_data = []
    for idx, sorted_df in enumerate(sorted_dfs):
        y_data.append(list(sorted_df.param.values))
    y_data = {'lo_'+str(i):y for i, y in enumerate(y_data)}
    data = {**x_data, **y_data}

    fit = symfit.Fit(model_dict, **data)
    fit_result = fit.execute()
    print( fit_result)

def lmfit_global_fit(sorted_dfs, E_p, save_results):
    ''' global fit of Lo vs E_p for all directions
        a, d are shared between all directions
        b, c are unique for each direction
    '''

    def fit_dataset(params, i, x):
        a = params['a_%i' % (i)]
        b = params['b_%i' % (i)]
        c = params['c_%i' % (i)]       
        d = params['d_%i' % (i)]
        return fit(x, a, b, c, d)

    def objective(params, x, data):
        '''Calculate total residual for fits to several data sets '''
        ndata, _ = data.shape
        resid = 0.0*data[:]

        # make residual per data set
        for i in range(ndata):
            #print(data[i, :], '\n\n', fit_dataset(params, i, x))
            resid[i, :] = data[i, :] - fit_dataset(params, i, x)

        # now flatten this to a 1D array, as minimize() needs
        return resid.flatten()

    dfs = sorted_dfs
    sorted_dfs = []
    for i in range(10):
        sorted_dfs.append(random.choice(dfs))

    print('\nPerforming global fit on %i directions with lmfit\n' % len(sorted_dfs))
    x_data = np.array(E_p)
    y_data, xs, ys, zs = [], [], [], []
    for idx, sorted_df in enumerate(sorted_dfs):
        y_data.append(list(sorted(sorted_df.param.values)))
        xs.append(sorted_df.iloc[0]['x'])
        ys.append(sorted_df.iloc[0]['y'])
        zs.append(sorted_df.iloc[0]['z'])
    y_data = np.array(y_data)

    fit_params = lmfit.Parameters()
    for idx in range(len(sorted_dfs)):
        fit_params.add('a_%i' % (idx), value=0.76, min=0, max=1)
        fit_params.add('b_%i' % (idx), value=2.8, min=0, max=10)
        fit_params.add('c_%i' % (idx), value=0.25, min=0, max=1)
        fit_params.add('d_%i' % (idx), value=0.98, min=0.9, max=1.5)

    # constrain values of a and d to be the same for all fits
    for idx in range(1, len(sorted_dfs)):
        fit_params['a_%i' % idx].expr = 'a_0'
        fit_params['d_%i' % idx].expr = 'd_0'

    res = lmfit.minimize(objective, fit_params, args=(x_data, y_data))
    #lmfit.report_fit(res.params, show_correl=False)

    if save_results:
        results_a, results_b, results_c, results_d = [], [], [], []
        idx = 0
        for name, par in res.params.items():
            if 'a_' in name: 
                results_a.append((name, par.value, par.stderr, xs[idx], ys[idx], zs[idx]))
            if 'b_' in name: 
                results_b.append((name, par.value, par.stderr, xs[idx], ys[idx], zs[idx]))
            if 'c_' in name: 
                results_c.append((name, par.value, par.stderr, xs[idx], ys[idx], zs[idx]))
            if 'd_' in name: 
                results_d.append((name, par.value, par.stderr, xs[idx], ys[idx], zs[idx]))
                idx += 1
        print('\n\n\n', len(results_b))
        np.save(cwd + '/pickles/lmfit_results', np.array([results_a, results_b, results_c, results_d]))
        np.save(cwd + '/pickles/lmfit_res_covar', res.covar)
        np.save(cwd + '/pickles/lmfit_res_var_names', res.var_names)
        #lmfit.model.save_modelresult(res, 'lmfit_model_result.sav') # doesn't work...

    colors = cm.viridis(np.linspace(-1, 1, len(sorted_dfs)))
    plt.figure()
    for i in range(len(sorted_dfs)):
        y_fit = fit_dataset(res.params, i, x_data)
        plt.plot(x_data, y_data[i, :], 'o', color=colors[i])
        plt.plot( x_data, y_fit, '--', color=colors[i], alpha=0.6)  

def legendre_poly_param_fit_same_coeffs(sorted_dfs):

    def get_coeff_names(order, central_idx_only):
        ''' return list of coefficient names for a given order
            set central_idx_only = True if only using central index terms
            set range in lmfit section to plot desired order of sph harmonics
        '''
        names, order_coeff = [], []

        if central_idx_only:
            for o in range(0, order + 1):
                names.append('c_' + str(o) + str(o))
                order_coeff.append((o, o))
            return names, order_coeff
        # else:
        #     for o in range(0, order + 1):
        #         coeff_no = 2*o + 1
        #         idx = coeff_no - o - 1
        #         for i in range(0, coeff_no):
        #             names.append('c_' + str(i) + str(o))
        #             order_coeff.append((o, i - idx)) # n, m
        #             print( order_coeff)
        #     return names, order_coeff

        # positive orders only
        else:
            for o in range(0, order + 1):
                coeff_no = o + 1
                idx = coeff_no - o - 1
                for i in range(0, coeff_no):
                    names.append('c_' + str(i) + str(o))
                    order_coeff.append((o, i - idx)) # n, m
            # for c, n in zip(order_coeff, names):
            #     print( c, n)
            return names, order_coeff

    def add_params(names_t, names_p):
        # create parameter argument for lmfit
        fit_params = lmfit.Parameters()
        for t, p in zip(names_t, names_p):
            fit_params.add(t, value=0.5)
            fit_params.add(p, value=0.5)
        return fit_params

    def minimize(fit_params, *args):
        vals, theta, phi, names_t, names_p, order_coeff = args

        pars = fit_params.valuesdict()
        legendre_t, legendre_p = 0, 0
        for name_t, name_p, oc in zip(names_t, names_p, order_coeff):
            ct = pars[name_t]
            cp = pars[name_p]
            legendre_t += ct*lpmv(oc[1], oc[0], np.cos(theta))  # oc[0] = n (degree n>=0), oc[1] = m (order |m| <= n)
            legendre_p += cp*lpmv(oc[1], oc[0], np.sin(phi)) # real value of spherical hamonics

        legendre = legendre_t * legendre_p
        return vals - legendre

    a_data, b_data, c_data, d_data = np.load(cwd + '/pickles/lmfit_results.npy', allow_pickle=True)
    df_a = pd.DataFrame(a_data, columns=['params', 'val', 'stderr', 'x', 'y', 'z'])
    df_b = pd.DataFrame(b_data, columns=['params', 'val', 'stderr', 'x', 'y', 'z'])
    df_c = pd.DataFrame(c_data, columns=['params', 'val', 'stderr', 'x', 'y', 'z'])
    df_d = pd.DataFrame(d_data, columns=['params', 'val', 'stderr', 'x', 'y', 'z'])

    xs = np.array([float(x) for x in df_b.x.values])
    ys = np.array([float(y) for y in df_b.y.values])
    zs = np.array([float(z) for z in df_b.z.values])
    thetas, phis = cartesian_to_spherical(xs, ys, zs)

    vals = [float(i) for i in df_b.val.values]

    for i in range(1, 14):
    #for i in [1, 11]:
        print( '\norder = %i' % i)
        order = i
        # generate coefficients for phi and theta terms
        names_p, order_coeff_p = get_coeff_names(order, central_idx_only=True)
        names_t, order_coeff_t = get_coeff_names(order, central_idx_only=True)
        fit_params = add_params(names_t, names_p)
        #fit_kws={'nan_policy': 'omit'}
        res = lmfit.minimize(minimize, fit_params, args=(vals, thetas, phis, names_t, names_p, order_coeff_t))
        print( '\n', res.message)
        print( lmfit.fit_report(res, show_correl=False))

        leg_poly_t, leg_poly_p = 0, 0
        for idx, (name, par) in enumerate(res.params.items()):
            leg_poly_t += par.value*lpmv(order_coeff_t[idx][1], order_coeff_t[idx][0], np.cos(thetas))
            leg_poly_p += par.value*lpmv(order_coeff_t[idx][1], order_coeff_t[idx][0], np.sin(phis))
            #print leg_poly_t
        legendre_poly_fit = leg_poly_t * leg_poly_p
        legendre_poly_sorted = sorted(legendre_poly_fit)
        vals_idx = np.argsort(vals)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(xs[vals_idx], ys[vals_idx], zs[vals_idx], c=legendre_poly_sorted)
        ax.set_title('b')
        fig.colorbar(p)

        #for v, l in zip(sorted(vals), legendre_poly_sorted):
        #    print( '%8.3f %8.3f' % (v, l))

        print( '\n\n max_fit  min_fit  max_legen  min_legen')
        print( '%8.4f %8.4f %8.4f %10.4f' % (max(vals), min(vals), max(legendre_poly_sorted), min(legendre_poly_sorted)))

    print('\nmean std = %6.4f' % np.mean([float(x) for x in df_b.stderr.values]))

def legendre_poly_param_fit_diff_coeffs(sorted_dfs, orders):
    def get_coeff_names(order, name, central_idx_only):
        ''' return list of coefficient names for a given order
            set central_idx_only = True if only using central index terms
            set range in lmfit section to plot desired order of sph harmonics
        '''
        names, order_coeff = [], []

        if central_idx_only:
            for o in range(0, order + 1):
                names.append(name + str(o) + str(o))
                order_coeff.append((o, o))
            return names, order_coeff
        # else:
        #     for o in range(0, order + 1):
        #         coeff_no = 2*o + 1
        #         idx = coeff_no - o - 1
        #         for i in range(0, coeff_no):
        #             names.append(name + str(i) + str(o))
        #             order_coeff.append((o, i - idx)) # n, m
        #             print( order_coeff)
        #     return names, order_coeff

        # positive orders only
        else:
            for o in range(0, order + 1):
                coeff_no = o + 1
                idx = coeff_no - o - 1
                for i in range(0, coeff_no):
                    names.append(name + str(i) + str(o))
                    order_coeff.append((o, i - idx)) # n, m
            # for c, n in zip(order_coeff, names):
            #     print( c, n)
            return names, order_coeff

    def add_params(names_t, names_p):
        # create parameter argument for lmfit
        fit_params = lmfit.Parameters()
        for t, p in zip(names_t, names_p):
            fit_params.add(t, value=1)
            fit_params.add(p, value=1)
        return fit_params

    def minimize(fit_params, *args):
        vals, theta, phi, names_t, names_p, order_coeff = args

        pars = fit_params.valuesdict()
        legendre_t, legendre_p = 0, 0
        for name_t, name_p, oc in zip(names_t, names_p, order_coeff):
            ct = pars[name_t]
            cp = pars[name_p]
            legendre_t += ct*lpmv(oc[1], oc[0], np.cos(theta))  # oc[0] = n (degree n>=0), oc[1] = m (order |m| <= n)
            legendre_p += cp*lpmv(oc[1], oc[0], np.sin(oc[1] * phi)) # real value of spherical hamonics

        legendre = legendre_t * legendre_p
        return vals - legendre

    a_data, b_data, c_data, d_data = np.load(cwd + '/pickles/lmfit_results.npy', allow_pickle=True)
    df_a = pd.DataFrame(a_data, columns=['params', 'val', 'stderr', 'x', 'y', 'z'])
    df_b = pd.DataFrame(b_data, columns=['params', 'val', 'stderr', 'x', 'y', 'z'])
    df_c = pd.DataFrame(c_data, columns=['params', 'val', 'stderr', 'x', 'y', 'z'])
    df_d = pd.DataFrame(d_data, columns=['params', 'val', 'stderr', 'x', 'y', 'z'])

    xs = np.array([float(x) for x in df_b.x.values])
    ys = np.array([float(y) for y in df_b.y.values])
    zs = np.array([float(z) for z in df_b.z.values])
    thetas, phis = cartesian_to_spherical(xs, ys, zs)

    vals = [float(i) for i in df_b.val.values]

    for i in orders: # over 11 is overkill
    #for i in [1, 11]:
        print( '\norder = %i' % i)
        order = i
        # generate coefficients for phi and theta terms
        names_p, order_coeff_p = get_coeff_names(order, name='p_', central_idx_only=True)
        names_t, order_coeff_t = get_coeff_names(order, name='t_', central_idx_only=True)
        fit_params = add_params(names_t, names_p)
        #fit_kws={'nan_policy': 'omit'}
        res = lmfit.minimize(minimize, fit_params, args=(vals, thetas, phis, names_t, names_p, order_coeff_t))
        print( '\n', res.message)
        print( lmfit.fit_report(res, show_correl=False))

        leg_poly_t, leg_poly_p, t_idx, p_idx = 0, 0, 0, 0
        for idx, (name, par) in enumerate(res.params.items()):
            if 't_' in name:
                #print(name, par.value)
                leg_poly_t += par.value*lpmv(order_coeff_t[t_idx][1], order_coeff_t[t_idx][0], np.cos(thetas))
                t_idx += 1
            elif 'p_' in name:
                #print(name, par.value)
                leg_poly_p += par.value*lpmv(order_coeff_p[p_idx][1], order_coeff_p[p_idx][0], np.sin(order_coeff_p[p_idx][1]*phis))
                p_idx += 1
            else:
                print('name is ', name)
        legendre_poly_fit = leg_poly_t * leg_poly_p
        legendre_poly_sorted = sorted(legendre_poly_fit)
        vals_idx = np.argsort(vals)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(xs[vals_idx], ys[vals_idx], zs[vals_idx], c=legendre_poly_sorted)
        ax.set_title('b')
        fig.colorbar(p)

        print( '\n\n max_fit  min_fit  max_legen  min_legen')
        print( '%8.4f %8.4f %8.4f %10.4f' % (max(vals), min(vals), max(legendre_poly_sorted), min(legendre_poly_sorted)))

    print('\nmax_val std = %6.4f' % [float(x) for x in df_b.stderr.values][np.argmax(vals)])
    return legendre_poly_sorted, xs[vals_idx], ys[vals_idx], zs[vals_idx]

def sph_harm_fit(sorted_dfs):
    from scipy.special import sph_harm

    def get_coeff_names(order, central_idx_only):
        ''' return list of coefficient names for a given order
            set central_idx_only = True if only using central index terms
            set range in lmfit section to plot desired order of sph harmonics
        '''
        names, order_coeff = [], []

        if central_idx_only:
            for o in range(0, order + 1):
                names.append('c_' + str(o) + str(o))
                order_coeff.append((o, o))
            return names, order_coeff
        else:
            for o in range(0, order + 1):
                coeff_no = 2*o + 1
                idx = coeff_no - o - 1
                for i in range(0, coeff_no):
                    names.append('c_' + str(i) + str(o))
                    order_coeff.append((o, i - idx))
            return names, order_coeff

    def add_params(names):
        # create parameter argument for lmfit
        fit_params = lmfit.Parameters()
        for name in names:
            fit_params.add(name, value=1)
        return fit_params

    def minimize(fit_params, *args):
        ql, theta, phi, names, order_coeff = args

        pars = fit_params.valuesdict()
        y_lm = 0
        for name, (l, m) in zip(names, order_coeff):
            c = pars[name]
            #harmonics += c*sph_harm(oc[1], oc[0], theta, phi).real #oc[0] = n (degree n>=0), oc[1] = m (order |m| <= n)
            if m < 0:
                y_lm += c*lpmv(m, l, np.cos(theta))*np.sin(abs(m)*phi)
            elif m == 0:
                y_lm += c*lpmv(m, l, np.cos(theta))
            else:
                y_lm += c*lpmv(m, l, np.cos(theta))*np.cos(m*phi)
            # if m < 0:
            #     y_lm += c*lpmv(m, l, np.cos(theta))*np.sin(abs(m)*(phi + np.pi))
            # elif m == 0:
            #     y_lm += c*lpmv(m, l, np.cos(theta))
            # else:
            #     y_lm += c*lpmv(m, l, np.cos(theta))*np.cos(m*(phi + np.pi))
        #harmonics = harmonics.real
        return ql - y_lm

    a_data, b_data, c_data, d_data = np.load(cwd + '/pickles/lmfit_results.npy', allow_pickle=True)
    df_a = pd.DataFrame(a_data, columns=['params', 'val', 'stderr', 'x', 'y', 'z'])
    df_b = pd.DataFrame(b_data, columns=['params', 'val', 'stderr', 'x', 'y', 'z'])
    df_c = pd.DataFrame(c_data, columns=['params', 'val', 'stderr', 'x', 'y', 'z'])
    df_d = pd.DataFrame(d_data, columns=['params', 'val', 'stderr', 'x', 'y', 'z'])

    xs = np.array([float(x) for x in df_b.x.values])
    ys = np.array([float(y) for y in df_b.y.values])
    zs = np.array([float(z) for z in df_b.z.values])
    thetas, phis = cartesian_to_spherical(xs, ys, zs)

    vals = [float(i) for i in df_b.val.values]

    for i in range(1, 20):
    #for i in [1, 11]:
        print( '\norder = %i' % i)
        order = i
        # generate coefficients for phi and theta terms
        names, order_coeff = get_coeff_names(order, central_idx_only=True)
        fit_params = add_params(names)
        fit_kws={'nan_policy': 'omit'}
        res = lmfit.minimize(minimize, fit_params, args=(vals, thetas, phis, names, order_coeff))
        print( '\n', res.message)
        print( lmfit.fit_report(res, show_correl=False))

        y_lm = 0
        for idx, (name, par) in enumerate(res.params.items()):
            m = order_coeff[idx][1]
            l = order_coeff[idx][0]
            c = par.value
            #sph_harmonics += par.value*sph_harm(order_coeff[idx][1], order_coeff[idx][0], thetas, phis)
            if m < 0:
                y_lm += c*lpmv(m, l, np.cos(thetas))*np.sin(abs(m)*phis)
            elif m == 0:
                y_lm += c*lpmv(m, l, np.cos(thetas))
            else:
                y_lm += c*lpmv(m, l, np.cos(thetas))*np.cos(m*phis)
            # if m < 0:
            #     y_lm += c*lpmv(m, l, np.cos(thetas))*np.sin(abs(m)*(phis + np.pi))
            # elif m == 0:
            #     y_lm += c*lpmv(m, l, np.cos(thetas))
            # else:
            #     y_lm += c*lpmv(m, l, np.cos(thetas))*np.cos(m*(phis + np.pi))

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        p = ax.scatter(xs, ys, zs, c=y_lm)
        ax.set_title('b')
        fig.colorbar(p)

        print( '\n\n max_fit  min_fit  max_legen  min_legen')
        print( '%8.4f %8.4f %8.4f %10.4f' % (max(vals), min(vals), max(y_lm), min(y_lm)))

def main():
    
    # sort data for global fit
    det_angles = [20, 30, 40, 50, 60, 70]
    E_p = sorted([4.83*np.sin(np.deg2rad(a))**2 for a in det_angles] + [11.33*np.sin(np.deg2rad(a))**2 for a in det_angles])
    files = glob.glob(cwd + '/text_data/lo*smoothed*')
 
    dfs = []
    for fi in files:
        with open(fi, 'r') as f:
            dfs.append(pd.read_csv(f, delimiter=r'\s+', names=['param', 'x', 'y', 'z', 'E_p']))

    df = pd.concat(dfs).reset_index(drop=True)
    # correct rounding errors
    df[(df < 1e-5) & (df > -1e-5)] = 0

    # group by x, y, z trajectory and make list of dataframes
    grouped_df = df.groupby(['x', 'y', 'z'])
    sorted_dfs = []
    for key, item in grouped_df:
        sorted_dfs.append(grouped_df.get_group(key))

    #individual_fit(sorted_dfs)    
    #symfit_global_fit(sorted_dfs, E_p)
    #lmfit_global_fit(sorted_dfs, E_p, save_results=False)
    #legendre_poly_param_fit_same_coeffs(sorted_dfs)
    #sph_harm_fit(sorted_dfs)

    order = (11,)
    legendre_fit, xs, ys, zs = legendre_poly_param_fit_diff_coeffs(sorted_dfs, order)



if __name__ == '__main__':
    main()
    print( "--- %s seconds ---" % (time.time() - start_time))
    plt.show()

