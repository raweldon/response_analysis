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

    # global fit
    eps = symfit.variables(', '.join('ep_{}'.format(i) for i in range(len(sorted_dfs))))
    los = symfit.variables(', '.join('lo_{}'.format(i) for i in range(len(sorted_dfs))))
    a, d = symfit.parameters('a, d')
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
    print( fit_result.value(a))

def lmfit_global_fit(sorted_dfs, E_p, save_results):

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

    # dfs = sorted_dfs
    # sorted_dfs = []
    # for i in range(100):
    #     sorted_dfs.append(random.choice(dfs))

    print('\nPerforming global fit with lmfit\n')
    x_data = np.array(E_p)
    y_data = []
    for idx, sorted_df in enumerate(sorted_dfs):
        y_data.append(list(sorted(sorted_df.param.values)))
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
    lmfit.report_fit(res.params, show_correl=False)

    if save_results:
        results = []
        for name, par in res.params.items():
            results.append((name, par.value, par.stderr))
        np.save(cwd + '/pickles/lmfit_results', np.array(results))

    colors = cm.viridis(np.linspace(-1, 1, len(sorted_dfs)))
    plt.figure()
    for i in range(len(sorted_dfs)):
        y_fit = fit_dataset(res.params, i, x_data)
        plt.plot(x_data, y_data[i, :], 'o', color=colors[i])
        plt.plot( x_data, y_fit, '--', color=colors[i], alpha=0.6)  

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
    lmfit_global_fit(sorted_dfs, E_p, save_results)


if __name__ == '__main__':
    main()
    print( "--- %s seconds ---" % (time.time() - start_time))
    plt.show()

