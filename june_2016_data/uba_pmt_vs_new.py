''' Used to compare results from UBA PMT measurements (June 2016) with new measurements
    Used max LO measured from June data (not the fit) -- rough estimate of max LO
'''
import csv
import numpy as np
import lmfit
import matplotlib.pyplot as plt
import pickle
import pandas as pd

def get_a_axis_data(p_dir):
    data = pd.read_pickle(p_dir + 'a_axis_data.p')
    a_axis_data = pd.DataFrame(data)
    qls = a_axis_data[:6][4].values[::-1]
    
    angs = [20, 30, 40, 50, 60, 70]
    perg = [11.325*np.sin(np.deg2rad(ang))**2 for ang in angs]
    print '\nBA PMT\n E_p     ql'
    for q, p in zip(qls, perg):
        print '{:^6.2f} {:>8.3f}'.format(p, q)

    return qls, perg

def sin_func(x, a, b, phi):
    w = 2 # period is 2
    return a*np.sin(w*(x + phi)) + b

def fit_tilt_data(data, angles, print_report):
        # sinusoid fit with lmfit
        angles = [np.deg2rad(x) for x in angles]
        gmodel = lmfit.Model(sin_func)
        params = gmodel.make_params(a=1, b=1, phi=0)
        res = gmodel.fit(data, params=params, x=angles, nan_policy='omit')#, method='nelder')
        if print_report:
            print '\n', lmfit.fit_report(res)
        return res

def main(show_sin):
    p_dir = 'C:/Users/raweldon/Research/TUNL/git_programs/response_analysis/pickles/'
    a_axis_qls, perg_feb = get_a_axis_data(p_dir)

    directory = 'C:/Users/raweldon/Research/TUNL/git_programs/response_analysis/june_2016_data/'
    angles = (0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360) # rotation angles
    det_ang = (70.2, 61.4, 52.4, 43.4, 34.4, 25.1, 17, 20.4, 29.8, 38.9, 48.1, 56.9, 70.4)
    perg = [11.4*np.sin(np.deg2rad(ang))**2 for ang in det_ang]
    # bvert crystal data, refer to C:\Users\raweldon\Research\TUNL\beam_char_runs\6_24_run\cluster_analysis\peak_localization for additional info
    qls = np.genfromtxt(directory + 'ql_means_0.csv', delimiter=",", skip_footer=1).T

    ymax = []
    print '\n\nUBA PMT\n E_p    fit_max    meas_max'
    for q, ql in enumerate(qls):
        res = fit_tilt_data(ql, angles, print_report=False)
        pars = res.best_values
        x_vals = np.linspace(0, 380, 100)
        x_vals_rad = np.deg2rad(x_vals)
        y_vals = sin_func(x_vals_rad, pars['a'], pars['b'], pars['phi'])

        if show_sin:
            plt.figure()
            plt.errorbar(angles, ql, ecolor='black', markerfacecolor='r', fmt='o', 
                        markeredgecolor='k', markeredgewidth=1, markersize=10, capsize=1, label='det ' + str(det_ang[q]))
            plt.plot(x_vals, y_vals, 'r--')
            plt.legend()

        # get max y_vals (a-axis recoils)
        ymax.append(max(ql))
        print '{:^6.2f} {:>8.3f} {:>8.3f}'.format(perg[q], max(y_vals), max(ql))

    # plot june 2016 and feb 2018 qls
    plt.figure()
    plt.scatter(perg, ymax, color='b', s=50, alpha=0.7, label='UBA PMT')
    plt.scatter(perg_feb, a_axis_qls, color='r', s=50, alpha=0.7, marker='^', label='BA PMT')
    plt.ylabel('Light output (MeVee)', fontsize=16)
    plt.xlabel('Proton recoil energy (MeV)', fontsize=16)
    plt.legend(loc=2)
    plt.show()

if __name__ == '__main__':
    main(show_sin=False)
