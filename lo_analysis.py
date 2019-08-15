''' Main functions:
        tilt_check 
            - use to plot measured data for each tilt angle
            - data is fit with sinusoids
        scatter_check
            - scatter plot of measured data in 3d
            - points are colored according to the crystal measured (bvert or cpvert)
        plot_heatmaps
            - 3d heatmap plots
            - use delaunay triagulation for interpolation/extrapolation
'''
import pyface.qt  # must import first to use mayavi
import numpy as np 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
import re
import time
import os
from mayavi import mlab
import pylab as plt
import scipy
import lmfit
import pickle
import dask.array as da
import sys
sys.path.insert(1, os.getcwd() + '/../stilbene_uncertainties/')
from coinc_scatter_uncerts_final import calc_ep_uncerts

def pd_load(filename, p_dir):
    # converts pickled data into pandas DataFrame
    print '\nLoading pickle data from:\n', p_dir + filename
    with open(p_dir + filename, 'r') as f:
        data = pd.read_pickle(f)
    headers = data.pop(0)
    #print headers
    #time.sleep(100)
    return pd.DataFrame(data, columns=headers)

def split_filenames(data):
    # split filenames into columns of dataframe
    filenames = data.filename
    crys, erg, tilt, rot, det = [], [], [], [], []
    for f, fi in enumerate(filenames):
        tmp = re.split('_|\.', fi)
        # crystal
        crys.append(tmp[0])
        # energy
        if tmp[1] == '11MeV':
            erg.append(11)
        else:
            erg.append(4)
        # tilt
        tmp1 = re.split('(\d+)', tmp[2])
        if tmp1[0] == 'neg':
            tilt.append(str('-') + tmp1[1])
        else:
            tilt.append(tmp1[1])
        # rot
        tmp2 = re.split('(\d+)', tmp[3])
        rot.append(tmp2[1])
        # det
        tmp3 = re.split('(\d+)', tmp[4])
        det.append(tmp3[1])

    new_data = {'crystal': crys, 'energy': erg, 'tilt': tilt, 'rotation': rot, 'det_no': det}
    df = pd.DataFrame(data=new_data)

    df_full = pd.concat([df, data], axis=1)
    return df_full

def order_by_rot(data, beam_11MeV):
    # order by correct rot angle
    if beam_11MeV:
        if len(data.filename) == 19:
            rot_order = [15, 16, 17, 18, 6, 7, 0, 1, 2, 3, 4, 8, 9, 10, 5, 11, 12, 13, 14]
            angles = np.arange(0, 190, 10)
        else:
            rot_order = [15, 16, 17, 18, 19, 6, 7, 0, 1, 2, 3, 4, 8, 9, 10, 5, 11, 12, 13, 14]
            angles = np.arange(0, 200, 10)
    else:
        if len(data.filename) == 15:
            rot_order = [12, 13, 14, 5, 6, 0, 1, 2, 3, 7, 8, 9, 4, 10, 11]
            angles = np.array([0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 190])
        else:
            rot_order = [12, 13, 5, 6, 0, 1, 2, 3, 7, 8, 9, 4, 10, 11]
            angles = np.array([0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180])

    data = data.assign(rot_order = rot_order)
    data = data.sort_values('rot_order')
    return data, angles

def sin_func(x, a, b, phi):
    w = 2 # period is 2
    return a*np.sin(w*(x + phi)) + b

def tilt_check(det_data, dets, tilts, pickle_name, cwd, p_dir, beam_11MeV, print_max_ql, get_a_data, pulse_shape, delayed, prompt, show_plots, save_plots, save_pickle):
    ''' Use to check lo, pulse shape (pulse_shape=True), or delayed pulse (delayed=True) for a given det and tilt
        Data is fitted with sinusoids
        Sinusoid fit parameters can be saved for later use by setting save_pickle=True
        a and c' axes directions are marked with scatter points
    '''

    def fit_tilt_data(data, angles, print_report):
            # sinusoid fit with lmfit
            angles = [np.deg2rad(x) for x in angles]
            gmodel = lmfit.Model(sin_func)
            params = gmodel.make_params(a=1, b=1, phi=0.1)
            params['phi'].max = np.deg2rad(90)
            params['phi'].min = np.deg2rad(-90)
            res = gmodel.fit(data, params=params, x=angles, nan_policy='omit')#, method='nelder')
            if print_report:
                print '\n', lmfit.fit_report(res)
            return res

    def get_max_ql_per_det(det_data, dets, tilts):
        # use to check max_ql values for each tilt per detector (all tilts measure recoils along the a-axis)
        for det in dets:
            print '\n-----------\ndet_no ', det, '\n-----------\n'
            print 'tilt      ql'
            det_df = det_data.loc[(det_data.det_no == str(det))]
            for tilt in tilts:
                tilt_df = det_df.loc[(det_df.tilt == str(tilt))]
                tilt_df = tilt_df.reset_index()
                idxmax = tilt_df.ql_mean.idxmax()
                max_col = tilt_df.iloc[idxmax]
                print '{:^5} {:>8}'.format(max_col.tilt, max_col.ql_mean)

    if print_max_ql:
        get_max_ql_per_det(det_data, dets, tilts)

    if pulse_shape:
        print '\nANALYZING PULSE SHAPE DATA'
        # pulse shape uncertainties, from daq2:/home/radians/raweldon/tunl.2018.1_analysis/peak_localization/pulse_shape_get_hotspots.py
        if beam_11MeV:
            ps_unc = (0.0017, 0.0014, 0.0013, 0.0012, 0.0013, 0.0019, 0.0019, 0.0013, 0.0012, 0.0012, 0.0013, 0.0018)
        else:
            ps_unc = (0.0015, 0.0015, 0.0015, 0.0015,  0.002, 0.0036, 0.0036, 0.0021, 0.0016, 0.0015, 0.0015, 0.0015)
    elif delayed:
        print '\nANALYZING QS'
    elif prompt:
        print '\nANALYZING QP'
    else:
        print '\nANALYZING QL'

    fig_no = [1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1]
    a_label = ['a-axis', '', 'a-axis', '', 'a-axis', '', 'a-axis', '', 'a-axis', '', 'a-axis', '', ]
    min_ql_label = ['expected min', '', 'expected min', '', 'expected min', '', 'expected min', '', 'expected min', '', 'expected min', '', ]
    color = ['r', 'r', 'r', 'r', 'r', 'r', 'b', 'b', 'b', 'b', 'b', 'b']
    a_axis_dir = [20, 30, 40, 50, 60, 70, 110, 120, 130, 140, 150, 160] # angle for recoils along a-axis (relative)
    cp_b_axes_dir = [x + 90 for x in a_axis_dir[:6]] + [x - 90 for x in a_axis_dir[6:]] # angles for recoils along c' or b axes (only for tilt=0)

    sin_params, a_axis_data, cp_b_axes_data = [], [], []
    sin_params.append(['tilt', 'det', 'a', 'b', 'phi'])
    a_axis_data.append(['crystal', 'energy', 'det', 'tilt', 'ql', 'abs_uncert', 'fit_ql', 'counts'])
    for tilt in tilts:
        tilt_df = det_data.loc[(det_data.tilt == str(tilt))]

        if show_plots:
            print ''

        for d, det in enumerate(dets):
            det_df = tilt_df[(tilt_df.det_no == str(det))]
            det_df, angles = order_by_rot(det_df, beam_11MeV)

            if pulse_shape:
                # add pulse shape values and uncert to dataframe
                ql = det_df.ql_mean.values
                qs = det_df.qs_mean.values
                ps = 1 - qs/ql
                ps_uncert = [ps_unc[d]]*len(ql) # from daq2:/home/radians/raweldon/tunl.2018.1_analysis/peak_localization/pulse_shape_get_hotspots.py
                det_df = det_df.assign(ps = ps)
                det_df = det_df.assign(ps_uncert = ps_uncert)

                data = det_df.ps
                data_uncert = det_df.ps_uncert
                counts = det_df.counts

            elif delayed:
                data = det_df.qs_mean
                data_uncert = det_df.qs_abs_uncert
                counts = det_df.counts
            
            elif prompt:
                ql = det_df.ql_mean.values
                qs = det_df.qs_mean.values
                q_prompt = ql - qs
                det_df = det_df.assign(qp = q_prompt)            

                ql_uncert = det_df.ql_abs_uncert.values
                qs_uncert = det_df.qs_abs_uncert.values
                qp_uncert = np.sqrt(ql_uncert**2 + qs_uncert**2)  
                det_df = det_df.assign(qp_uncert = qp_uncert)

                data = det_df.qp
                data_uncert = det_df.qp_uncert
                counts = det_df.counts

            else:
                data = det_df.ql_mean
                print len(data)
                data_uncert = det_df.ql_abs_uncert
                counts = det_df.counts

            # fit
            res = fit_tilt_data(data.values, angles, print_report=False)
            pars = res.best_values
            x_vals = np.linspace(0, 190, 100)
            x_vals_rad = np.deg2rad(x_vals)
            y_vals = sin_func(x_vals_rad, pars['a'], pars['b'], pars['phi'])
            sin_params.append([tilt, det, pars['a'], pars['b'], pars['phi']])

            name = re.split('\.|_', det_df.filename.iloc[0])  
            # get lo along a, b, c' axes
            if a_axis_dir[d] in angles:
                a_axis_ql = data.iloc[np.where(angles == a_axis_dir[d])].values[0]
                #a_axis_ql = max(data.values)
                a_axis_data.append([name[0], name[1], name[4], tilt, a_axis_ql, data_uncert.iloc[np.where(angles == a_axis_dir[d])].values[0], 
                                   y_vals.max(), counts.iloc[np.where(angles == a_axis_dir[d])].values[0]])
            if cp_b_axes_dir[d] in angles:
                cp_b_ql = data.iloc[np.where(angles == cp_b_axes_dir[d])].values[0]
                cp_b_axes_data.append([name[0], name[1], name[4], tilt, cp_b_ql, data_uncert.iloc[np.where(angles == cp_b_axes_dir[d])].values[0], 
                                       y_vals.min(), counts.iloc[np.where(angles == cp_b_axes_dir[d])].values[0]])

            # plot same det angles together
            if show_plots:
                
                plt.figure(fig_no[d], figsize=(10,8))
                plt.errorbar(angles, data, yerr=data_uncert.values, ecolor='black', markerfacecolor=color[d], fmt='o', 
                            markeredgecolor='k', markeredgewidth=1, markersize=10, capsize=1, label='det ' + str(det))
                
                # annotate
                for rot, ang, t in zip(det_df.rotation, angles, data):
                    plt.annotate( str(rot) + '$^{\circ}$', xy=(ang, t), xytext=(-3, 10), textcoords='offset points')
               
                # plot fit
                plt.plot(x_vals, y_vals, '--', color=color[d])
                
                # check a-axis recoils have max ql - 11 and 4 MeV look good
                plt.scatter(angles[np.where(angles == a_axis_dir[d])], data.iloc[np.where(angles == a_axis_dir[d])], c='k', s=120, zorder=10, label=a_label[d])
                plt.scatter(angles[np.where(angles == cp_b_axes_dir[d])], data.iloc[np.where(angles == cp_b_axes_dir[d])], c='g', s=120, zorder=10, label=min_ql_label[d])
                                            
                plt.xlim(-5, max(angles)+5)
                if pulse_shape:
                    plt.ylabel('pulse shape parameter')
                else:
                    plt.ylabel('light output (MeVee)')
                plt.xlabel('rotation angle (degree)')
                name = name[0] + '_' + name[1] + '_' + name[2] + '_' + name[4]
                print name
                plt.title(name)
                plt.legend(fontsize=10)
                if save_plots:
                    if d > 5:
                        if pulse_shape:
                            plt.savefig(cwd + '/figures/tilt_plots/pulse_shape/' + name + '_pulse_shape.png')
                        else:
                            plt.savefig(cwd + '/figures/tilt_plots/' + name + '.png')
                            print 'plots saved to /figures/tilt_plots/' + name + '.png'

        if show_plots:
            plt.show()

    # save sinusoid fit to pickle
    if save_pickle:
        name = pickle_name.split('.')[0]
        out = open( p_dir + name + '_sin_params.p', "wb" )
        pickle.dump( sin_params, out)
        out.close()
        print 'pickle saved to ' + p_dir + name + '_sin_params.p'

    if get_a_data:
        headers = a_axis_data.pop(0)
        #pickle.dump( a_axis_data, open( p_dir + 'a_axis_data.p', 'wb'))
        return pd.DataFrame(a_axis_data, columns=headers), pd.DataFrame(cp_b_axes_data, columns=headers)

def smoothing_tilt(dets, pickle_name, cwd, p_dir, pulse_shape, delayed, prompt, show_plots, save_plots, save_pickle):
    ''' Smoothing assumes BL measurements are more precise than BR (change in anisotropy is greater, BR dets were most likely below the plane of measurement)
            and that the cpvert crystals lower light output at 11 MeV is due to poor calibration
        Code taken from tilt_check function
        Normalizes all measurements to max light output measured for a given energy, centers max at a-axis direction
    '''

    def fit_tilt_data(data, angles, print_report):
            # sinusoid fit with lmfit
            angles = [np.deg2rad(x) for x in angles]
            gmodel = lmfit.Model(sin_func)
            params = gmodel.make_params(a=1, b=1, phi=0.1)
            params['phi'].max = np.deg2rad(90)
            params['phi'].min = np.deg2rad(-90)
            res = gmodel.fit(data, params=params, x=angles, nan_policy='omit')#, method='nelder')
            if print_report:
                print '\n', lmfit.fit_report(res)
            return res

    for f in pickle_name:
        if '11' in f:
            beam_11MeV = True
            # from /home/radians/raweldon/tunl.2018.1_analysis/stilbene_final/lo_calibration/uncert_gamma_cal.py 
            cal_unc = [0.00792, 0.00540, 0.00285, 0.00155, 0.00271, 0.00372, 0.00375, 0.00275, 0.00156, 0.00278, 0.00540, 0.00800]
        else:
            beam_11MeV = False
            cal_unc = [0.01920, 0.01502, 0.01013, 0.00541, 0.00176, 0.00116, 0.00116, 0.00185, 0.00552, 0.01025, 0.01506, 0.01935]
        if 'bvert' in f:
            max_ql = []
            tilts = [0, 45, -45, 30, -30, 15, -15]
        else:
            tilts = [0, 30, -30, 15, -15]

        data = pd_load(f, p_dir)
        det_data = split_filenames(data)

        if pulse_shape:
            print '\nANALYZING PULSE SHAPE DATA'
            # pulse shape uncertainties, from daq2:/home/radians/raweldon/tunl.2018.1_analysis/stilbene_final/peak_localization/pulse_shape_get_hotspots.py
            if beam_11MeV:
                ps_unc = [0.00175, 0.00135, 0.00125, 0.00126, 0.0014, 0.00198, 0.00195, 0.0014, 0.00124, 0.00123, 0.00134, 0.00177]
            else:
                ps_unc = [0.00164, 0.00142, 0.00142, 0.00147, 0.0018, 0.00306, 0.0031, 0.00179, 0.00143, 0.00142, 0.00142, 0.0016]
        elif delayed:
            print '\nANALYZING QS'
        elif prompt:
            print '\nANALYZING QP'
        else:
            print '\nANALYZING QL'

        fig_no = [1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1]
        a_label = ['a-axis', '', 'a-axis', '', 'a-axis', '', 'a-axis', '', 'a-axis', '', 'a-axis', '', ]
        min_ql_label = ['expected min', '', 'expected min', '', 'expected min', '', 'expected min', '', 'expected min', '', 'expected min', '', ]
        color = cm.viridis(np.linspace(0, 1, len(tilts)))
        a_axis_dir = [20, 30, 40, 50, 60, 70, 110, 120, 130, 140, 150, 160] # angle for recoils along a-axis (relative)
        cp_b_axes_dir = [x + 90 for x in a_axis_dir[:6]] + [x - 90 for x in a_axis_dir[6:]] # angles for recoils along c' or b axes (only for tilt=0)

        sin_params, a_axis_data, cp_b_axes_data = [], [], []
        sin_params.append(['tilt', 'det', 'a', 'b', 'phi'])
        a_axis_data.append(['crystal', 'energy', 'det', 'tilt', 'ql', 'abs_uncert', 'fit_ql', 'counts'])
        for tidx, tilt in enumerate(tilts):
            tilt_df = det_data.loc[(det_data.tilt == str(tilt))]

            if show_plots:
                print ''

            for d, det in enumerate(dets):
                det_df = tilt_df[(tilt_df.det_no == str(det))]
                det_df, angles = order_by_rot(det_df, beam_11MeV)

                if pulse_shape:
                    # add pulse shape values and uncert to dataframe
                    ql = det_df.ql_mean.values
                    qs = det_df.qs_mean.values
                    ps = 1 - qs/ql
                    ps_uncert = [ps_unc[d]]*len(ql) # from daq2:/home/radians/raweldon/tunl.2018.1_analysis/peak_localization/pulse_shape_get_hotspots.py
                    det_df = det_df.assign(ps = ps)
                    det_df = det_df.assign(ps_uncert = ps_uncert)

                    data = det_df.ps
                    data_uncert = det_df.ps_uncert
                    counts = det_df.counts

                elif delayed:
                    data = det_df.qs_mean
                    data_uncert = det_df.qs_abs_uncert
                    counts = det_df.counts
                
                elif prompt:
                    ql = det_df.ql_mean.values
                    qs = det_df.qs_mean.values
                    q_prompt = ql - qs
                    det_df = det_df.assign(qp = q_prompt)            

                    ql_uncert = det_df.ql_abs_uncert.values
                    qs_uncert = det_df.qs_abs_uncert.values
                    qp_uncert = np.sqrt(ql_uncert**2 + qs_uncert**2)  
                    det_df = det_df.assign(qp_uncert = qp_uncert)

                    data = det_df.qp
                    data_uncert = det_df.qp_uncert
                    counts = det_df.counts

                else:
                    data = det_df.ql_mean
                    data_uncert = det_df.ql_abs_uncert
                    counts = det_df.counts
              
                #
                # smoothing and averaging 
                #
                if d > 5:
                    continue
                else:
                    name = re.split('\.|_', det_df.filename.iloc[0]) 
                    name = name[0] + '_' + name[1] + '_' + name[2] + '_' + name[4]
                    print name
                    
                    # fit
                    res = fit_tilt_data(data.values, angles, print_report=False)
                    pars = res.best_values
                    x_vals = np.linspace(0, 190, 1000)
                    x_orig = np.linspace(0, 190, 1000)
                    x_vals_rad = np.deg2rad(x_vals)
                    y_vals = sin_func(x_vals_rad, pars['a'], pars['b'], pars['phi'])
                    y_orig = y_vals

                    ## use to get scaling factor for smoothing
                    #if tilt == 0 and 'bvert' in f:
                    #    max_ql.append(max(y_vals))
                    #    print max_ql
                    #else:
                    #    if max(y_vals) > max_ql[d]:
                    #        max_ql[d] = max(data)
                    #        print max_ql

                    if beam_11MeV:
                        if pulse_shape:
                            max_ql = [0.3211530198583945, 0.3310542105630364, 0.34502415458937197, 0.3687198259854635, 0.3996132774734128, 0.4287629902782434]
                            lo_unc = [np.sqrt((cal_unc[d]*q)**2 + ps_unc[d]**2 + (0.005*q)**2) for q in data.values]
                        else:
                            max_ql = [5.361060691111871, 4.321438060046981, 3.087094177473607, 1.902, 0.9435, 0.3169] # calulcated directly above (uncomment, copy from output)
                            lo_unc = [np.sqrt(cal_unc[d]**2 + u**2 + (0.005*q)**2) for u, q in zip(data_uncert.values, data.values)]
                    else:
                        if pulse_shape:
                            max_ql = [0.36916873449131526, 0.37724319306930687, 0.39375000000000004, 0.40695108495770493, 0.4193293885601578, 0.38900203665987776]
                            lo_unc = [np.sqrt((cal_unc[d]*q)**2 + ps_unc[d]**2 + (0.005*q)**2) for q in data.values]

                        else:
                            max_ql = [1.6429, 1.3094, 0.9084, 0.5553, 0.2733, 0.1042]
                            lo_unc = [np.sqrt(cal_unc[d]**2 + u**2 + (0.005*q)**2) for u, q in zip(data_uncert.values, data.values)]

                    c = max(y_vals)/max_ql[d]
                    smoothed_data = data/c
                    y_vals = y_vals/c
                    res = fit_tilt_data(smoothed_data.values, angles, print_report=False)
                    pars = res.best_values
                    #print max_ql[d], max(y_vals), data.iloc[np.where(angles == a_axis_dir[d])].values

                    # shift max y_val to a-axis direction (all sinusoid maxes are aligned)
                    max_yidx = np.argmax(y_vals)
                    diff = x_vals[max_yidx] - a_axis_dir[d]
                    x_vals -= diff
                    res = fit_tilt_data(y_vals, x_vals, print_report=False)
                    pars = res.best_values
                    #print y_vals[max_yidx], x_vals[max_yidx], a_axis_dir[d], diff

                    sin_params.append([tilt, det, pars['a'], pars['b'], pars['phi']])

                    if show_plots:
                        # smoothed data plots
                        plt.figure(fig_no[d] + 10)
                        plt.errorbar(angles, smoothed_data, yerr=lo_unc, ecolor='black', markerfacecolor=color[tidx], fmt='o', 
                                    markeredgecolor='k', markeredgewidth=1, markersize=10, capsize=1, label=str(tilt) + ' deg tilt')
                        plt.plot(x_vals, y_vals, '--', color=color[tidx])
                        # annotate
                        for rot, ang, t in zip(det_df.rotation, angles, smoothed_data):
                            plt.annotate( str(rot) + '$^{\circ}$', xy=(ang, t), xytext=(-3, 10), textcoords='offset points')
                                            
                        plt.xlim(-5, 200)
                        if pulse_shape:
                            plt.ylabel('pulse shape parameter')
                        else:
                            plt.ylabel('light output (MeVee)')
                        plt.xlabel('rotation angle (degree)')
                        plt.title(det)
                        plt.legend(fontsize=10)

                        # original data plots
                        plt.figure(fig_no[d])
                        plt.errorbar(angles, data, yerr=lo_unc, ecolor='black', markerfacecolor=color[tidx], fmt='o', 
                                    markeredgecolor='k', markeredgewidth=1, markersize=10, capsize=1, label=str(tilt) + ' deg tilt')
                        plt.plot(x_orig, y_orig, '--', color=color[tidx])
                        # annotate
                        for rot, ang, t in zip(det_df.rotation, angles, data):
                            plt.annotate( str(rot) + '$^{\circ}$', xy=(ang, t), xytext=(-3, 10), textcoords='offset points')
                                            
                        plt.xlim(-5, 200)
                        if pulse_shape:
                            plt.ylabel('pulse shape parameter')
                        else:
                            plt.ylabel('light output (MeVee)')
                        plt.xlabel('rotation angle (degree)')
                        plt.title(det)
                        plt.legend(fontsize=10)
                    
        if show_plots:
            plt.show()

        #print '\n\nFinal max vals:\n', max_ql
        # save sinusoid fit to pickle
        if save_pickle:
            name = f.split('.')[0]
            if pulse_shape:
                out = open( p_dir + name + '_sin_params_smoothed_ps.p', "wb" )
            else:
                out = open( p_dir + name + '_sin_params_smoothed.p', "wb" )
            pickle.dump( sin_params, out)
            out.close()
            print 'pickle saved to ' + p_dir + name + '_sin_params_smoothed.p'

def polar_to_cartesian(theta, phi, crystal_orientation, cp_up):
    # cp_up
    if np.array_equal(crystal_orientation, cp_up):
        #print 'Converting to cartesian with cp_up'
        x = np.sin(theta)*np.cos(phi)
        y = np.sin(theta)*np.sin(phi)
        z = np.cos(theta)
    # b_up
    else:
        #print 'Converting to cartesian with b_up'
        x = -np.sin(theta)*np.cos(phi)
        y = -np.sin(theta)*np.sin(phi)
        z = np.cos(theta)       
    return x, y , z

def crystal_basis_vectors(angle, axis_up):
    # Rotation is counterclockwise about the a-axis (x-axis)
    theta = np.deg2rad(angle)
    rot_matrix_x = np.asarray(((1,0,0),(0,np.cos(theta),np.sin(theta)),(0,-np.sin(theta),np.cos(theta))))
    rotated_axes = np.transpose(np.dot(rot_matrix_x,np.transpose(axis_up)))  
    return rotated_axes

def map_3d(tilt, crystal_orientation, angles, theta_neutron, phi_neutron):
    # map points to sphere surface using rotation angle (angles), neutron scatter angle, and tilt angle

    #print '\n', tilt
    basis_vectors = crystal_basis_vectors(tilt, crystal_orientation)
    thetap, phip = [], []
    for angle in angles:
        # code from C:\Users\raweldon\Research\TUNL\crystal_orientation\crystal_orientations_3d_plot_v8.py
        angle = np.deg2rad(angle)
        # counterclockwise rotation matrix about y
        rot_matrix_y = np.asarray(((np.cos(angle), 0, -np.sin(angle)), (0, 1, 0), (np.sin(angle), 0, np.cos(angle))))
        rot_orientation = np.transpose(np.dot(rot_matrix_y, np.transpose(basis_vectors)))
    
        # proton recoil
        theta_proton = np.deg2rad(theta_neutron) # proton recoils at 90 deg relative to theta_neutron
        phi_proton = np.deg2rad(phi_neutron  + 180) # phi_proton will be opposite sign of phi_neutron
    
        # cartesian vector    
        x_p = np.sin(theta_proton)*np.cos(phi_proton)
        y_p = np.sin(theta_proton)*np.sin(phi_proton)    
        z_p = np.cos(theta_proton)
        
        p_vector = np.asarray((x_p, y_p, z_p))
        
        #get theta_p
        p_vector_dot_cp = np.dot(p_vector,rot_orientation[2]) # a.b=||a||*||b|| cos(theta)
        theta_p = np.rad2deg(np.arccos(p_vector_dot_cp))
        
        # get phi_p
        vector_proj_ab = p_vector - p_vector_dot_cp*rot_orientation[2] # remove c' to get ab plane projection (scalar proj: a1 = a*cos(theta))
        vector_proj_ab = vector_proj_ab/np.sqrt(np.dot(vector_proj_ab,vector_proj_ab)) # vector proj: a1^ = (|a^|cos(theta)) b^/|b|
        vector_proj_dot_a = np.dot(vector_proj_ab,rot_orientation[0])
        
        # account for rounding errors on 1.0 and -1.0 (ex: 1.00000002 -> phi_p = nan)
        if abs(vector_proj_dot_a) > 1.0:
            if vector_proj_dot_a > 0:
                vector_proj_dot_a = 1.0
            else:
                vector_proj_dot_a = -1.0
    
        phi_p = np.rad2deg(np.arccos(vector_proj_dot_a))
        
        # check if phi > 180 deg
        if np.dot(vector_proj_ab, rot_orientation[1]) < 0:
            phi_p = 360 - phi_p

        thetap.append(np.deg2rad(theta_p))
        phip.append(np.deg2rad(phi_p)) 

    return thetap, phip   

def map_data_3d(data, det, tilts, crystal_orientation, theta_neutron, phi_neutron, beam_11MeV):
    # calculates proton recoil trajectory in correct frame using crystal tilt and neutron scatter angles 

    det_df = data.loc[(data.det_no == str(det))]
    dfs=[]
    for t, tilt in enumerate(tilts):
        tilt_df = det_df.loc[(data.tilt == str(tilt))]
        tilt_df, angles = order_by_rot(tilt_df, beam_11MeV)

        thetap, phip = map_3d(tilt, crystal_orientation, angles, theta_neutron, phi_neutron)       

        update_df = tilt_df.assign(phi = phip)
        update_df = update_df.assign(theta = thetap)
        update_df = update_df.assign(rot_angles = angles)
        dfs.append(update_df)
    return pd.concat(dfs)

def scatter_check_3d(fin1, fin2, dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, beam_11MeV, matplotlib):
    ''' Scatter plots of proton recoil trajectories (matplotlib)
        Colors recoils from bvert and cpvert crytals
    '''
    data_bvert = pd_load(fin1, p_dir)
    data_bvert = split_filenames(data_bvert)
    data_cpvert = pd_load(fin2, p_dir)
    data_cpvert = split_filenames(data_cpvert)

    for d, det in enumerate(dets):       
        df_b_mapped = map_data_3d(data_bvert, det, bvert_tilt, b_up, theta_n[d], phi_n[d], beam_11MeV) 
        df_cp_mapped = map_data_3d(data_cpvert, det, cpvert_tilt, cp_up, theta_n[d], phi_n[d], beam_11MeV)

        # convert to cartesian
        theta1 = df_b_mapped.theta.values
        phi1 = df_b_mapped.phi.values
        theta2 = df_cp_mapped.theta.values
        phi2 = df_cp_mapped.phi.values
        angles1 = df_b_mapped.rot_angles.values
        angles2 = df_cp_mapped.rot_angles.values
        x1, y1, z1 = polar_to_cartesian(theta1, phi1, b_up, cp_up)
        x2, y2, z2 = polar_to_cartesian(theta2, phi2, cp_up, cp_up)

        if matplotlib:
            fig = plt.figure(det)
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(x1, y1, z1, c='r', label='bvert')
            ax.scatter(x2, y2, z2, c='b', label='cpvert')
            ax.set_xlim([-1.1,1.1])
            ax.set_ylim([-1.1,1.1])
            ax.set_zlim([-1.1,1.1])
            ax.set_xlabel('a')
            ax.set_ylabel('b')
            ax.set_zlabel('c\'')
            ax.set_aspect('equal')
            if beam_11MeV:
                ax.set_title('11.3 MeV beam, det ' + str(det) + '  (' + str(theta_n[d]) +'$^{\circ}$)')
            else:
                ax.set_title('4.8 MeV beam, det ' + str(det) + '  (' + str(theta_n[d]) +'$^{\circ}$)')
            plt.tight_layout()
            plt.legend()
            plt.show()

        else:
            fig = mlab.figure(size=(400*2, 350*2)) 
            pts = mlab.points3d(x1, y1, z1, color=(1,0,0), scale_mode='none', scale_factor=0.03)
            mlab.points3d(x2, y2, z2, color=(0,0,1), scale_mode='none', scale_factor=0.03)
            
            for x_val, y_val, z_val, angle in zip(x1, y1, z1, angles1):
                mlab.text3d(x_val, y_val, z_val, str(angle), scale=0.03, color=(0,0,0), figure=fig)
            for x_val, y_val, z_val, angle in zip(x2, y2, z2, angles2):
                mlab.text3d(x_val, y_val, z_val, str(angle), scale=0.03, color=(0,0,0), figure=fig)

            mlab.axes(pts, xlabel='a', ylabel='b', zlabel='c\'')
            mlab.view(azimuth=0, elevation=90, distance=7.5, figure=fig)  
            mlab.title('det ' + str(det))
            mlab.show()

def heatmap_multiplot(x, y, z, data, theta_n, d, cwd, save_multiplot, beam_11MeV, fitted):
    #            a   b  c'  nice 
    plot_angle = 70
    if beam_11MeV:
        beam = '11MeV'
    else:
        beam = '4MeV'
    if theta_n[d] == plot_angle:
        azimuth =   [180,  90, 0, 205]
        elevation = [90, 90, 0, 75]
        if fitted:
            names = [str(plot_angle) + 'deg_' + beam + '_a_fitted', str(plot_angle) + 'deg_' + beam + '_b_fitted', str(plot_angle) + 
                     'deg_' + beam + '_c\'_fitted', str(plot_angle) + 'deg_' + beam + '_nice_fitted']
        else:
            names = [str(plot_angle) + 'deg_' + beam + '_a', str(plot_angle) + 'deg_' + beam + '_b', str(plot_angle) + 'deg_' + beam + '_c\'', str(plot_angle) + 'deg_' + beam + '_nice']

        max_idx = np.argmax(data)
        print ' max = ', data[max_idx]
        for i, (az, el) in enumerate(zip(azimuth, elevation)):
            fig = mlab.figure(size=(400*2, 350*2)) 
            pts = mlab.points3d(x, y, z, data, colormap='viridis', scale_mode='none', scale_factor=0.03)
            if fitted:
                mlab.points3d(x[max_idx], y[max_idx], z[max_idx], data[max_idx], color=(1,0,0), scale_mode='none', scale_factor=0.03)
            tri = mlab.pipeline.delaunay3d(pts)
            tri_smooth = mlab.pipeline.poly_data_normals(tri) # smooths delaunay triangulation mesh
            surf = mlab.pipeline.surface(tri_smooth, colormap='viridis')
            mlab.axes(pts, xlabel='a', ylabel='b', zlabel='c\'')
            mlab.colorbar(surf, orientation='vertical') 
            mlab.view(azimuth=az, elevation=el, distance=7.5, figure=fig)
            if save_multiplot:
                print theta_n[d], names[i]
                mlab.savefig(cwd + '/' + names[i] + '.png')
                print 'saved to' + cwd + '/' + names[i] + '.png'
                mlab.clf()
                mlab.close()      

def heatmap_singleplot(x, y, z, data, tilts, name, show_delaunay):
    max_idx = np.argmax(data)
    print name + ' max = ', round(data[max_idx], 4)

    fig = mlab.figure(size=(400*2, 350*2)) 
    fig.scene.disable_render = True
    pts = mlab.points3d(x, y, z, data, colormap='viridis', scale_mode='none', scale_factor=0.03)

    # plot max ql point (red)
    mlab.points3d(x[max_idx], y[max_idx], z[max_idx], data[max_idx], color=(1,0,0), scale_mode='none', scale_factor=0.03)
    
    # delaunay triagulation (mesh, interpolation)
    tri = mlab.pipeline.delaunay3d(pts)
    edges = mlab.pipeline.extract_edges(tri)
    if show_delaunay:
        edges = mlab.pipeline.surface(edges, colormap='viridis')

    tri_smooth = mlab.pipeline.poly_data_normals(tri) # smooths delaunay triangulation mesh
    surf = mlab.pipeline.surface(tri_smooth, colormap='viridis')
    
    for x_val, y_val, z_val, ql_val, tilt in zip(x, y, z, data, tilts):
        #mlab.text3d(x_val, y_val, z_val, str(ql_val), scale=0.03, color=(0,0,0), figure=fig)
        mlab.text3d(x_val, y_val, z_val, str(tilt), scale=0.03, color=(0,0,0), figure=fig) 

    mlab.axes(pts, xlabel='a', ylabel='b', zlabel='c\'')
    mlab.colorbar(surf, orientation='vertical') 
    mlab.view(azimuth=0, elevation=90, distance=7.5, figure=fig)  
    mlab.title(name)
    fig.scene.disable_render = False          

def plot_heatmaps(fin1, fin2, dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, beam_11MeV, plot_pulse_shape, multiplot, save_multiplot, show_delaunay):
    ''' Plots the heatmap for a hemishpere of measurements
        Full sphere is made by plotting a mirror image of the hemiphere measurements
    '''
    data_bvert = pd_load(fin1, p_dir)
    data_bvert = split_filenames(data_bvert)
    data_cpvert = pd_load(fin2, p_dir)
    data_cpvert = split_filenames(data_cpvert)

    for d, det in enumerate(dets):       
        print '\ndet_no =', det, 'theta_n =', theta_n[d]
        df_b_mapped = map_data_3d(data_bvert, det, bvert_tilt, b_up, theta_n[d], phi_n[d], beam_11MeV) 
        df_b_mapped_mirror = map_data_3d(data_bvert, det, bvert_tilt, np.asarray(((1,0,0), (0,1,0), (0,0,-1))), theta_n[d], phi_n[d], beam_11MeV)
        df_cp_mapped = map_data_3d(data_cpvert, det, cpvert_tilt, cp_up, theta_n[d], phi_n[d], beam_11MeV)
        df_cp_mapped_mirror = map_data_3d(data_cpvert, det, cpvert_tilt, np.asarray(((-1,0,0), (0,0,-1), (0,1,0))), theta_n[d], phi_n[d], beam_11MeV)

        # convert to cartesian
        theta_b = np.concatenate([df_b_mapped.theta.values, df_b_mapped_mirror.theta.values])
        theta_cp = np.concatenate([df_cp_mapped.theta.values, df_cp_mapped_mirror.theta.values])
        phi_b = np.concatenate([df_b_mapped.phi.values, df_b_mapped_mirror.phi.values])
        phi_cp = np.concatenate([df_cp_mapped.phi.values, df_cp_mapped_mirror.phi.values])

        x_b, y_b, z_b = polar_to_cartesian(theta_b, phi_b, b_up, cp_up)
        x_cp, y_cp, z_cp = polar_to_cartesian(theta_cp, phi_cp, cp_up, cp_up)

        x = np.round(np.concatenate((x_b, x_cp)), 12)
        y = np.round(np.concatenate((y_b, y_cp)), 12)
        z = np.round(np.concatenate((z_b, z_cp)), 12)
        ql = np.concatenate([df_b_mapped.ql_mean.values, df_b_mapped_mirror.ql_mean.values, df_cp_mapped.ql_mean.values, df_cp_mapped_mirror.ql_mean.values])
        tilts = np.concatenate([df_b_mapped.tilt.values, df_b_mapped_mirror.tilt.values, df_cp_mapped.tilt.values, df_cp_mapped_mirror.tilt.values])

        # remove repeated points
        xyz = np.array(zip(x, y, z))
        xyz_u, indices = np.unique(xyz, axis=0, return_index=True)
        ql = ql[indices]
        tilts = tilts[indices]
        x, y, z = xyz_u.T
        if plot_pulse_shape:
            qs = np.concatenate([df_b_mapped.qs_mean.values, df_b_mapped_mirror.qs_mean.values, df_cp_mapped.qs_mean.values, df_cp_mapped_mirror.qs_mean.values])
            qs = qs[indices]
            ps = [1 - a/b for a, b in zip(qs, ql)]

        # points3d with delaunay filter - works best!!
        ## use for nice looking plots
        if multiplot:
            heatmap_multiplot(x, y, z, ql, theta_n, d, cwd + '/figures/3d_lo', save_multiplot, beam_11MeV, fitted=False)
            if plot_pulse_shape:
                heatmap_multiplot(x, y, z, ps, theta_n, d, cwd + '/figures/3d_pulse_shape', save_multiplot, beam_11MeV, fitted=False)
            if not save_multiplot:
                mlab.show()

        ## use for analysis 
        else:
            heatmap_singleplot(x, y, z, ql, tilts, 'ql', show_delaunay)
            if plot_pulse_shape:
                heatmap_singleplot(x, y, z, ps, tilts, 'qs/ql', show_delaunay)
            mlab.show()

def map_fitted_data_3d(data, det, tilts, crystal_orientation, theta_neutron, phi_neutron, beam_11MeV):
    # like map_data_3d but for fitted data
    det_df = data.loc[(data.det == det)]
    ql_all, theta_p, phi_p = [], [], []
    for t, tilt in enumerate(tilts):
        tilt_df = det_df.loc[(data.tilt == tilt)]
        angles = np.arange(0, 180, 5) # 5 and 2 look good

        #print '{:>4} {:>8} {:>8} {:>8}'.format(tilt, round(tilt_df['a'].values[0], 4), round(tilt_df['b'].values[0], 4), round(tilt_df['phi'].values[0],4))

        ql = sin_func(np.deg2rad(angles), tilt_df['a'].values, tilt_df['b'].values, tilt_df['phi'].values)
        thetap, phip = map_3d(tilt, crystal_orientation, angles, theta_neutron, phi_neutron)       
        ql_all.extend(ql)
        theta_p.extend(thetap)
        phi_p.extend(phip)

    d = {'ql': ql_all, 'theta': theta_p, 'phi': phi_p}
    df = pd.DataFrame(data=d)
    return df

def plot_fitted_heatmaps(fin1, fin2, dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, beam_11MeV, multiplot, save_multiplot):
    data_bvert = pd_load(fin1, p_dir)
    data_cpvert = pd_load(fin2, p_dir)

    for d, det in enumerate(dets):
        print 'det_no =', det, 'theta_n =', theta_n[d]
        df_b_mapped = map_fitted_data_3d(data_bvert, det, bvert_tilt, b_up, theta_n[d], phi_n[d], beam_11MeV)
        df_cp_mapped = map_fitted_data_3d(data_cpvert, det, cpvert_tilt, cp_up, theta_n[d], phi_n[d], beam_11MeV)
        df_b_mapped_mirror = map_fitted_data_3d(data_bvert, det, bvert_tilt, np.asarray(((1,0,0), (0,1,0), (0,0,-1))), theta_n[d], phi_n[d], beam_11MeV)
        df_cp_mapped_mirror = map_fitted_data_3d(data_cpvert, det, cpvert_tilt, np.asarray(((-1,0,0), (0,0,-1), (0,1,0))), theta_n[d], phi_n[d], beam_11MeV)

        # convert to cartesian
        theta_b = np.concatenate([df_b_mapped.theta.values, df_b_mapped_mirror.theta.values])
        theta_cp = np.concatenate([df_cp_mapped.theta.values, df_cp_mapped_mirror.theta.values])
        phi_b = np.concatenate([df_b_mapped.phi.values, df_b_mapped_mirror.phi.values])
        phi_cp = np.concatenate([df_cp_mapped.phi.values, df_cp_mapped_mirror.phi.values])

        x_b, y_b, z_b = polar_to_cartesian(theta_b, phi_b, b_up, cp_up)
        x_cp, y_cp, z_cp = polar_to_cartesian(theta_cp, phi_cp, cp_up, cp_up)

        x = np.round(np.concatenate((x_b, x_cp)), 12)
        y = np.round(np.concatenate((y_b, y_cp)), 12)
        z = np.round(np.concatenate((z_b, z_cp)), 12)
        ql = np.concatenate([df_b_mapped.ql.values, df_b_mapped_mirror.ql.values, df_cp_mapped.ql.values, df_cp_mapped_mirror.ql.values])

        # remove repeated points
        xyz = np.array(zip(x,y,z))
        xyz, indices = np.unique(xyz, axis=0, return_index=True)
        ql = ql[indices]
        x, y, z = xyz.T

        if multiplot:
            heatmap_multiplot(x, y, z, ql, theta_n, d, cwd, save_multiplot, beam_11MeV, fitted=True)
            if save_multiplot:
                continue
            mlab.show()

        else:
            max_idx = np.argmax(ql)
            print 'ql_max = ', ql[max_idx], '\n'
            fig = mlab.figure(size=(400*2, 350*2)) 
            pts = mlab.points3d(x, y, z, ql, colormap='viridis', scale_mode='none', scale_factor=0.03)
            # plot max ql point (red)
            mlab.points3d(x[max_idx], y[max_idx], z[max_idx], ql[max_idx], color=(1,0,0), scale_mode='none', scale_factor=0.03)
            # delaunay triagulation (mesh, interpolation)
            tri = mlab.pipeline.delaunay3d(pts)
            tri_smooth = mlab.pipeline.poly_data_normals(tri) # smooths delaunay triangulation mesh
            surf = mlab.pipeline.surface(tri_smooth, colormap='viridis')
            mlab.axes(pts, xlabel='a', ylabel='b', zlabel='c\'')
            mlab.colorbar(surf, orientation='vertical') 
            mlab.view(azimuth=0, elevation=-90, distance=7.5, figure=fig)            
            mlab.show()        

def plot_avg_heatmaps(fin1, fin2, dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, beam_11MeV, plot_pulse_shape, multiplot, save_multiplot, show_delaunay):
    ''' Plots the heatmap for a hemishpere of measurements
        Full sphere is made by plotting a mirror image of the hemiphere measurements
    '''
    data_bvert = pd_load(fin1, p_dir)
    data_bvert = split_filenames(data_bvert)
    data_cpvert = pd_load(fin2, p_dir)
    data_cpvert = split_filenames(data_cpvert)

    bl_br_vals = []
    order = [11, 10, 9, 8, 7, 6]

    for d, det in enumerate(dets):       
        df_b_mapped = map_data_3d(data_bvert, det, bvert_tilt, b_up, theta_n[d], phi_n[d], beam_11MeV) 
        df_b_mapped_mirror = map_data_3d(data_bvert, det, bvert_tilt, np.asarray(((1,0,0), (0,1,0), (0,0,-1))), theta_n[d], phi_n[d], beam_11MeV)
        df_cp_mapped = map_data_3d(data_cpvert, det, cpvert_tilt, cp_up, theta_n[d], phi_n[d], beam_11MeV)
        df_cp_mapped_mirror = map_data_3d(data_cpvert, det, cpvert_tilt, np.asarray(((-1,0,0), (0,0,-1), (0,1,0))), theta_n[d], phi_n[d], beam_11MeV)

        # convert to cartesian
        theta_b = np.concatenate([df_b_mapped.theta.values, df_b_mapped_mirror.theta.values])
        theta_cp = np.concatenate([df_cp_mapped.theta.values, df_cp_mapped_mirror.theta.values])
        phi_b = np.concatenate([df_b_mapped.phi.values, df_b_mapped_mirror.phi.values])
        phi_cp = np.concatenate([df_cp_mapped.phi.values, df_cp_mapped_mirror.phi.values])

        x_b, y_b, z_b = polar_to_cartesian(theta_b, phi_b, b_up, cp_up)
        x_cp, y_cp, z_cp = polar_to_cartesian(theta_cp, phi_cp, cp_up, cp_up)

        x = np.round(np.concatenate((x_b, x_cp)), 12)
        y = np.round(np.concatenate((y_b, y_cp)), 12)
        z = np.round(np.concatenate((z_b, z_cp)), 12)
        ql = np.concatenate([df_b_mapped.ql_mean.values, df_b_mapped_mirror.ql_mean.values, df_cp_mapped.ql_mean.values, df_cp_mapped_mirror.ql_mean.values])
        tilts = np.concatenate([df_b_mapped.tilt.values, df_b_mapped_mirror.tilt.values, df_cp_mapped.tilt.values, df_cp_mapped_mirror.tilt.values])
        # remove repeated points
        xyz = np.array(zip(x, y, z))
        xyz_u, indices = np.unique(xyz, axis=0, return_index=True)
        ql = ql[indices]
        tilts = tilts[indices]
        x, y, z = xyz_u.T

        if plot_pulse_shape:
            qs = np.concatenate([df_b_mapped.qs_mean.values, df_b_mapped_mirror.qs_mean.values, df_cp_mapped.qs_mean.values, df_cp_mapped_mirror.qs_mean.values])
            qs = qs[indices]
            ps = [1 - a/b for a, b in zip(qs, ql)]
            vals = zip(*sorted(zip(ql, ps, x, y, z, tilts)))
        else:
            vals = zip(*sorted(zip(ql, x, y, z, tilts)))

        bl_br_vals.append(vals)

    if plot_pulse_shape:
        df = pd.DataFrame(bl_br_vals, columns=['ql', 'ps', 'x', 'y', 'z', 'tilts'])
    else:
        df = pd.DataFrame(bl_br_vals, columns=['ql', 'x', 'y', 'z', 'tilts'])

    for i in xrange(0, 6):
        print '\ndet_no =', i + 4, 'theta_n =', theta_n[i]
        x = df.x.iloc[i]
        y = df.y.iloc[i]
        z = df.z.iloc[i]
        tilts = np.array(df.tilts.iloc[i])
        ql = np.array([(q + p)/2 for q, p in zip(df.ql.iloc[i], df.ql.iloc[order[i]])])

        # points3d with delaunay filter - works best!!
        ## use for nice looking plots
        if multiplot:
            heatmap_multiplot(x, y, z, ql, theta_n, d, cwd + '/figures/3d_lo', save_multiplot, beam_11MeV, fitted=False)
            if plot_pulse_shape:
                ps = np.array([(q + p)/2 for q, p in zip(df.ps.iloc[i], df.ps.iloc[order[i]])])
                heatmap_multiplot(x, y, z, ps, theta_n, d, cwd + '/figures/3d_pulse_shape', save_multiplot, beam_11MeV, fitted=False)
            if not save_multiplot:
                mlab.show()

        ## use for analysis 
        else:
            heatmap_singleplot(x, y, z, ql, tilts, 'ql', show_delaunay)
            if plot_pulse_shape:
                ps = np.array([(q + p)/2 for q, p in zip(df.ps.iloc[i], df.ps.iloc[order[i]])])
                heatmap_singleplot(x, y, z, ps, tilts, 'qs/ql', show_delaunay)
    mlab.show()

def plot_smoothed_fitted_heatmaps(fin1, fin2, dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, pulse_shape, beam_11MeV, multiplot, save_multiplot):
    ''' Plots data from smoothing_tilt()
        Taken from plot_fitted_heatmaps
    '''

    def map_smoothed_fitted_data_3d(data, det, tilts, crystal_orientation, theta_neutron, phi_neutron, beam_11MeV):
        # like map_data_3d but for fitted data
        det_df = data.loc[(data.det == det)]
        ql_all, theta_p, phi_p, angles_p = [], [], [], []
        for t, tilt in enumerate(tilts):
            
            tilt_df = det_df.loc[(data.tilt == tilt)]
            angles = np.arange(0, 180, 5) # 5 and 2 look good
      
            ql = sin_func(np.deg2rad(angles), tilt_df['a'].values, tilt_df['b'].values, tilt_df['phi'].values)

            #plt.figure(0)
            #plt.plot(angles, ql, 'o')
            
            thetap, phip = map_3d(tilt, crystal_orientation, angles, theta_neutron, phi_neutron)       
            ql_all.extend(ql)
            theta_p.extend(thetap)
            phi_p.extend(phip)
            angles_p.extend(angles)
    
        d = {'ql': ql_all, 'theta': theta_p, 'phi': phi_p, 'angles': angles_p}
        df = pd.DataFrame(data=d)
        return df

    if pulse_shape:
        f1 = fin1.split('.')
        fin1 = f1[0] + '_ps.' + f1[1]
        f2 = fin2.split('.')
        fin2 = f2[0] + '_ps.' + f2[1]

    data_bvert = pd_load(fin1, p_dir)
    data_cpvert = pd_load(fin2, p_dir)

    for d, det in enumerate(dets):
        if d > 5:
            continue
        print 'det_no =', det, 'theta_n =', theta_n[d]
        df_b_mapped = map_smoothed_fitted_data_3d(data_bvert, det, bvert_tilt, b_up, theta_n[d], phi_n[d], beam_11MeV)
        df_cp_mapped = map_smoothed_fitted_data_3d(data_cpvert, det, cpvert_tilt, cp_up, theta_n[d], phi_n[d], beam_11MeV)
        df_b_mapped_mirror = map_smoothed_fitted_data_3d(data_bvert, det, bvert_tilt, np.asarray(((1,0,0), (0,1,0), (0,0,-1))), theta_n[d], phi_n[d], beam_11MeV)
        df_cp_mapped_mirror = map_smoothed_fitted_data_3d(data_cpvert, det, cpvert_tilt, np.asarray(((-1,0,0), (0,0,-1), (0,1,0))), theta_n[d], phi_n[d], beam_11MeV)

        # convert to cartesian
        theta_b = np.concatenate([df_b_mapped.theta.values, df_b_mapped_mirror.theta.values])
        theta_cp = np.concatenate([df_cp_mapped.theta.values, df_cp_mapped_mirror.theta.values])
        phi_b = np.concatenate([df_b_mapped.phi.values, df_b_mapped_mirror.phi.values])
        phi_cp = np.concatenate([df_cp_mapped.phi.values, df_cp_mapped_mirror.phi.values])

        angles_b = np.concatenate([df_b_mapped.angles.values, df_b_mapped_mirror.angles.values])
        angles_cp = np.concatenate([df_cp_mapped.angles.values, df_cp_mapped_mirror.angles.values])
        angles = np.concatenate((angles_b, angles_cp))

        x_b, y_b, z_b = polar_to_cartesian(theta_b, phi_b, b_up, cp_up)
        x_cp, y_cp, z_cp = polar_to_cartesian(theta_cp, phi_cp, cp_up, cp_up)

        x = np.round(np.concatenate((x_b, x_cp)), 12)
        y = np.round(np.concatenate((y_b, y_cp)), 12)
        z = np.round(np.concatenate((z_b, z_cp)), 12)
        ql = np.concatenate([df_b_mapped.ql.values, df_b_mapped_mirror.ql.values, df_cp_mapped.ql.values, df_cp_mapped_mirror.ql.values])
        print max(ql)

        # remove repeated points
        xyz = np.array(zip(x,y,z))
        xyz, indices = np.unique(xyz, axis=0, return_index=True)
        ql = ql[indices]
        x, y, z = xyz.T

        # recover theta, phi to check 3d plotting
        theta = np.arccos(z)
        phi = []
        for a, b in zip(x, y):
            if (abs(a) < 1e-5 and abs(b) < 1e-5):
                print a, b
                p = 0.
            else:
                p = np.arctan(b/a) 
            phi.append(p)
        phi = np.array(phi)  

        if multiplot:
            heatmap_multiplot(x, y, z, ql, theta_n, d, cwd, save_multiplot, beam_11MeV, fitted=True)
            if save_multiplot:
                continue
            mlab.show()

        else:
            max_idx = np.argmax(ql)
            print 'ql_max = ', ql[max_idx], '\n'
            fig = mlab.figure(size=(400*2, 350*2)) 
            pts = mlab.points3d(x, y, z, ql, colormap='viridis', scale_mode='none', scale_factor=0.03)
            # plot max ql point (red)
            mlab.points3d(x[max_idx], y[max_idx], z[max_idx], ql[max_idx], color=(1,0,0), scale_mode='none', scale_factor=0.03)
            # delaunay triagulation (mesh, interpolation)
            tri = mlab.pipeline.delaunay3d(pts)
            tri_smooth = mlab.pipeline.poly_data_normals(tri) # smooths delaunay triangulation mesh
            surf = mlab.pipeline.surface(tri_smooth, colormap='viridis')
            
            #for x_val, y_val, z_val, ql_val, th, ph in zip(x, y, z, ql, np.concatenate((theta_b, theta_cp))[indices], angles[indices]):
            #    mlab.text3d(x_val, y_val, z_val, str(round(ph, 2)), scale=0.02, color=(0,0,0), figure=fig)

            mlab.axes(pts, xlabel='a', ylabel='b', zlabel='c\'')
            mlab.colorbar(surf, orientation='vertical') 
            mlab.view(azimuth=0, elevation=-90, distance=7.5, figure=fig)            
            mlab.show()  
        #plt.show()

def compare_a_axis_recoils(fin, dets, cwd, p_dir, plot_by_det, save_plots):

    def remove_cal(lo, m, b):
        return m*np.array(lo) + b

    def new_cal(lo, m, b, new_m, new_b):
        y = m*np.array(lo) + b
        return (y - new_b)/new_m

    a_qls, tilts_arr = [], []
    for f in fin:
        if '11' in f:
            beam_11MeV = True
        else:
            beam_11MeV = False
        if 'bvert' in f:
            tilts = [0, 45, -45, 30, -30, 15, -15]
        else:
            tilts = [0, 30, -30, 15, -15]

        data = pd_load(f, p_dir)
        data = split_filenames(data)
        a_axis_df, cp_b_axes_df = tilt_check(data, dets, tilts, f, cwd, p_dir, beam_11MeV, 
                                             print_max_ql=False, get_a_data=True, pulse_shape=False, delayed=False, prompt=False, show_plots=False, save_plots=False, save_pickle=False)

        # plot each det separately
        if plot_by_det:
            print '\n', f
            fig_no = [1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1]
            color = ['r', 'r', 'r', 'r', 'r', 'r', 'b', 'b', 'b', 'b', 'b', 'b']
            print '\ndet   ql_mean    std    rel uncert'
            print '------------------------------------'
            means = []
            for d, det in enumerate(dets):

                det_df = a_axis_df[(a_axis_df.det == 'det'+str(det))]
                if det_df.empty:
                    continue

                # calculate mean and std
                ql_mean = det_df.ql.mean()
                ql_std = det_df.ql.std()/np.sqrt(len(det_df.ql.values)) # uncertainty on the mean
                means.append(ql_mean)
                print '{:^4} {:>8} {:>8} {:>8}'.format(det, round(ql_mean, 4), round(ql_std, 4), round(ql_std/ql_mean, 4))

                plt.figure(fig_no[d], figsize=(10,7))
                plt.errorbar(det_df.tilt.values, det_df.ql.values, yerr=det_df.abs_uncert.values, ecolor='black', markerfacecolor='None', fmt='o', 
                                markeredgecolor=color[d], markeredgewidth=1, markersize=10, capsize=1, label='det' + str(det), zorder=10)
                xvals = np.linspace(-47, 47, 10)
                plt.plot(xvals, [ql_mean]*10, color=color[d], label='det' + str(det) + ' mean')
                plt.plot(xvals, [ql_mean + ql_std]*10, '--', color=color[d], alpha=0.2)
                plt.plot(xvals, [ql_mean - ql_std]*10, '--', color=color[d], alpha=0.2)
                plt.fill_between(xvals, [ql_mean + ql_std]*10, [ql_mean - ql_std]*10, facecolor=color[d], alpha=0.05, label='std')
                plt.xlabel('tilt angle (deg)')
                plt.ylabel('light output (MeVee)')
                plt.xlim(-47, 47)
                plt.title(f)
                plt.legend(fontsize=10)
                if save_plots:
                    if d > 5:
                        plt.savefig('bvert_a_det' + str(det) + '.png')

            # difference between BL mean and BR mean with uncerts
            det_no = ['4  15', '5  14', '6  13', '7  12', '8  11', '9  10']
            bl_dets = [4, 5, 6, 7, 8, 9]
            br_dets = [15, 14, 13, 12, 11, 10]
            print '\nbl_det  br_det  rel_diff'
            print '---------------------------'
            for i, mean in enumerate(reversed(means[6:])):
                print '{:^5} {:>6} {:>10}'.format(bl_dets[i], br_dets[i], round(abs((mean - means[i])/((mean + means[i])/2)), 4))
            if save_plots:
                print '\nfigures were saved\n'
            plt.show()

        # plot all qls for each crystal/energy
        else:
            plt.figure()
            plt.errorbar(a_axis_df.tilt.values, a_axis_df.ql.values, yerr=a_axis_df.abs_uncert.values, ecolor='black', markerfacecolor='None', fmt='o', 
                            markeredgecolor='red', markeredgewidth=1, markersize=10, capsize=1)
            plt.xlabel('tilt angle (deg)')
            plt.ylabel('light output (MeVee)')
            plt.xlim(-50, 50)
            plt.title(f)
        
        a_qls.append(a_axis_df.ql.values)
        tilts_arr.append(a_axis_df.tilt.values)
    
    # m and E_0 for 11 MeV cpvert
    m = 8662.28
    b = -166.65
    new_m = 8500
    mb = 8598.74
    bb = -155.15
    new_mb = 8700
    rel_diff = []
    print '\n11 MeV a-axis LO rel diff between bvert and cpvert crystals\n'
    print ' tilt  LO_bvert  LO_cpvert  rel_diff'
    for bvert_11, cpvert_11, tilt in zip(a_qls[0], a_qls[1], tilts_arr[0]):
        cpvert_11 = new_cal(cpvert_11, m, b, new_m, b)
        bvert_11 = new_cal(bvert_11, mb, bb, new_mb, bb)
        print '{:^5} {:>8} {:>9} {:>8}%'.format(tilt, bvert_11, cpvert_11, round((bvert_11 - cpvert_11)/(bvert_11 + cpvert_11)*200, 2))
        rel_diff.append((bvert_11 - cpvert_11)/(bvert_11 + cpvert_11)*200)
    print'\n rel_diff mean = ', np.mean(rel_diff), '%'   

    print '\n4 MeV a-axis LO rel diff between bvert and cpvert crystals\n'
    print ' tilt  LO_bvert  LO_cpvert  rel_diff'
    rel_diff = []
    for bvert_4, cpvert_4, tilt in zip(a_qls[2], a_qls[3], tilts_arr[2]):
        print '{:^5} {:>8} {:>9} {:>8}%'.format(tilt, bvert_4, cpvert_4, round((bvert_4 - cpvert_4)/(bvert_4 + cpvert_4)*200, 2))
        rel_diff.append((bvert_4 - cpvert_4)/(bvert_4 + cpvert_4)*200)
    print'\n rel_diff mean = ', np.mean(rel_diff), '%'

    plt.show()

def plot_ratios(fin, dets, cwd, p_dir, pulse_shape, plot_fit_ratio):
    ''' Plots a/c' and a/b ql or pulse shape ratios
        Set pulse_shape=True for pulse shape ratios
        Uses tilt_check function to get data from dataframes
    '''

    sin_fits = ['bvert_11MeV_sin_params_smoothed.p', 'cpvert_11MeV_sin_params_smoothed.p', 'bvert_4MeV_sin_params_smoothed.p', 'cpvert_4MeV_sin_params_smoothed.p']   
    def get_smoothed_data(f, dets):
        ratio = []
        data = pd_load(f, p_dir)
        for d, det in enumerate(dets):
            if d > 5:
                continue
            det_df = data.loc[(data.det == det)]
            for t, tilt in enumerate(tilts):
                if tilt == 0:
                    tilt_df = det_df.loc[(data.tilt == tilt)]
                    angles = np.arange(0, 180, 5) # 5 and 2 look good
                
                    ql = sin_func(np.deg2rad(angles), tilt_df['a'].values, tilt_df['b'].values, tilt_df['phi'].values)
                    ratio.append(max(ql)/min(ql))                
                else:
                    continue
        return ratio

    for i, f in enumerate(fin):
        label_fit = ['', '', '', 'fit ratios']
        label_smooth = ['', '', '', 'smooth fit ratios']
        label = ['L$_a$/L$_c\'$', 'L$_a$/L$_b$', '', '']
        label_pat = ['', '', '', 'Schuster ratios']
        if '11' in f:
            beam_11MeV = True
            angles = [70, 60, 50, 40, 30, 20, 20, 30, 40, 50, 60, 70]
            p_erg = 11.325*np.sin(np.deg2rad(angles))**2
            #ps_baseline_uncert = (0.01, 0.01, 0.02, 0.03, 0.08, 0.25, 0.25, 0.08, 0.03, 0.02, 0.01, 0.01) # ql uncer
            ps_baseline_uncert = (0.001, 0.001, 0.001, 0.002, 0.005, 0.016, 0.016, 0.005, 0.002, 0.001, 0.001, 0.001) # qs uncer
        else:
            beam_11MeV = False
            angles = [60, 40, 20, 30, 50, 70]
            p_erg = 4.825*np.sin(np.deg2rad(angles))**2
            #ps_baseline_uncert = (0.01, 0.01, 0.02, 0.04, 0.1, 0.3, 0.3, 0.1, 0.04, 0.02, 0.01, 0.01) # ql uncert
            ps_baseline_uncert = (0.001, 0.001, 0.002, 0.003, 0.008, 0.027, 0.026, 0.008, 0.003, 0.002, 0.001) # qs uncert
        if 'bvert' in f:
            #tilts = [0, 45, -45, 30, -30, 15, -15]
            tilts = [0]
            color = 'r'
        else:
            #tilts = [0, 30, -30, 15, -15]
            tilts = [0]
            color = 'b'

        data = pd_load(f, p_dir)
        data = split_filenames(data)  
        a_axis_df, cp_b_axes_df = tilt_check(data, dets, tilts, f, cwd, p_dir, beam_11MeV, print_max_ql=False, get_a_data=True, 
                                             pulse_shape=pulse_shape, delayed=False, prompt=False, show_plots=False, save_plots=False, save_pickle=False)
        print a_axis_df.to_string()
        print cp_b_axes_df.to_string()
    
        if beam_11MeV:
            a_ql = a_axis_df.ql.iloc[np.where(a_axis_df.tilt == 0)].values
            a_uncert = a_axis_df.abs_uncert.iloc[np.where(a_axis_df.tilt == 0)].values
            a_fit_ql = a_axis_df.fit_ql.iloc[np.where(a_axis_df.tilt == 0)].values
            cp_b_ql = cp_b_axes_df.ql.iloc[np.where(cp_b_axes_df.tilt == 0)].values
            cp_b_uncert = cp_b_axes_df.abs_uncert.iloc[np.where(cp_b_axes_df.tilt == 0)].values
            cp_b_fit_ql = cp_b_axes_df.fit_ql.iloc[np.where(cp_b_axes_df.tilt == 0)].values
            ratio = a_ql/cp_b_ql
            fit_ratio = a_fit_ql/cp_b_fit_ql
            baseline_unc_a = a_ql*ps_baseline_uncert
            baseline_unc_cp = cp_b_ql*ps_baseline_uncert
            #uncert = np.sqrt(ratio**2*(((a_uncert + baseline_unc_a)/a_ql)**2 + ((cp_b_uncert + baseline_unc_cp)/cp_b_ql)**2)) # includes baseline uncert
            uncert = np.sqrt(ratio**2 * ((a_uncert/a_ql)**2 + (cp_b_uncert/cp_b_ql)**2))
            shape = 'o'
        else:
            # account for skipped detectors with 4 MeV beam measurements
            ratio, uncert, fit_ratio = [], [], []
            for d, det in enumerate(a_axis_df.det.values):
                if det in cp_b_axes_df.det.values:
                    a_ql = a_axis_df.ql.iloc[np.where(a_axis_df.det == det)].values
                    a_uncert = a_axis_df.abs_uncert.iloc[np.where(a_axis_df.det == det)].values
                    a_fit_ql = a_axis_df.fit_ql.iloc[np.where(a_axis_df.det == det)].values
                    cp_b_ql = cp_b_axes_df.ql.iloc[np.where(cp_b_axes_df.det == det)].values
                    cp_b_uncert = cp_b_axes_df.abs_uncert.iloc[np.where(cp_b_axes_df.det == det)].values
                    cp_b_fit_ql = cp_b_axes_df.fit_ql.iloc[np.where(cp_b_axes_df.det == det)].values
                    rat = a_ql/cp_b_ql
                    baseline_unc_a = a_ql*ps_baseline_uncert[d]
                    baseline_unc_cp = cp_b_ql*ps_baseline_uncert[d]
                    #unc = np.sqrt(rat**2*(((a_uncert + baseline_unc_a)/a_ql)**2 + ((cp_b_uncert + baseline_unc_cp)/cp_b_ql)**2)) # includes baseline uncert
                    unc = np.sqrt(rat**2 * ((a_uncert/a_ql)**2 + (cp_b_uncert/cp_b_ql)**2)) # no baseline uncert
                    rat_fit = a_fit_ql/cp_b_fit_ql
                    ratio.append(rat)
                    uncert.append(unc)
                    fit_ratio.append(rat_fit)
                    shape = '^'
                else:
                    continue

        if pulse_shape:
            plt.figure(0)
            # plot measured a/cp and a/b ratios 
            plt.errorbar(p_erg, ratio, yerr=uncert, ecolor='black', markerfacecolor='None', fmt=shape, 
                         markeredgecolor=color, markeredgewidth=1, markersize=10, capsize=1, label=label[i])
            # schuster ratios 2.5
            pat_ratios = [1.071, 1.100, 1.078, 1.066, 1.034, 1.058, 1.039, 1.048]     
            pat_ergs = [14.1, 14.1, 14.1, 14.1, 2.5, 2.5, 2.5, 2.5]        
            plt.errorbar(pat_ergs, pat_ratios, ecolor='black', markerfacecolor='None', fmt='x', 
                         markeredgecolor='k', markeredgewidth=1, markersize=7, capsize=1, label=label_pat[i])            
            # plot fitted data
            if plot_fit_ratio:
                plt.errorbar(p_erg, fit_ratio, ecolor='black', markerfacecolor='None', fmt='s', 
                             markeredgecolor='g', markeredgewidth=1, markersize=10, capsize=1, label=label_fit[i])
            xmin, xmax = plt.xlim(0, 14.5)
            plt.plot(np.linspace(xmin, xmax, 10), [1.0]*10, 'k--')            
            plt.ylabel('psd parameter ratio')
            plt.xlabel('proton recoil energy (MeV)')
            plt.ylim(0.91, 1.12)
            plt.legend(loc=4)
        else:
            plt.figure(0)
            # plot measured a/cp and a/b ratios 
            plt.errorbar(p_erg, ratio, yerr=uncert, ecolor='black', markerfacecolor='None', fmt=shape, 
                         markeredgecolor=color, markeredgewidth=1, markersize=10, capsize=1, label=label[i])
            # plot fitted data
            if plot_fit_ratio:
                plt.errorbar(p_erg, fit_ratio, ecolor='black', markerfacecolor='None', fmt='s', 
                             markeredgecolor='g', markeredgewidth=1, markersize=10, capsize=1, label=label_fit[i])
            # plot smoothed ratios
            smoothed_ratio = get_smoothed_data(sin_fits[i], dets)
            print p_erg, smoothed_ratio
            plt.errorbar(p_erg[:6], smoothed_ratio, ecolor='black', markerfacecolor='None', fmt='>', 
            markeredgecolor='k', markeredgewidth=1, markersize=10, capsize=1, label=label_smooth[i])

            xmin, xmax = plt.xlim(0, 11)
            plt.plot(np.linspace(xmin, xmax, 10), [1.0]*10, 'k--')
            plt.ylabel('light output ratio')
            plt.legend()
            plt.xlabel('proton recoil energy (MeV)')     
        
    plt.show()

def adc_vs_cal_ratios(fin, dets, cwd, p_dir, plot_fit_ratio):
    ''' Use to analyze effect of calibration on light output ratios
    
    '''
    def remove_cal(lo, m, b):
        return m*lo + b

    def new_cal(lo, m, b, new_m, new_b):
        y = m*lo + b
        return (y - new_b)/new_m

    ratios, ratios_adc, ratios_new, p_ergs = [], [], [], []
    for i, f in enumerate(fin):
        label_fit = ['', '', '', 'fit ratios']
        label = ['L$_a$/L$_c\'$', 'L$_a$/L$_b$', '', '']
        label_pat = ['', '', '', 'Schuster ratios']
        if '11' in f:
            beam_11MeV = True
            angles = [70, 60, 50, 40, 30, 20, 20, 30, 40, 50, 60, 70]
            p_erg = 11.325*np.sin(np.deg2rad(angles))**2
        else:
            beam_11MeV = False
            angles = [60, 40, 20, 30, 50, 70]
            p_erg = 4.825*np.sin(np.deg2rad(angles))**2
        if 'bvert' in f:
            #tilts = [0, 45, -45, 30, -30, 15, -15]
            tilts = [0]
            color = 'r'
            if '11MeV' in f:
                m = 8598.74  # calibration terms from /home/radians/raweldon/tunl.2018.1_analysis/stilbene_final/lo_calibration/gamma_calibration.py
                b = -155.15
                new_m = 8868.6
                new_b = -190.0
            if '4MeV' in f:
                m = 25686.35
                b = -544.24
                new_m = 25000 #25690.4
                new_b = 125 #-497.5
        else:
            #tilts = [0, 30, -30, 15, -15]
            tilts = [0]
            color = 'b'
            if '11MeV' in f:
                m = 8662.28
                b = -166.65
                new_m = 8894.9
                new_b = -212.2
            if '4MeV' in f:
                m = 26593.44
                b = -534.64
                new_m = 26573.7
                new_b = -658.3

        data = pd_load(f, p_dir)
        data = split_filenames(data)  
        a_axis_df, cp_b_axes_df = tilt_check(data, dets, tilts, f, cwd, p_dir, beam_11MeV, print_max_ql=False, get_a_data=True, 
                                             pulse_shape=False, delayed=False, prompt=False, show_plots=False, save_plots=False, save_pickle=False)
        print a_axis_df.to_string()
        print cp_b_axes_df.to_string()
    
        if beam_11MeV:
            a_ql = a_axis_df.ql.iloc[np.where(a_axis_df.tilt == 0)].values
            a_uncert = a_axis_df.abs_uncert.iloc[np.where(a_axis_df.tilt == 0)].values
            a_fit_ql = a_axis_df.fit_ql.iloc[np.where(a_axis_df.tilt == 0)].values
            a_ql_adc = np.array([remove_cal(lo, m, b) for lo in a_ql])
            a_uncert_adc = np.array([remove_cal(lo, m, b) for lo in a_uncert])
            a_fit_ql_adc = np.array([remove_cal(lo, m, b) for lo in a_fit_ql])    
            a_ql_new = np.array([new_cal(lo, m, b, new_m, new_b) for lo in a_ql])
            a_uncert_new = np.array([new_cal(lo, m, b, new_m, new_b) for lo in a_uncert])
            a_fit_ql_new = np.array([new_cal(lo, m, b, new_m, new_b) for lo in a_fit_ql]) 

            cp_b_ql = cp_b_axes_df.ql.iloc[np.where(cp_b_axes_df.tilt == 0)].values
            cp_b_uncert = cp_b_axes_df.abs_uncert.iloc[np.where(cp_b_axes_df.tilt == 0)].values
            cp_b_fit_ql = cp_b_axes_df.fit_ql.iloc[np.where(cp_b_axes_df.tilt == 0)].values
            cp_b_ql_adc = np.array([remove_cal(lo, m, b) for lo in cp_b_ql])
            cp_b_uncert_adc = np.array([remove_cal(lo, m, b) for lo in cp_b_uncert])
            cp_b_fit_ql_adc = np.array([remove_cal(lo, m, b) for lo in cp_b_fit_ql])    
            cp_b_ql_new = np.array([new_cal(lo, m, b, new_m, new_b) for lo in cp_b_ql])
            cp_b_uncert_new = np.array([new_cal(lo, m, b, new_m, new_b) for lo in cp_b_uncert])
            cp_b_fit_ql_new = np.array([new_cal(lo, m, b, new_m, new_b) for lo in cp_b_fit_ql]) 

            ratio = a_ql/cp_b_ql
            ratio_adc = a_ql_adc/cp_b_ql_adc
            ratio_new = a_ql_new/cp_b_ql_new
            fit_ratio = a_fit_ql/cp_b_fit_ql
            fit_ratio_adc = a_fit_ql_adc/cp_b_fit_ql_adc
            fit_ratio_new = a_fit_ql_new/cp_b_fit_ql_new
            uncert = np.sqrt(ratio**2 * ((a_uncert/a_ql)**2 + (cp_b_uncert/cp_b_ql)**2))
            uncert_adc = np.sqrt(ratio_adc**2 * ((a_uncert_adc/a_ql_adc)**2 + (cp_b_uncert_adc/cp_b_ql_adc)**2))
            uncert_new = np.sqrt(ratio_new**2 * ((a_uncert_new/a_ql_new)**2 + (cp_b_uncert_new/cp_b_ql_new)**2))
            shape = 'o'
        else:
            # account for skipped detectors with 4 MeV beam measurements
            ratio, uncert, fit_ratio, ratio_adc, uncert_adc, fit_ratio_adc, ratio_new, uncert_new, fit_ratio_new = [], [], [], [], [], [], [], [], []
            for d, det in enumerate(a_axis_df.det.values):
                if det in cp_b_axes_df.det.values:
                    a_ql = a_axis_df.ql.iloc[np.where(a_axis_df.det == det)].values
                    a_uncert = a_axis_df.abs_uncert.iloc[np.where(a_axis_df.det == det)].values
                    a_fit_ql = a_axis_df.fit_ql.iloc[np.where(a_axis_df.det == det)].values
                    a_ql_adc = np.array([remove_cal(lo, m, b) for lo in a_ql])
                    a_uncert_adc = np.array([remove_cal(lo, m, b) for lo in a_uncert])
                    a_fit_ql_adc = np.array([remove_cal(lo, m, b) for lo in a_fit_ql])  
                    a_ql_new = np.array([new_cal(lo, m, b, new_m, new_b) for lo in a_ql])
                    a_uncert_new = np.array([new_cal(lo, m, b, new_m, new_b) for lo in a_uncert])
                    a_fit_ql_new = np.array([new_cal(lo, m, b, new_m, new_b) for lo in a_fit_ql])

                    cp_b_ql = cp_b_axes_df.ql.iloc[np.where(cp_b_axes_df.det == det)].values
                    cp_b_uncert = cp_b_axes_df.abs_uncert.iloc[np.where(cp_b_axes_df.det == det)].values
                    cp_b_fit_ql = cp_b_axes_df.fit_ql.iloc[np.where(cp_b_axes_df.det == det)].values
                    cp_b_ql_adc = np.array([remove_cal(lo, m, b) for lo in cp_b_ql])
                    cp_b_uncert_adc = np.array([remove_cal(lo, m, b) for lo in cp_b_uncert])
                    cp_b_fit_ql_adc = np.array([remove_cal(lo, m, b) for lo in cp_b_fit_ql])  
                    cp_b_ql_new = np.array([new_cal(lo, m, b, new_m, new_b) for lo in cp_b_ql])
                    cp_b_uncert_new = np.array([new_cal(lo, m, b, new_m, new_b) for lo in cp_b_uncert])
                    cp_b_fit_ql_new = np.array([new_cal(lo, m, b, new_m, new_b) for lo in cp_b_fit_ql]) 
                    
                    rat = a_ql/cp_b_ql
                    rat_adc = a_ql_adc/cp_b_ql_adc
                    rat_new = a_ql_new/cp_b_ql_new
                    unc = np.sqrt(rat**2 * ((a_uncert/a_ql)**2 + (cp_b_uncert/cp_b_ql)**2)) # no baseline uncert
                    unc_adc = np.sqrt(rat_adc**2 * ((a_uncert_adc/a_ql_adc)**2 + (cp_b_uncert_adc/cp_b_ql_adc)**2))
                    unc_new = np.sqrt(rat_new**2 * ((a_uncert_new/a_ql_new)**2 + (cp_b_uncert_new/cp_b_ql_new)**2))
                    print a_ql_new, a_uncert_new, cp_b_ql_new, cp_b_uncert_new, unc_new
                    rat_fit = a_fit_ql/cp_b_fit_ql
                    rat_fit_adc = a_fit_ql_adc/cp_b_fit_ql_adc
                    rat_fit_new = a_fit_ql_new/cp_b_fit_ql_new
                    ratio.extend(rat)
                    ratio_adc.extend(rat_adc)
                    ratio_new.extend(rat_new)
                    uncert.extend(unc)
                    uncert_adc.extend(unc_adc)
                    uncert_new.extend(unc_new)
                    fit_ratio.extend(rat_fit)
                    fit_ratio_adc.extend(rat_fit_adc)
                    fit_ratio_new.extend(rat_fit_new)
                    shape = '^'
                else:
                    continue
        ratios.append(ratio)    
        ratios_adc.append(ratio_adc)  
        ratios_new.append(ratio_new)
        p_ergs.append(p_erg)

        plt.figure(0)
        # plot measured a/cp and a/b ratios 
        plt.errorbar(p_erg, ratio, yerr=uncert, ecolor='black', markerfacecolor='None', fmt=shape, 
                        markeredgecolor=color, markeredgewidth=1, markersize=10, capsize=1, label=label[i])
        # plot fitted data
        if plot_fit_ratio:
            plt.errorbar(p_erg, fit_ratio, ecolor='black', markerfacecolor='None', fmt='s', 
                            markeredgecolor='g', markeredgewidth=1, markersize=10, capsize=1, label=label_fit[i])
        xmin, xmax = plt.xlim(0, 11)
        plt.plot(np.linspace(xmin, xmax, 10), [1.0]*10, 'k--')
        plt.ylabel('light output ratio')
        plt.legend()
        plt.xlabel('proton recoil energy (MeV)')   

        plt.figure(1)
        # plot measured a/cp and a/b ratios 
        plt.errorbar(p_erg, ratio_adc, ecolor='black', markerfacecolor='None', fmt=shape, 
                        markeredgecolor=color, markeredgewidth=1, markersize=10, capsize=1, label=label[i])
        # plot fitted data
        if plot_fit_ratio:
            plt.errorbar(p_erg, fit_ratio_adc, ecolor='black', markerfacecolor='None', fmt='s', 
                            markeredgecolor='g', markeredgewidth=1, markersize=10, capsize=1, label=label_fit[i])
        xmin, xmax = plt.xlim(0, 11)
        plt.plot(np.linspace(xmin, xmax, 10), [1.0]*10, 'k--')
        plt.ylabel('relative light yield ratio')
        plt.legend()
        plt.xlabel('proton recoil energy (MeV)')

        plt.figure(2)
        # plot measured a/cp and a/b ratios 
        plt.errorbar(p_erg, ratio_new, yerr=uncert_new, ecolor='black', markerfacecolor='None', fmt=shape, 
                        markeredgecolor=color, markeredgewidth=1, markersize=10, capsize=1, label=label[i])
        # plot fitted data
        if plot_fit_ratio:
            plt.errorbar(p_erg, fit_ratio_new, ecolor='black', markerfacecolor='None', fmt='s', 
                            markeredgecolor='g', markeredgewidth=1, markersize=10, capsize=1, label=label_fit[i])
        xmin, xmax = plt.xlim(0, 11)
        plt.plot(np.linspace(xmin, xmax, 10), [1.0]*10, 'k--')
        plt.ylabel('relative light yield ratio')
        plt.legend()
        plt.xlabel('proton recoil energy (MeV)')
   
    # print results
    print '   Ep     MeVee     ADC     %diff    new_cal     %diff'
    for ep, ratio, ratio_adc, ratio_new in zip(p_ergs, ratios, ratios_adc, ratios_new):
        for e, r, r_adc, r_new in zip(ep, ratio, ratio_adc, ratio_new):
            print '{:^8.2f} {:^8.3f} {:^8.3f} {:^8.2f} {:^8.3f} {:^8.2f}'.format(e, r, r_adc, 100*2*(r_adc - r)/(r_adc+r), r_new, 100*2*(r_adc - r_new)/(r_new + r_adc))
 

    plt.show()

def plot_acp_lo_curves(fin, dets, cwd, p_dir, pulse_shape, br_only, plot_fit_data):
    ''' plot light ouput curves of major axes
        includes function to remove calibration for checking true shape
    '''

    def remove_cal(lo, m, b):
        return m*np.array(lo) + b

    def new_cal(lo, m, b, new_m, new_b):
        y = m*np.array(lo) + b
        return (y - new_b)/new_m

    def calc_avg(data, uncert, only_uncert):    
        if only_uncert:
                return np.array([np.sqrt((uncert[:6][i]**2 + u**2)/2) for i, u in enumerate(uncert[6:][::-1])])
        else:                                                                                                                                               
            avg = np.array((data[:6] + data[6:][::-1]))/2.
            uncert = np.array([np.sqrt((uncert[:6][i]**2 + u**2)/2) for i, u in enumerate(uncert[6:][::-1])])
            return avg, uncert

    sin_fits = ['bvert_11MeV_sin_params_smoothed.p', 'cpvert_11MeV_sin_params_smoothed.p', 'bvert_4MeV_sin_params_smoothed.p', 'cpvert_4MeV_sin_params_smoothed.p']   
    def get_smoothed_data(f, dets):
        ql_max, ql_min = [], []
        data = pd_load(f, p_dir)
        for d, det in enumerate(dets):
            if d > 5:
                continue
            det_df = data.loc[(data.det == det)]
         
            tilt_df = det_df.loc[(data.tilt == 0)]
            angles = np.arange(0, 180, 5) # 5 and 2 look good
            ql = sin_func(np.deg2rad(angles), tilt_df['a'].values, tilt_df['b'].values, tilt_df['phi'].values)
            ql_max.append(max(ql))
            ql_min.append(min(ql))                               
        return ql_max, ql_min

    for i, f in enumerate(fin):
        label_fit = ['', '', '', 'fit ratios']
        label_smooth = ['', '', '', 'smooth fit ratios']
        label = [', b-vert', ', c\'-vert',  '', '']
        a_axis = ['a-axis', 'a-axis', '', '']
        cp_b_axis = ['cp-axis', 'b-axis', '', '']
        if '11' in f:
            beam_11MeV = True
            angles = [70, 60, 50, 40, 30, 20, 20, 30, 40, 50, 60, 70]
            dists = [66.4, 63.8, 63.2, 63.6, 64.9, 65.8, 66.0, 65.3, 63.4, 62.7, 64.3, 66.5]
            a_p_erg = 11.325*np.sin(np.deg2rad(angles))**2
            cp_b_p_erg = a_p_erg
            if br_only:
                a_p_erg = a_p_erg[:6]
                cp_b_p_erg = a_p_erg
        else:
            beam_11MeV = False
            a_angles = [70, 60, 50, 40, 30, 20, 30, 50, 70]
            a_dists = [66.4, 63.8, 63.2, 63.6, 64.9, 65.8, 65.3, 62.7, 66.5]
            a_p_erg = 4.825*np.sin(np.deg2rad(a_angles))**2
            cp_b_angles = [60, 40, 20, 20, 30, 40, 50, 60, 70]
            cp_b_dists = [63.8, 63.6, 65.8, 66.0, 65.3, 63.4, 62.7, 64.3, 66.5]
            cp_b_p_erg = 4.825*np.sin(np.deg2rad(cp_b_angles))**2                

        if 'bvert' in f:
            #tilts = [0, 45, -45, 30, -30, 15, -15]
            tilts = [0]
            color = 'b'
            shape = '^'
            if '11MeV' in f:
                m = 8598.74  # 8/6/19 - calibration terms from /home/radians/raweldon/tunl.2018.1_analysis/stilbene_final/lo_calibration/gamma_calibration.py
                b = -155.15
                cal_476 = remove_cal(0.476, m, b)
                new_m = 8750
            if '4MeV' in f:
                m = 25868.35
                b = -544.24
                cal_476 = remove_cal(0.476, m, b)
                new_m = m
        else:
            #tilts = [0, 30, -30, 15, -15]
            tilts = [0]
            color = 'g'
            shape = 's'
            if '11MeV' in f:
                m = 8662.28  # 8/6/19 - calibration terms from /home/radians/raweldon/tunl.2018.1_analysis/stilbene_final/lo_calibration/gamma_calibration.py
                b = -166.65
                cal_476 = remove_cal(0.476, m, b)
                new_m = 8600
            if '4MeV' in f:
                m = 26593.35
                b = -534.64
                cal_476 = remove_cal(0.476, m, b)
                new_m = 26000

        data = pd_load(f, p_dir)
        data = split_filenames(data)  
        a_axis_df, cp_b_axes_df = tilt_check(data, dets, tilts, f, cwd, p_dir, beam_11MeV, print_max_ql=False, get_a_data=True, 
                                             pulse_shape=pulse_shape, delayed=False, prompt=False, show_plots=False, save_plots=False, save_pickle=False)
        print a_axis_df.to_string()
        print cp_b_axes_df.to_string()
    
        if beam_11MeV:
            a_ql = a_axis_df.ql.values
            a_uncert = a_axis_df.abs_uncert.values
            a_fit_ql = a_axis_df.fit_ql.values
            a_ep_err = calc_ep_uncerts(a_axis_df.counts.values, angles, dists, beam_11MeV, print_unc=False) 
            cp_b_ql = cp_b_axes_df.ql.values
            cp_b_uncert = cp_b_axes_df.abs_uncert.values
            cp_b_fit_ql = cp_b_axes_df.fit_ql.values
            cp_b_ep_err = calc_ep_uncerts(cp_b_axes_df.counts.values, angles, dists, beam_11MeV, print_unc=False) 

            if br_only:
                a_ql = a_ql[:6]
                a_uncert = a_uncert[:6]
                a_fit_ql = a_fit_ql[:6]
                a_ep_err = a_ep_err[:6]
                cp_b_ql = cp_b_ql[:6]
                cp_b_uncert = cp_b_uncert[:6]
                cp_b_fit_ql = cp_b_fit_ql[:6]
                cp_b_ep_err = cp_b_ep_err[:6]
        else:
            # account for skipped detectors with 4 MeV beam measurements
            a_ql, cp_b_ql, a_uncert, cp_b_uncert, a_fit_ql, cp_b_fit_ql, a_counts, cp_b_counts  = [], [], [], [], [], [], [], []
            for d, det in enumerate(a_axis_df.det.values):
                    a_ql.append(a_axis_df.ql.iloc[np.where(a_axis_df.det == det)].values)
                    a_uncert.append(a_axis_df.abs_uncert.iloc[np.where(a_axis_df.det == det)].values)
                    a_fit_ql.append(a_axis_df.fit_ql.iloc[np.where(a_axis_df.det == det)].values)
                    a_counts.append(a_axis_df.counts.iloc[np.where(a_axis_df.det == det)].values)
            for d, det in enumerate(cp_b_axes_df.det.values):
                    cp_b_ql.append(cp_b_axes_df.ql.iloc[np.where(cp_b_axes_df.det == det)].values)
                    cp_b_uncert.append(cp_b_axes_df.abs_uncert.iloc[np.where(cp_b_axes_df.det == det)].values)
                    cp_b_fit_ql.append(cp_b_axes_df.fit_ql.iloc[np.where(cp_b_axes_df.det == det)].values)
                    cp_b_counts.append(cp_b_axes_df.counts.iloc[np.where(cp_b_axes_df.det == det)].values)
  
            a_ep_err = calc_ep_uncerts(a_counts, a_angles, a_dists, beam_11MeV, print_unc=False)
            cp_b_ep_err = calc_ep_uncerts(cp_b_counts, cp_b_angles, cp_b_dists, beam_11MeV, print_unc=False)

        plt.figure(0)
        plt.errorbar(a_p_erg, a_ql, yerr=a_uncert, xerr=a_ep_err, ecolor='black', markerfacecolor='None', fmt=shape, 
                        markeredgecolor='r', markeredgewidth=1, markersize=9, capsize=1, label= a_axis[i]+label[i])
        plt.errorbar(cp_b_p_erg, cp_b_ql, yerr=cp_b_uncert, xerr=cp_b_ep_err, ecolor='black', markerfacecolor='None', fmt=shape, 
                        markeredgecolor=color, markeredgewidth=1, markersize=9, capsize=1, label=cp_b_axis[i]+label[i])
    
        a_smoothed, cp_b_smoothed = get_smoothed_data(sin_fits[i], dets)
        if br_only:
            if '11MeV' in f:
                a_erg = a_p_erg
                cp_erg = cp_b_p_erg[-6:]
            else:
                a_erg = a_p_erg[:6]
                cp_erg = cp_b_p_erg[-6:][::-1]
                print a_erg, a_smoothed
            plt.errorbar(a_erg, a_smoothed, ecolor='black', markerfacecolor='None', fmt=shape, 
                            markeredgecolor='r', markeredgewidth=1, markersize=9, capsize=1, label= a_axis[i]+label[i])
            plt.errorbar(cp_erg, cp_b_smoothed, ecolor='black', markerfacecolor='None', fmt=shape, 
                            markeredgecolor=color, markeredgewidth=1, markersize=9, capsize=1, label=cp_b_axis[i]+label[i])
        else:
            plt.errorbar(a_p_erg, a_smoothed, ecolor='black', markerfacecolor='None', fmt=shape, 
                            markeredgecolor='r', markeredgewidth=1, markersize=9, capsize=1, label= a_axis[i]+label[i])
            plt.errorbar(cp_b_p_erg[-6:][::-1], cp_b_smoothed, ecolor='black', markerfacecolor='None', fmt=shape, 
                            markeredgecolor=color, markeredgewidth=1, markersize=9, capsize=1, label=cp_b_axis[i]+label[i])

        # plot fitted data
        if plot_fit_data:
            plt.errorbar(a_p_erg, a_fit_ql, ecolor='black', markerfacecolor='None', fmt=shape, 
                        markeredgecolor='r', markeredgewidth=1, markersize=9, capsize=1, label= a_axis[i]+label[i])
            plt.errorbar(cp_b_p_erg, cp_b_fit_ql, ecolor='black', markerfacecolor='None', fmt=shape, 
                        markeredgecolor=color, markeredgewidth=1, markersize=9, capsize=1, label=cp_b_axis[i]+label[i])
        plt.ylabel('Light output (MeVee)')
        plt.xlabel('Proton recoil energy (MeV)')     
        plt.legend(loc=4)
        plt.ylim(-0.1, 6)

        # plot relative light output 
        plt.figure(1)
        plt.errorbar(a_p_erg, remove_cal(a_ql, m, b)/cal_476, yerr=remove_cal(a_uncert, m, b)/cal_476, ecolor='black', markerfacecolor='None', fmt=shape, 
                        markeredgecolor='r', markeredgewidth=1, markersize=10, capsize=1, label= a_axis[i]+label[i])
        plt.errorbar(cp_b_p_erg, remove_cal(cp_b_ql, m, b)/cal_476, yerr=remove_cal(cp_b_uncert, m, b)/cal_476, ecolor='black', markerfacecolor='None', fmt=shape, 
                        markeredgecolor=color, markeredgewidth=1, markersize=10, capsize=1, label=cp_b_axis[i]+label[i])
    
        # plot fitted data
        if plot_fit_data:
            plt.errorbar(a_p_erg, remove_cal(a_fit_ql, m, b)/cal_476, ecolor='black', markerfacecolor='None', fmt=shape, 
                            markeredgecolor='r', markeredgewidth=1, markersize=10, capsize=1, label= a_axis[i]+label[i])
            plt.errorbar(cp_b_p_erg, remove_cal(cp_b_fit_ql, m, b)/cal_476, ecolor='black', markerfacecolor='None', fmt=shape, 
                            markeredgecolor=color, markeredgewidth=1, markersize=10, capsize=1, label=cp_b_axis[i]+label[i])
        plt.yscale('log')
        plt.xscale('log')
        plt.xlim(0.4, 12)
        plt.ylim(0.08, 15)
        plt.ylabel('Relative light output')
        plt.xlabel('Proton recoil energy (MeV)')     
        plt.legend(loc=4)

        plt.figure(2)
        plt.errorbar(a_p_erg, new_cal(a_ql, m, b, new_m, b), yerr=a_uncert, ecolor='black', markerfacecolor='None', fmt=shape, 
                        markeredgecolor='r', markeredgewidth=1, markersize=10, capsize=1, label= a_axis[i]+label[i])
        plt.errorbar(cp_b_p_erg, new_cal(cp_b_ql, m, b, new_m, b), yerr=cp_b_uncert, ecolor='black', markerfacecolor='None', fmt=shape, 
                        markeredgecolor=color, markeredgewidth=1, markersize=10, capsize=1, label=cp_b_axis[i]+label[i])  
        plt.ylabel('Light output (MeVee)')
        plt.xlabel('Proton recoil energy (MeV)')     
        plt.legend(loc=4)

    plt.figure(0)
    plt.plot((2.5, 14.1), (0.886, 8.981), 'x', color='k', label='Schuster')
    plt.legend(loc=4)

    plt.show()

def polar_norm(x1, x2):
    # norm distance function for polar coords (default is euclidean, cartesian) - from wikipedia
    norm = np.sqrt(x1[1]**2 + x2[1]**2 - 2*x1[1]*x2[1]*np.cos(x1[0] - x2[0]))
    norm[np.isnan(norm)] = 0
    return norm

def polar_plot(fin1, fin2, dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, beam_11MeV):
    ''' I was going to use this function to interpolate over the points in polar coordinates, but that does not work well due to 
            wrap around from the edges
        Need to use interpolation on the sphere to avoid edge effects
    '''
    data_bvert = pd_load(fin1, p_dir)
    data_bvert = split_filenames(data_bvert)
    data_cpvert = pd_load(fin2, p_dir)
    data_cpvert = split_filenames(data_cpvert)
    for d, det in enumerate(dets):
        print '\ndet_no =', det, 'theta_n =', theta_n[d]
        
        df_b_mapped = map_data_3d(data_bvert, det, bvert_tilt, b_up, theta_n[d], phi_n[d], beam_11MeV) 
        df_b_mapped_mirror = map_data_3d(data_bvert, det, bvert_tilt, np.asarray(((1,0,0), (0,1,0), (0,0,-1))), theta_n[d], phi_n[d], beam_11MeV)
        df_cp_mapped = map_data_3d(data_cpvert, det, cpvert_tilt, cp_up, theta_n[d], phi_n[d], beam_11MeV)
        df_cp_mapped_mirror = map_data_3d(data_cpvert, det, cpvert_tilt, np.asarray(((-1,0,0), (0,0,-1), (0,1,0))), theta_n[d], phi_n[d], beam_11MeV)

        theta_b = np.concatenate([df_b_mapped.theta.values, df_b_mapped_mirror.theta.values])
        theta_cp = np.concatenate([df_cp_mapped.theta.values, df_cp_mapped_mirror.theta.values])
        theta = np.concatenate([theta_b, theta_cp])
        phi_b = np.concatenate([df_b_mapped.phi.values, df_b_mapped_mirror.phi.values])
        phi_cp = np.concatenate([df_cp_mapped.phi.values, df_cp_mapped_mirror.phi.values])
        phi = np.concatenate([phi_b, phi_cp])

        #theta_b = df_b_mapped.theta.values
        #theta_cp = df_cp_mapped.theta.values
        #theta = np.concatenate([theta_b, theta_cp])
        #phi_b = df_b_mapped.phi.values
        #phi_cp = df_cp_mapped.phi.values
        #phi = np.concatenate([phi_b, phi_cp])

        ql = np.concatenate([df_b_mapped.ql_mean.values, df_b_mapped_mirror.ql_mean.values, df_cp_mapped.ql_mean.values, df_cp_mapped_mirror.ql_mean.values])
        tilts = np.concatenate([df_b_mapped.tilt.values, df_b_mapped_mirror.tilt.values, df_cp_mapped.tilt.values, df_cp_mapped_mirror.tilt.values])

        # remove duplicates
        phi_theta = np.array(zip(np.round(phi, 10), np.round(theta, 10)))
        phi_theta, indices = np.unique(phi_theta, axis=0, return_index=True)
        ql = ql[indices]
        phi, theta = phi_theta.T

        r = np.sqrt(1 - np.cos(theta))
        p = np.linspace(0, max(phi), 100)
        t = np.linspace(0, max(r), 100)

        print min(phi), max(phi)
        print min(np.rad2deg(theta)), max(np.rad2deg(theta))
        print min(r), max(r)

        phi_mesh, r_mesh = np.meshgrid(p, t)

        # radial basis function
        interp = scipy.interpolate.Rbf(phi, r, ql, function='linear', smooth=0.5, norm=polar_norm)
        ql_int = interp(phi_mesh, r_mesh)

        ax = plt.subplot(111, projection='polar')
        ax.scatter(phi, r, c='k', s=20)
        ax.pcolormesh(phi_mesh, r_mesh, ql_int, cmap='viridis')
        ax.set_rmax(1.415)
        ax.grid(True)
        plt.show()

def sph_norm(x1, x2):
    '''Distance metric on the surface of the unit sphere (from http://jessebett.com/Radial-Basis-Function-USRA/)'''
    norm = np.arccos((x1 * x2).sum(axis=0))
    norm[np.isnan(norm)] = 0
    return norm

def rbf_interp_heatmap(fin1, fin2, dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, beam_11MeV, plot_pulse_shape, multiplot, save_multiplot):
    data_bvert = pd_load(fin1, p_dir)
    data_bvert = split_filenames(data_bvert)
    data_cpvert = pd_load(fin2, p_dir)
    data_cpvert = split_filenames(data_cpvert)

    for d, det in enumerate(dets):       
        print '\ndet_no =', det, 'theta_n =', theta_n[d]
        df_b_mapped = map_data_3d(data_bvert, det, bvert_tilt, b_up, theta_n[d], phi_n[d], beam_11MeV) 
        df_b_mapped_mirror = map_data_3d(data_bvert, det, bvert_tilt, np.asarray(((1,0,0), (0,1,0), (0,0,-1))), theta_n[d], phi_n[d], beam_11MeV)
        df_cp_mapped = map_data_3d(data_cpvert, det, cpvert_tilt, cp_up, theta_n[d], phi_n[d], beam_11MeV)
        df_cp_mapped_mirror = map_data_3d(data_cpvert, det, cpvert_tilt, np.asarray(((-1,0,0), (0,0,-1), (0,1,0))), theta_n[d], phi_n[d], beam_11MeV)

        # convert to cartesian
        theta_b = np.concatenate([df_b_mapped.theta.values, df_b_mapped_mirror.theta.values])
        theta_cp = np.concatenate([df_cp_mapped.theta.values, df_cp_mapped_mirror.theta.values])
        phi_b = np.concatenate([df_b_mapped.phi.values, df_b_mapped_mirror.phi.values])
        phi_cp = np.concatenate([df_cp_mapped.phi.values, df_cp_mapped_mirror.phi.values])

        x_b, y_b, z_b = polar_to_cartesian(theta_b, phi_b, b_up, cp_up)
        x_cp, y_cp, z_cp = polar_to_cartesian(theta_cp, phi_cp, cp_up, cp_up)

        x = np.round(np.concatenate((x_b, x_cp)), 12)
        y = np.round(np.concatenate((y_b, y_cp)), 12)
        z = np.round(np.concatenate((z_b, z_cp)), 12)
        ql = np.concatenate([df_b_mapped.ql_mean.values, df_b_mapped_mirror.ql_mean.values, df_cp_mapped.ql_mean.values, df_cp_mapped_mirror.ql_mean.values])
        tilts = np.concatenate([df_b_mapped.tilt.values, df_b_mapped_mirror.tilt.values, df_cp_mapped.tilt.values, df_cp_mapped_mirror.tilt.values])

        # remove repeated points
        xyz = np.array(zip(x, y, z))
        xyz_u, indices = np.unique(xyz, axis=0, return_index=True)
        ql = ql[indices]
        tilts = tilts[indices]
        x, y, z = xyz_u.T

        # make mesh
        t = np.linspace(0, np.pi, 500)
        p = np.linspace(0, 2*np.pi, 500)
        t_mesh, p_mesh = np.meshgrid(t, p)
        x_mesh, y_mesh, z_mesh = polar_to_cartesian(t_mesh, p_mesh, b_up, cp_up)

        #funcs = ['multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate'] # multiquadric, inverse, cubic, and thin_plate all look good
        funcs = ['cubic']
        for func in funcs:
            rbf_interp = scipy.interpolate.Rbf(x, y, z, ql, function=func, norm=sph_norm, smooth=0.) # smooth doesn't really make it better

            # use dask to chunk and thread data to limit memory usage
            n1 = x_mesh.shape[1]
            ix = da.from_array(x_mesh, chunks=(1, n1))
            iy = da.from_array(y_mesh, chunks=(1, n1))
            iz = da.from_array(z_mesh, chunks=(1, n1))
            iq = da.map_blocks(rbf_interp, ix, iy, iz)
            ql_int = iq.compute()

            #for xq, yq, zq, qq in zip(x_mesh, y_mesh, z_mesh, ql_int):
            #    print xq, yq, zq, qq

            fig = mlab.figure(size=(400*2, 350*2)) 
            pts = mlab.points3d(x, y, z, ql, colormap='viridis', scale_mode='none', scale_factor=0.03)
            mesh = mlab.mesh(x_mesh, y_mesh, z_mesh, scalars=ql_int, colormap='viridis')
            # wireframe
            #mlab.mesh(x_mesh, y_mesh, z_mesh, color=(0.5, 0.5, 0.5), representation='wireframe')
            mlab.axes(pts, xlabel='a', ylabel='b', zlabel='c\'')
            mlab.colorbar(mesh, orientation='vertical')
        mlab.show()

def sph_harm_fit(fin1, fin2, dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, beam_11MeV, plot_pulse_shape, multiplot, save_multiplot):
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
        ql, ql_uncert, theta, phi, names, order_coeff = args

        pars = fit_params.valuesdict()
        harmonics = 0
        for name, oc in zip(names, order_coeff):
            c = pars[name]
            harmonics += c*sph_harm(oc[1], oc[0], theta, phi)

        harmonics = harmonics.real
        return (ql - harmonics)/ql_uncert

    data_bvert = pd_load(fin1, p_dir)
    data_bvert = split_filenames(data_bvert)
    data_cpvert = pd_load(fin2, p_dir)
    data_cpvert = split_filenames(data_cpvert)

    for d, det in enumerate(dets):       
        #if d > 0 :
        #    continue
        print '\ndet_no =', det, 'theta_n =', theta_n[d]
        df_b_mapped = map_data_3d(data_bvert, det, bvert_tilt, b_up, theta_n[d], phi_n[d], beam_11MeV) 
        df_b_mapped_mirror = map_data_3d(data_bvert, det, bvert_tilt, np.asarray(((1,0,0), (0,1,0), (0,0,-1))), theta_n[d], phi_n[d], beam_11MeV)
        df_cp_mapped = map_data_3d(data_cpvert, det, cpvert_tilt, cp_up, theta_n[d], phi_n[d], beam_11MeV)
        df_cp_mapped_mirror = map_data_3d(data_cpvert, det, cpvert_tilt, np.asarray(((-1,0,0), (0,0,-1), (0,1,0))), theta_n[d], phi_n[d], beam_11MeV)

        # convert to proper frame (b and cp orientations are different)
        theta_b = np.concatenate([df_b_mapped.theta.values, df_b_mapped_mirror.theta.values])
        theta_cp = np.concatenate([df_cp_mapped.theta.values, df_cp_mapped_mirror.theta.values])
        phi_b = np.concatenate([df_b_mapped.phi.values, df_b_mapped_mirror.phi.values])
        phi_cp = np.concatenate([df_cp_mapped.phi.values, df_cp_mapped_mirror.phi.values])

        x_b, y_b, z_b = polar_to_cartesian(theta_b, phi_b, b_up, cp_up)
        x_cp, y_cp, z_cp = polar_to_cartesian(theta_cp, phi_cp, cp_up, cp_up)

        x = np.concatenate((x_b, x_cp))
        y = np.concatenate((y_b, y_cp))
        z = np.concatenate((z_b, z_cp))
        ql = np.concatenate([df_b_mapped.ql_mean.values, df_b_mapped_mirror.ql_mean.values, df_cp_mapped.ql_mean.values, df_cp_mapped_mirror.ql_mean.values])
        ql_uncert = np.concatenate([df_b_mapped.ql_abs_uncert.values, df_b_mapped_mirror.ql_abs_uncert.values, df_cp_mapped.ql_abs_uncert.values, 
                                    df_cp_mapped_mirror.ql_abs_uncert.values])
        tilts = np.concatenate([df_b_mapped.tilt.values, df_b_mapped_mirror.tilt.values, df_cp_mapped.tilt.values, df_cp_mapped_mirror.tilt.values])

        ## remove repeated points
        xyz = np.array(zip(x, y, z))
        xyz_u, indices = np.unique(xyz, axis=0, return_index=True)
        ql = ql[indices]
        ql_uncert = ql_uncert[indices]
        tilts = tilts[indices]
        x, y, z = xyz_u.T

        ## convert back to spheical coords for sph harmonics fitting
        theta = np.arccos(z)
        phi = []
        for a, b in zip(x, y):
            if a < 0 and b >= 0:
                p = np.arctan(b/a) + np.pi
            elif a < 0 and b < 0:
                p = np.arctan(b/a) - np.pi
            elif a == 0 and b > 0:
                p = np.pi/2.
            elif a == 0 and b < 0:
                p = -np.pi/2. 
            else:
                p = np.arctan(b/a)
            phi.append(p)
        phi = np.array(phi)    
        ## check for nan in phi
        nans = np.argwhere(np.isnan(phi))
        for nan in nans:
            phi[nan] = phi[nan-1]

        # lmfit
        for i in range(7, 8):
            order = i
            names, order_coeff = get_coeff_names(order, central_idx_only=False)
            fit_params = add_params(names)
            fit_kws={'nan_policy': 'omit'}
            res = lmfit.minimize(minimize, fit_params, args=(ql, ql_uncert, theta, phi, names, order_coeff), **fit_kws)
            print '\n', res.message
            print lmfit.fit_report(res, show_correl=False)

            sph_harmonics = 0
            for idx, (name, par) in enumerate(res.params.items()):
                sph_harmonics += par.value*sph_harm(order_coeff[idx][1], order_coeff[idx][0], theta, phi)

            # plot
            fig = mlab.figure(size=(400*2, 350*2)) 
            fig.scene.disable_render = True
            pts = mlab.points3d(x, y, z, sph_harmonics.real, colormap='viridis', scale_mode='none', scale_factor=0.03)

            ## delaunay triagulation (mesh, interpolation)
            tri = mlab.pipeline.delaunay3d(pts)
            edges = mlab.pipeline.extract_edges(tri)
            #edges = mlab.pipeline.surface(edges, colormap='viridis')

            tri_smooth = mlab.pipeline.poly_data_normals(tri) # smooths delaunay triangulation mesh
            surf = mlab.pipeline.surface(tri_smooth, colormap='viridis')
            
            #for x_val, y_val, z_val, ql_val, tilt in zip(x, y, z, sph_harmonics, tilts):
            #    #mlab.text3d(x_val, y_val, z_val, str(ql_val), scale=0.03, color=(0,0,0), figure=fig)
            #    mlab.text3d(x_val, y_val, z_val, str(tilt), scale=0.03, color=(0,0,0), figure=fig) 

            mlab.axes(pts, xlabel='a', ylabel='b', zlabel='c\'')
            mlab.colorbar(surf, orientation='vertical') 
            mlab.view(azimuth=0, elevation=90, distance=7.5, figure=fig)  
            mlab.title('Order =' + str(i))
            fig.scene.disable_render = False   
        mlab.show()

def lsq_sph_biv_spl(fin1, fin2, dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, beam_11MeV, plot_pulse_shape, multiplot, save_multiplot):
    ''' Implementations of SmoothSphereBivriateSpline and LSQSphereBivariateSpline
        Note - they are the same if weights are not used with lsq
        Current method used SmoothSphere
        Adjust fitted grid and s parameter to change smoothing
        Note: has issues with fitting values less than 1.0 (11MeV 20 and 30 deg, 4 MeV - most)
    '''
    from scipy.interpolate import LSQSphereBivariateSpline as lsqsbs
    from scipy.interpolate import SmoothSphereBivariateSpline as ssbs

    data_bvert = pd_load(fin1, p_dir)
    data_bvert = split_filenames(data_bvert)
    data_cpvert = pd_load(fin2, p_dir)
    data_cpvert = split_filenames(data_cpvert)

    for d, det in enumerate(dets):       
        print '\ndet_no =', det, 'theta_n =', theta_n[d]
        df_b_mapped = map_data_3d(data_bvert, det, bvert_tilt, b_up, theta_n[d], phi_n[d], beam_11MeV) 
        df_b_mapped_mirror = map_data_3d(data_bvert, det, bvert_tilt, np.asarray(((1,0,0), (0,1,0), (0,0,-1))), theta_n[d], phi_n[d], beam_11MeV)
        df_cp_mapped = map_data_3d(data_cpvert, det, cpvert_tilt, cp_up, theta_n[d], phi_n[d], beam_11MeV)
        df_cp_mapped_mirror = map_data_3d(data_cpvert, det, cpvert_tilt, np.asarray(((-1,0,0), (0,0,-1), (0,1,0))), theta_n[d], phi_n[d], beam_11MeV)

        # convert to proper frame (b and cp orientations are different)
        theta_b = np.concatenate([df_b_mapped.theta.values, df_b_mapped_mirror.theta.values])
        theta_cp = np.concatenate([df_cp_mapped.theta.values, df_cp_mapped_mirror.theta.values])
        phi_b = np.concatenate([df_b_mapped.phi.values, df_b_mapped_mirror.phi.values])
        phi_cp = np.concatenate([df_cp_mapped.phi.values, df_cp_mapped_mirror.phi.values])

        x_b, y_b, z_b = polar_to_cartesian(theta_b, phi_b, b_up, cp_up)
        x_cp, y_cp, z_cp = polar_to_cartesian(theta_cp, phi_cp, cp_up, cp_up)

        x = np.concatenate((x_b, x_cp))
        y = np.concatenate((y_b, y_cp))
        z = np.concatenate((z_b, z_cp))
        ql = np.concatenate([df_b_mapped.ql_mean.values, df_b_mapped_mirror.ql_mean.values, df_cp_mapped.ql_mean.values, df_cp_mapped_mirror.ql_mean.values])
        tilts = np.concatenate([df_b_mapped.tilt.values, df_b_mapped_mirror.tilt.values, df_cp_mapped.tilt.values, df_cp_mapped_mirror.tilt.values])

        ## remove repeated points
        xyz = np.array(zip(x, y, z))
        xyz_u, indices = np.unique(xyz, axis=0, return_index=True)
        ql = ql[indices]
        tilts = tilts[indices]
        x, y, z = xyz_u.T

        ## convert back to spheical coords for sph harmonics fitting
        theta = np.arccos(z)
        phi = []
        for a, b in zip(x, y):
            if a < 0 and b >= 0:
                p = np.arctan(b/a) + np.pi
            elif a < 0 and b < 0:
                p = np.arctan(b/a) - np.pi
            elif a == 0 and b > 0:
                p = np.pi/2.
            elif a == 0 and b < 0:
                p = -np.pi/2. 
            else:
                p = np.arctan(b/a)
            phi.append(p)
        phi = np.array(phi)     

        ## check for nan in phi
        nans = np.argwhere(np.isnan(phi))
        for nan in nans:
            phi[nan] = phi[nan-1]

        phi = phi + np.pi           

        # interpolate 
        ## set interpolator object with mesh 
        points = 50
        knotst = np.linspace(0, np.pi, points)
        knotsp = np.linspace(0, 2*np.pi, points)
        lats, lons = np.meshgrid(knotst, knotsp)

        knotst[0] += 0.0001
        knotst[-1] -= 0.00001
        knotsp[0] += 0.00001
        knotsp[-1] -= 0.00001

        lut = ssbs(theta, phi, ql, s=1.5)

        x_fine = np.sin(lats) * np.cos(lons)
        y_fine = np.sin(lats) * np.sin(lons)
        z_fine = np.cos(lats)

        ql_new = []
        for t, p in zip(lats.ravel(), lons.ravel()):
            ql_new.append(lut(t, p)[0][0])
        ql_new = np.reshape(np.array(ql_new), (points, points))

        # plot
        fig = mlab.figure(size=(400*2, 350*2)) 
        # fit
        pts = mlab.points3d(x_fine.ravel(), y_fine.ravel(), z_fine.ravel(), ql_new.ravel(), colormap='viridis', scale_mode='none', scale_factor=0.0)
        # measured data
        mlab.points3d(x, y, z, ql, colormap='viridis', scale_mode='none', scale_factor=0.03)
        tri = mlab.pipeline.delaunay3d(pts)
        edges = mlab.pipeline.extract_edges(tri)

        tri_smooth = mlab.pipeline.poly_data_normals(tri) # smooths delaunay triangulation mesh
        surf = mlab.pipeline.surface(tri_smooth, colormap='viridis')
        
        #for x_val, y_val, z_val, ql_val, th, ph in zip(x_fine.ravel(), y_fine.ravel(), z_fine.ravel(), ql_new.ravel(), lats.ravel(), lons.ravel()):
        #    if th==0 or ph==0:
        #        mlab.text3d(x_val, y_val, z_val, str(round(ql_val,3)), scale=0.03, color=(0,0,0), figure=fig)
        #        #mlab.text3d(x_val, y_val, z_val, str(round(th, 3)) + ', ' + str(round(ph, 3)), scale=0.03, color=(0,0,0), figure=fig)

        mlab.axes(pts, xlabel='a', ylabel='b', zlabel='c\'')
        mlab.colorbar(surf, orientation='vertical') 

        # lsqsbs implementation
        #lut = lsqsbs(lats.ravel(), lons.ravel(), ql_new.ravel(), knotst, knotsp)
        #fig = mlab.figure(size=(400*2, 350*2)) 
        ## fit
        #pts = mlab.points3d(x_fine.ravel(), y_fine.ravel(), z_fine.ravel(), ql_new.ravel(), colormap='viridis', scale_mode='none', scale_factor=0.0)
        ## measured data
        #mlab.points3d(x, y, z, ql, colormap='viridis', scale_mode='none', scale_factor=0.03)
        #tri = mlab.pipeline.delaunay3d(pts)
        #edges = mlab.pipeline.extract_edges(tri)

        #tri_smooth = mlab.pipeline.poly_data_normals(tri) # smooths delaunay triangulation mesh
        #surf = mlab.pipeline.surface(tri_smooth, colormap='viridis')
        #
        ##for x_val, y_val, z_val, ql_val, th, ph in zip(x_fine.ravel(), y_fine.ravel(), z_fine.ravel(), ql_new.ravel(), lats.ravel(), lons.ravel()):
        ##    if th==0 or ph==0:
        ##        mlab.text3d(x_val, y_val, z_val, str(round(ql_val,3)), scale=0.03, color=(0,0,0), figure=fig)
        ##        #mlab.text3d(x_val, y_val, z_val, str(round(th, 3)) + ', ' + str(round(ph, 3)), scale=0.03, color=(0,0,0), figure=fig)

        #mlab.axes(pts, xlabel='a', ylabel='b', zlabel='c\'')
        #mlab.colorbar(surf, orientation='vertical') 
        mlab.show()

def legendre_poly_fit(fin1, fin2, dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, beam_11MeV, plot_pulse_shape, multiplot, save_multiplot):
    ''' Note: abandoned because it does not reproduce max and min LO well (i.e. the LO rations are underpredicted - > 5% and worse for low energy scatters)'''

    from scipy.special import lpmv 

    def new_cal(lo, m, b, new_m, new_b):
        y = m*np.array(lo) + b
        return (y - new_b)/new_m

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
                    order_coeff.append((o, i - idx)) # n, m
            return names, order_coeff

    def add_params(names_t, names_p):
        # create parameter argument for lmfit
        fit_params = lmfit.Parameters()
        for t, p in zip(names_t, names_p):
            fit_params.add(t, value=0.5)
            fit_params.add(p, value=0.5)
        return fit_params

    def minimize(fit_params, *args):
        ql, ql_uncert, theta, phi, names_t, names_p, order_coeff = args

        pars = fit_params.valuesdict()
        legendre_t, legendre_p = 0, 0
        for name_t, name_p, oc in zip(names_t, names_p, order_coeff):
            ct = pars[name_t]
            cp = pars[name_p]
            legendre_t += ct*lpmv(oc[1], oc[0], np.cos(theta))  # oc[0] = n (degree n>=0), oc[1] = m (order |m| <= n)
            legendre_p += cp*lpmv(oc[1], oc[0], np.sin(phi)) # real value of spherical hamonics

        legendre = legendre_t * legendre_p
        return (ql - legendre)**2/ql_uncert**2

    data_bvert = pd_load(fin1, p_dir)
    data_bvert = split_filenames(data_bvert)
    data_cpvert = pd_load(fin2, p_dir)
    data_cpvert = split_filenames(data_cpvert)

    for d, det in enumerate(dets):       
        #if d > 0 :
        #    continue
        print '\ndet_no =', det, 'theta_n =', theta_n[d]
        df_b_mapped = map_data_3d(data_bvert, det, bvert_tilt, b_up, theta_n[d], phi_n[d], beam_11MeV) 
        df_b_mapped_mirror = map_data_3d(data_bvert, det, bvert_tilt, np.asarray(((1,0,0), (0,1,0), (0,0,-1))), theta_n[d], phi_n[d], beam_11MeV)
        df_cp_mapped = map_data_3d(data_cpvert, det, cpvert_tilt, cp_up, theta_n[d], phi_n[d], beam_11MeV)
        df_cp_mapped_mirror = map_data_3d(data_cpvert, det, cpvert_tilt, np.asarray(((-1,0,0), (0,0,-1), (0,1,0))), theta_n[d], phi_n[d], beam_11MeV)

        # convert to proper frame (b and cp orientations are different)
        theta_b = np.concatenate([df_b_mapped.theta.values, df_b_mapped_mirror.theta.values])
        theta_cp = np.concatenate([df_cp_mapped.theta.values, df_cp_mapped_mirror.theta.values])
        phi_b = np.concatenate([df_b_mapped.phi.values, df_b_mapped_mirror.phi.values])
        phi_cp = np.concatenate([df_cp_mapped.phi.values, df_cp_mapped_mirror.phi.values])

        x_b, y_b, z_b = polar_to_cartesian(theta_b, phi_b, b_up, cp_up)
        x_cp, y_cp, z_cp = polar_to_cartesian(theta_cp, phi_cp, cp_up, cp_up)

        x = np.concatenate((x_b, x_cp))
        y = np.concatenate((y_b, y_cp))
        z = np.concatenate((z_b, z_cp))
        ql = np.concatenate([df_b_mapped.ql_mean.values, df_b_mapped_mirror.ql_mean.values, df_cp_mapped.ql_mean.values, df_cp_mapped_mirror.ql_mean.values])
        ql_uncert = np.concatenate([df_b_mapped.ql_abs_uncert.values, df_b_mapped_mirror.ql_abs_uncert.values, df_cp_mapped.ql_abs_uncert.values, 
                                    df_cp_mapped_mirror.ql_abs_uncert.values])
        tilts = np.concatenate([df_b_mapped.tilt.values, df_b_mapped_mirror.tilt.values, df_cp_mapped.tilt.values, df_cp_mapped_mirror.tilt.values])

        ## remove repeated points
        xyz = np.array(zip(x, y, z))
        xyz_u, indices = np.unique(xyz, axis=0, return_index=True)
        ql = ql[indices]
        ql_uncert = ql_uncert[indices]
        tilts = tilts[indices]
        x, y, z = xyz_u.T

        ## convert back to spheical coords for sph harmonics fitting
        theta = np.arccos(z)
        phi = []
        for a, b in zip(x, y):
            #if a < 0 and b >= 0:
            #    p = np.arctan(b/a) + np.pi
            #    #print p
            #elif a < 0 and b < 0:
            #    p = np.arctan(b/a) - np.pi
            #elif a == 0 and b > 0:
            #    p = np.pi/2.
            #elif a == 0 and b < 0:
            #    p = -np.pi/2. 
            if (abs(a) < 1e-5 and abs(b) < 1e-5):
                print a, b
                p = 0.
            else:
                p = np.arctan(b/a) 
            phi.append(p)
        phi = np.array(phi)   

        # lmfit
        for i in range(2, 3):
            order = i
            # generate coefficients for phi and theta terms
            names_t, order_coeff_t = get_coeff_names(order, central_idx_only=True)
            names_p, order_coeff_p = get_coeff_names(order, central_idx_only=True)
            fit_params = add_params(names_t, names_p)
            #fit_kws={'nan_policy': 'omit'}
            res = lmfit.minimize(minimize, fit_params, args=(ql, ql_uncert, theta, phi, names_t, names_p, order_coeff_t))
            print '\n', res.message
            print lmfit.fit_report(res, show_correl=False)

            leg_poly_t, leg_poly_p = 0, 0
            for idx, (name, par) in enumerate(res.params.items()):
                leg_poly_t += par.value*lpmv(order_coeff_t[idx][1], order_coeff_t[idx][0], np.cos(theta))
                leg_poly_p += par.value*lpmv(order_coeff_t[idx][1], order_coeff_t[idx][0], np.sin(phi))
                #print leg_poly_t
            legendre_poly_fit = leg_poly_t * leg_poly_p
            ql_idx = np.argsort(ql)
            legendre_poly_sorted = sorted(legendre_poly_fit)
            for l, ll, q in zip(legendre_poly_fit, sorted(legendre_poly_fit), sorted(ql)):
                #print l, ll
                if q == max(sorted(ql)):
                    print l, q
                if q == min(sorted(ql)):
                    print l, q
                #print l, q

            #for i, (q, l, ll, lll) in enumerate(zip(ql_idx, legendre_poly_fit, legendre_poly_fit[ql_idx], legendre_poly_sorted)):
            #    if i==0:
            #        print i, l, ll, lll, q 
            #        continue
            #    print i, l, ll, lll, q, q-ql_idx[i-1]

            # plot
            fig = mlab.figure(size=(400*2, 350*2)) 
            fig.scene.disable_render = True
            pts = mlab.points3d(x[ql_idx], y[ql_idx], z[ql_idx], legendre_poly_sorted, colormap='viridis', scale_mode='none', scale_factor=0.03)

            ## delaunay triagulation (mesh, interpolation)
            tri = mlab.pipeline.delaunay3d(pts)
            edges = mlab.pipeline.extract_edges(tri)
            #edges = mlab.pipeline.surface(edges, colormap='viridis')

            tri_smooth = mlab.pipeline.poly_data_normals(tri) # smooths delaunay triangulation mesh
            surf = mlab.pipeline.surface(tri_smooth, colormap='viridis')
            
            for x_val, y_val, z_val, ql_val, ql_meas, tilt, p, t in zip(x, y, z, legendre_poly_fit, ql, tilts, phi, theta):
                #mlab.text3d(x_val, y_val, z_val, str(ql_val), scale=0.03, color=(0,0,0), figure=fig)
                #mlab.text3d(x_val, y_val, z_val, str(round(t, 5)) + ' ' + str(round(p, 5)), scale=0.03, color=(0,0,0), figure=fig) 
                if abs(y_val) < 1e-5 or abs(z_val) < 1e-5:
                    mlab.text3d(x_val, y_val, z_val, str(round(ql_val, 5)) + ' ' + str(round(ql_meas, 5)), scale=0.03, color=(0,0,0), figure=fig)
                    #mlab.text3d(x_val, y_val, z_val, str(round(t, 5)) + ' ' + str(round(p, 5)), scale=0.03, color=(0,0,0), figure=fig)
                    #mlab.text3d(x_val, y_val, z_val, str(round(x_val, 10)) + '\n ' + str(round(y_val, 10)) + '\n ' + str(round(z_val, 5)) + ' ' + str(round(ql_val, 5)) 
                    #            +' '+str(round(t, 5)) + ' ' + str(round(p, 5)), scale=0.03, color=(0,0,0), figure=fig) 
                
            mlab.axes(pts, xlabel='a', ylabel='b', zlabel='c\'')
            mlab.colorbar(surf, orientation='vertical') 
            mlab.view(azimuth=0, elevation=90, distance=7.5, figure=fig)  
            mlab.title('Order =' + str(i))
            fig.scene.disable_render = False   


            # TESTS - tried to get the fit to work better, unsuccessful
            #mesh_theta, mesh_phi = np.mgrid[0.0001:np.pi-0.0001:200j, 0.0001:2*np.pi-0.0001:200j]
            new_theta = np.linspace(0, np.pi, 500)
            new_phi = np.linspace(0, 2*np.pi, 500)
            mesh_theta, mesh_phi = np.meshgrid(new_theta, new_phi)
            #mesh_x = np.cos(mesh_phi)
            #mesh_y = np.sin(mesh_phi)*np.sin(mesh_theta)
            #mesh_z = np.sin(mesh_phi)*np.cos(mesh_theta)
 
            mesh_z = np.cos(mesh_theta)
            mesh_y = np.sin(mesh_theta)*np.sin(mesh_phi)
            mesh_x = np.sin(mesh_theta)*np.cos(mesh_phi)

            leg_poly_t, leg_poly_p = 0, 0
            for idx, (name, par) in enumerate(res.params.items()):
                leg_poly_t += par.value*lpmv(order_coeff_t[idx][1], order_coeff_t[idx][0], np.cos(mesh_theta))
                leg_poly_p += par.value*lpmv(order_coeff_t[idx][1], order_coeff_t[idx][0], np.sin(mesh_phi))
            legendre_poly_fit = leg_poly_t * leg_poly_p

            fig = mlab.figure(size=(400*2, 350*2)) 
            pts = mlab.points3d(x[ql_idx], y[ql_idx], z[ql_idx], legendre_poly_sorted, colormap='viridis', scale_mode='none', scale_factor=0.03)

            # sort legendre output
            #legendre_poly_fit = np.ravel(legendre_poly_fit)
            #leg_idx = np.argsort(legendre_poly_fit)
            #legendre_poly_sorted = np.reshape(sorted(legendre_poly_fit), (15, 15))
            #mesh_x = np.reshape(np.ravel(mesh_x)[leg_idx], (200, 200))
            #mesh_y = np.reshape(np.ravel(mesh_y)[leg_idx], (200, 200))
            #mesh_z = np.reshape(np.ravel(mesh_z)[leg_idx], (200, 200))

            #mesh = mlab.mesh(mesh_x, mesh_y, mesh_z, scalars=legendre_poly_fit, colormap='viridis')
            #new_x = np.cos(new_phi)
            #new_y = np.sin(new_phi)*np.sin(new_theta)
            #new_z = np.sin(new_phi)*np.cos(new_theta)

            new_z = np.cos(new_theta)
            new_y = np.sin(new_theta)*np.sin(new_phi)
            new_x = np.sin(new_theta)*np.cos(new_phi)

            new_x = np.ravel(mesh_x)
            new_y = np.ravel(mesh_y)
            new_z = np.ravel(mesh_z)
            new_theta = np.ravel(mesh_theta)
            new_phi = np.ravel(mesh_phi)
            legendre_poly_fit = np.ravel(legendre_poly_fit)

            #for mx, my, mz, mtheta, mphi, leg in zip(new_x, new_y, new_z, new_theta, new_phi, legendre_poly_fit):
            #    #if mz == 1.:
            #    #print '{:^8.5f} {:>8.5f} {:>8.5f} {:^8.5f} {:>8.5f} {:>8.5f}'.format(mx, my, mz, mtheta, mphi, leg)
            #    mlab.text3d(mx, my, mz, str(round(mtheta, 3)) + ' ' + str(round(mphi, 3)), scale=0.03, color=(0,0,0), figure=fig)
            #print min(legendre_poly_fit), max(legendre_poly_fit)


            leg_poly_t, leg_poly_p = 0, 0
            for idx, (name, par) in enumerate(res.params.items()):
                leg_poly_t += par.value*lpmv(order_coeff_t[idx][1], order_coeff_t[idx][0], np.cos(new_theta))
                leg_poly_p += par.value*lpmv(order_coeff_t[idx][1], order_coeff_t[idx][0], np.sin(new_phi))
            legendre_poly_fit = leg_poly_t * leg_poly_p

            #fig = mlab.figure(size=(400*2, 350*2)) 
            #fig.scene.disable_render = True
            #pts = mlab.points3d(new_x, new_y, new_z, legendre_poly_fit, colormap='viridis', scale_mode='none', scale_factor=0.03)

            ### delaunay triagulation (mesh, interpolation)
            #tri = mlab.pipeline.delaunay3d(pts)
            #edges = mlab.pipeline.extract_edges(tri)
            ##edges = mlab.pipeline.surface(edges, colormap='viridis')

            #tri_smooth = mlab.pipeline.poly_data_normals(tri) # smooths delaunay triangulation mesh
            #surf = mlab.pipeline.surface(tri_smooth, colormap='viridis')

            print 'ql_max   ql_min   ratio    fit_max     fit_min     ratio'
            print max(ql), min(ql), max(ql)/min(ql), max(legendre_poly_fit), min(legendre_poly_fit), max(legendre_poly_fit)/min(legendre_poly_fit) 

        mlab.show()

def lambertian(fin1, fin2, dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, beam_11MeV, pulse_shape):
    ''' 2D lambertian projection of the 3D spherical data
        Interpolation currently performed with griddata built in methods (linear (current), cubic, nearest)
    '''
    # ignore divide by zero warning
    np.seterr(divide='ignore', invalid='ignore')

    avg_uncerts, avg_qls, abs_uncerts = get_avg_lo_uncert(fin1, fin2, p_dir, dets, beam_11MeV, pulse_shape) 

    data_bvert = pd_load(fin1, p_dir)
    data_bvert = split_filenames(data_bvert)
    data_cpvert = pd_load(fin2, p_dir)
    data_cpvert = split_filenames(data_cpvert)

    print '\n det   angle   mean_ql  uncert   abs_unc rel_uncert'
    for d, det in enumerate(dets):       
        df_b_mapped = map_data_3d(data_bvert, det, bvert_tilt, b_up, theta_n[d], phi_n[d], beam_11MeV) 
        df_b_mapped_mirror = map_data_3d(data_bvert, det, bvert_tilt, np.asarray(((1,0,0), (0,1,0), (0,0,-1))), theta_n[d], phi_n[d], beam_11MeV)
        df_cp_mapped = map_data_3d(data_cpvert, det, cpvert_tilt, cp_up, theta_n[d], phi_n[d], beam_11MeV)
        df_cp_mapped_mirror = map_data_3d(data_cpvert, det, cpvert_tilt, np.asarray(((-1,0,0), (0,0,-1), (0,1,0))), theta_n[d], phi_n[d], beam_11MeV)

        # convert to proper frame (b and cp orientations are different)
        theta_b = np.concatenate([df_b_mapped.theta.values, df_b_mapped_mirror.theta.values])
        theta_cp = np.concatenate([df_cp_mapped.theta.values, df_cp_mapped_mirror.theta.values])
        phi_b = np.concatenate([df_b_mapped.phi.values, df_b_mapped_mirror.phi.values])
        phi_cp = np.concatenate([df_cp_mapped.phi.values, df_cp_mapped_mirror.phi.values])

        x_b, y_b, z_b = polar_to_cartesian(theta_b, phi_b, b_up, cp_up)
        x_cp, y_cp, z_cp = polar_to_cartesian(theta_cp, phi_cp, cp_up, cp_up)

        x = np.concatenate((x_b, x_cp))
        y = np.concatenate((y_b, y_cp))
        z = np.concatenate((z_b, z_cp))
        ql = np.concatenate([df_b_mapped.ql_mean.values, df_b_mapped_mirror.ql_mean.values, df_cp_mapped.ql_mean.values, df_cp_mapped_mirror.ql_mean.values])
        ql_uncert = np.concatenate([df_b_mapped.ql_abs_uncert.values, df_b_mapped_mirror.ql_abs_uncert.values, df_cp_mapped.ql_abs_uncert.values, 
                                    df_cp_mapped_mirror.ql_abs_uncert.values])
        tilts = np.concatenate([df_b_mapped.tilt.values, df_b_mapped_mirror.tilt.values, df_cp_mapped.tilt.values, df_cp_mapped_mirror.tilt.values])

        ## remove repeated points
        xyz = np.array(zip(x, y, z))
        xyz_u, indices = np.unique(xyz, axis=0, return_index=True)
        ql = ql[indices]
        ql_uncert = ql_uncert[indices]
        tilts = tilts[indices]
        x, y, z = xyz_u.T
        if pulse_shape:
            qs = np.concatenate([df_b_mapped.qs_mean.values, df_b_mapped_mirror.qs_mean.values, df_cp_mapped.qs_mean.values, df_cp_mapped_mirror.qs_mean.values])
            qs = qs[indices]
            ps = [1 - a/b for a, b in zip(qs, ql)]

        # convert to lambertian projection (from https://en.wikipedia.org/wiki/Lambert_azimuthal_equal-area_projection)
        X, Y = [], []
        for xi, yi, zi in zip(x, y, z):
            Xi = np.sqrt(2/(1-zi))*xi
            Yi = np.sqrt(2/(1-zi))*yi
            if np.isnan(Xi) or np.isnan(Yi):
                zi -= 0.0001
                Xi = np.sqrt(2/(1-zi))*xi
                Yi = np.sqrt(2/(1-zi))*yi    
            X.append(Xi)
            Y.append(Yi)
        X = np.array(X)/max(X)
        Y = np.array(Y)/max(Y)

        #print np.mean(ql_uncert), np.std(ql_uncert)
        grid_x, grid_y = np.mgrid[-1:1:2000j, -1:1:2000j]
        #methods = ('nearest', 'linear', 'cubic')
        methods = ('linear',)
        plt.rcParams['axes.facecolor'] = 'grey'
        f = 14
        for method in methods:
            if pulse_shape:
                interp = scipy.interpolate.griddata((X, Y), ps, (grid_x, grid_y), method=method)
                #print max(ps), min(ps), max(ps)/min(ps)
                plt.figure()
                plt.imshow(interp.T, extent=(-1,1,-1,1), origin='lower', cmap='viridis', interpolation='none')
                plt.scatter(X, Y, c=ps, cmap='viridis')
                # put avg uncert on colorbar
                ## need to scale y from (0, 1) to (min(ps), max(ps))
                avg_uncert = avg_uncerts[d]/(max(ps) - min(ps))
                abs_uncert = abs_uncerts[d]/(max(ps) - min(ps))
                cbar = plt.colorbar()
                cbar.ax.errorbar(0.5, 0.5, yerr=avg_uncert)
                cbar.ax.errorbar(0.5, 0.5, yerr=abs_uncert, ecolor='r', elinewidth=1.5)
                print '{:^5d} {:>6.1f} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f}%'.format(det, theta_n[d], np.mean(ps), avg_uncerts[d], abs_uncerts[d], avg_uncerts[d]/np.mean(ps)*100)

            else:
                interp = scipy.interpolate.griddata((X, Y), ql, (grid_x, grid_y), method=method)
                #print max(ql), min(ql), max(ql)/min(ql)
                plt.figure()
                plt.imshow(interp.T, extent=(-1,1,-1,1), origin='lower', cmap='viridis', interpolation='none')
                plt.scatter(X, Y, c=ql, cmap='viridis')
                # put avg uncert on colorbar
                ## need to scale y from (0, 1) to (min(ql), max(ql))
                avg_uncert = avg_uncerts[d]/(max(ql) - min(ql))
                abs_uncert = abs_uncerts[d]/(max(ql) - min(ql))
                cbar = plt.colorbar()
                cbar.ax.errorbar(0.5, 0.5, yerr=avg_uncert)
                cbar.ax.errorbar(0.5, 0.5, yerr=abs_uncert, ecolor='r', elinewidth=1.5)
                print '{:^5d} {:>6.1f} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f}%'.format(det, theta_n[d], np.mean(ql), avg_uncerts[d], abs_uncerts[d], avg_uncerts[d]/np.mean(ql)*100)
            
            plt.text(-0.71, 0.0, 'a', color='r', fontsize=f)
            plt.text(0.71, 0., 'a', color='r', fontsize=f)
            plt.text(0, 0.0, 'c\'', color='r', fontsize=f)
            plt.text(0, 0.713, 'b', color='r', fontsize=f)
            plt.text(0, -0.713, 'b', color='r', fontsize=f)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()

    plt.show()

def get_avg_lo_uncert(fin1, fin2, p_dir, dets, beam_11MeV, pulse_shape):

    data_bvert = pd_load(fin1, p_dir)
    data_bvert = split_filenames(data_bvert)
    data_cpvert = pd_load(fin2, p_dir)
    data_cpvert = split_filenames(data_cpvert)

    if beam_11MeV:
        # ps_unc - statistical uncertainty of 1-qs/ql (used cpvert uncert - worse than bvert)
        # from /home/radians/raweldon/tunl.2018.1_analysis/stilbene_final/peak_localization/pulse_shape_get_hotspots.py  
        ps_unc = [0.00175, 0.00135, 0.00125, 0.00126, 0.0014, 0.00198, 0.00195, 0.0014, 0.00124, 0.00123, 0.00134, 0.00177]
        # cal_unc - uncert due to change in calibration over experiment (used cpvert uncert - worse than bvert)
        # from /home/radians/raweldon/tunl.2018.1_analysis/stilbene_final/lo_calibration/uncert_gamma_cal.py 
        cal_unc = [0.00792, 0.00540, 0.00285, 0.00155, 0.00271, 0.00372, 0.00375, 0.00275, 0.00156, 0.00278, 0.00540, 0.00800]
    else:
        ps_unc = [0.00164, 0.00142, 0.00142, 0.00147, 0.0018, 0.00306, 0.0031, 0.00179, 0.00143, 0.00142, 0.00142, 0.0016]
        cal_unc = [0.01920, 0.01502, 0.01013, 0.00541, 0.00176, 0.00116, 0.00116, 0.00185, 0.00552, 0.01025, 0.01506, 0.01935]

    uncert, mean_ql, mean_ps = [], [], []
    for data in (data_bvert, data_cpvert):
        for d, det in enumerate(dets):
            if pulse_shape:
                det_df = data.loc[(data.det_no == str(det))]
                #print det, det_df.ql_abs_uncert.mean(), det_df.ql_abs_uncert.max(), det_df.ql_abs_uncert.median()
                uncert.append(ps_unc[d])
                #mean_ql.append(1 - (det_df.qs_mean.mean()/det_df.ql_mean.mean()))
                mean_ps.append(det_df.qs_mean.mean()/det_df.ql_mean.mean())
                mean_ql.append(det_df.ql_mean.mean())
            else:
                det_df = data.loc[(data.det_no == str(det))]
                #print det, det_df.ql_abs_uncert.mean(), det_df.ql_abs_uncert.max(), det_df.ql_abs_uncert.median()
                uncert.append(det_df.ql_abs_uncert.mean())
                mean_ql.append(det_df.ql_mean.mean())
    
    uncerts = [(x + y)/2 for x, y in zip(uncert[:len(dets)], uncert[len(dets):])]
    mean_qls = [(x + y)/2 for x, y in zip(mean_ql[:len(dets)], mean_ql[len(dets):])]
    mean_pss = [(x + y)/2 for x, y in zip(mean_ps[:len(dets)], mean_ps[len(dets):])]
    print mean_qls
    if pulse_shape:
        cal_unc = [c/q for c, q in zip(cal_unc, mean_qls)]
        print cal_unc
        cal_uncerts = [np.sqrt((c*q)**2 + u**2 + (0.005*q)**2) for c, q, u in zip(cal_unc, mean_pss, uncerts)]
        mean_pss= [1 - q for q in mean_qls]
        return cal_uncerts, mean_pss, uncerts

    else:
        cal_uncerts = [np.sqrt(c**2 + u**2 + (0.005*q)**2) for c, q, u in zip(cal_unc, mean_qls, uncerts)]
        #for i, j, k in zip(uncerts, cal_uncerts, mean_qls):
        #    print i, j, k, j/k*100, '%'
        return cal_uncerts, mean_qls, uncerts

def lambertian_smooth(fin1, fin2, fin, dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, beam_11MeV, pulse_shape):
    ''' 2D lambertian projection of the 3D spherical data
        Interpolation currently performed with griddata built in methods (linear (current), cubic, nearest)
    '''
    def map_smoothed_fitted_data_3d(data, det, tilts, crystal_orientation, theta_neutron, phi_neutron, beam_11MeV):
        # like map_data_3d but for fitted data
        det_df = data.loc[(data.det == det)]
        ql_all, theta_p, phi_p, angles_p = [], [], [], []
        for t, tilt in enumerate(tilts):
            
            tilt_df = det_df.loc[(data.tilt == tilt)]
            angles = np.arange(0, 180, 5) # 5 and 2 look good
      
            ql = sin_func(np.deg2rad(angles), tilt_df['a'].values, tilt_df['b'].values, tilt_df['phi'].values)

            #plt.figure(0)
            #plt.plot(angles, ql, 'o')
            
            thetap, phip = map_3d(tilt, crystal_orientation, angles, theta_neutron, phi_neutron)       
            ql_all.extend(ql)
            theta_p.extend(thetap)
            phi_p.extend(phip)
            angles_p.extend(angles)
    
        d = {'ql': ql_all, 'theta': theta_p, 'phi': phi_p, 'angles': angles_p}
        df = pd.DataFrame(data=d)
        return df

    # ignore divide by zero warning
    np.seterr(divide='ignore', invalid='ignore')
 
    avg_uncerts, avg_qls, abs_uncerts = get_avg_lo_uncert(fin[0], fin[1], p_dir, dets, beam_11MeV, pulse_shape) 
    if pulse_shape:
        f1 = fin1.split('.')
        fin1 = f1[0] + '_ps.' + f1[1]
        f2 = fin2.split('.')
        fin2 = f2[0] + '_ps.' + f2[1]

    data_bvert = pd_load(fin1, p_dir)
    data_cpvert = pd_load(fin2, p_dir)
    
    print '\n det   angle   mean_ql  uncert   abs_unc rel_uncert'
    for d, det in enumerate(dets):
        if d > 5:
            continue
        df_b_mapped = map_smoothed_fitted_data_3d(data_bvert, det, bvert_tilt, b_up, theta_n[d], phi_n[d], beam_11MeV)
        df_cp_mapped = map_smoothed_fitted_data_3d(data_cpvert, det, cpvert_tilt, cp_up, theta_n[d], phi_n[d], beam_11MeV)
        df_b_mapped_mirror = map_smoothed_fitted_data_3d(data_bvert, det, bvert_tilt, np.asarray(((1,0,0), (0,1,0), (0,0,-1))), theta_n[d], phi_n[d], beam_11MeV)
        df_cp_mapped_mirror = map_smoothed_fitted_data_3d(data_cpvert, det, cpvert_tilt, np.asarray(((-1,0,0), (0,0,-1), (0,1,0))), theta_n[d], phi_n[d], beam_11MeV)   

        # convert to cartesian
        theta_b = np.concatenate([df_b_mapped.theta.values, df_b_mapped_mirror.theta.values])
        theta_cp = np.concatenate([df_cp_mapped.theta.values, df_cp_mapped_mirror.theta.values])
        phi_b = np.concatenate([df_b_mapped.phi.values, df_b_mapped_mirror.phi.values])
        phi_cp = np.concatenate([df_cp_mapped.phi.values, df_cp_mapped_mirror.phi.values])

        angles_b = np.concatenate([df_b_mapped.angles.values, df_b_mapped_mirror.angles.values])
        angles_cp = np.concatenate([df_cp_mapped.angles.values, df_cp_mapped_mirror.angles.values])
        angles = np.concatenate((angles_b, angles_cp))

        x_b, y_b, z_b = polar_to_cartesian(theta_b, phi_b, b_up, cp_up)
        x_cp, y_cp, z_cp = polar_to_cartesian(theta_cp, phi_cp, cp_up, cp_up)

        x = np.round(np.concatenate((x_b, x_cp)), 12)
        y = np.round(np.concatenate((y_b, y_cp)), 12)
        z = np.round(np.concatenate((z_b, z_cp)), 12)
        ql = np.concatenate([df_b_mapped.ql.values, df_b_mapped_mirror.ql.values, df_cp_mapped.ql.values, df_cp_mapped_mirror.ql.values])
      
        ## remove repeated points
        xyz = np.array(zip(x, y, z))
        xyz_u, indices = np.unique(xyz, axis=0, return_index=True)
        ql = ql[indices]
        x, y, z = xyz_u.T

        # convert to lambertian projection (from https://en.wikipedia.org/wiki/Lambert_azimuthal_equal-area_projection)
        X, Y = [], []
        for xi, yi, zi in zip(x, y, z):
            Xi = np.sqrt(2/(1-zi))*xi
            Yi = np.sqrt(2/(1-zi))*yi
            if np.isnan(Xi) or np.isnan(Yi):
                zi -= 0.000001
                Xi = np.sqrt(2/(1-zi))*xi
                Yi = np.sqrt(2/(1-zi))*yi    
            if np.isinf(Xi) or np.isinf(Yi):
                zi -= 0.000001
                Xi = np.sqrt(2/(1-zi))*xi
                Yi = np.sqrt(2/(1-zi))*yi  
            X.append(Xi)
            Y.append(Yi)
        X = np.array(X)/max(X)
        Y = np.array(Y)/max(Y)

        #print np.mean(ql_uncert), np.std(ql_uncert)
        grid_x, grid_y = np.mgrid[-1:1:1000j, -1:1:1000j]
        #methods = ('nearest', 'linear', 'cubic')
        methods = ('linear',)
        plt.rcParams['axes.facecolor'] = 'grey'
        f = 14
        for method in methods:
            interp = scipy.interpolate.griddata((X, Y), ql, (grid_x, grid_y), method=method)
            #print max(ql), min(ql), max(ql)/min(ql)
            plt.figure()
            plt.imshow(interp.T, extent=(-1,1,-1,1), origin='lower', cmap='viridis', interpolation='none')
            #plt.scatter(X, Y, c=ql, cmap='viridis')
            # put avg uncert on colorbar
            ## need to scale y from (0, 1) to (min(ql), max(ql))
            avg_uncert = avg_uncerts[d]/(max(ql) - min(ql))
            abs_uncert = abs_uncerts[d]/(max(ql) - min(ql))
            cbar = plt.colorbar()
            cbar.ax.errorbar(0.5, 0.5, yerr=avg_uncert)
            cbar.ax.errorbar(0.5, 0.5, yerr=abs_uncert, ecolor='r', elinewidth=1.5)
            print '{:^5d} {:>6.1f} {:>8.3f} {:>8.3f} {:>8.3f} {:>8.3f}%'.format(det, theta_n[d], np.mean(ql), avg_uncerts[d], abs_uncerts[d], avg_uncerts[d]/np.mean(ql)*100)
            plt.text(-0.71, 0.0, 'a', color='r', fontsize=f)
            plt.text(0.71, 0., 'a', color='r', fontsize=f)
            plt.text(0, 0.0, 'c\'', color='r', fontsize=f)
            plt.text(0, 0.713, 'b', color='r', fontsize=f)
            plt.text(0, -0.713, 'b', color='r', fontsize=f)
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()

    plt.show()

def main():
    cwd = os.getcwd()
    p_dir = cwd + '/pickles/'
    fin = ['bvert_11MeV.p', 'cpvert_11MeV.p', 'bvert_4MeV.p', 'cpvert_4MeV.p']
    #fin = ['bvert_4MeV.p', 'cpvert_4MeV.p']
    dets = [4, 5, 6 ,7 ,8 , 9, 10, 11, 12, 13, 14, 15]

    # check individual tilt anlges
    if check_tilt:
        for f in fin:
            if '11' in f:
                beam_11MeV = True
            else:
                beam_11MeV = False
            if 'bvert' in f:
                tilts = [0, 45, -45, 30, -30, 15, -15]
            else:
                tilts = [0, 30, -30, 15, -15]

            data = pd_load(f, p_dir)
            data = split_filenames(data)
            tilt_check(data, dets, tilts, f, cwd, p_dir, beam_11MeV, print_max_ql=False, get_a_data=True, pulse_shape=True, 
                        delayed=False, prompt=False, show_plots=True, save_plots=False, save_pickle=False)

    # cleans measured data
    if smooth_tilt:
        smoothing_tilt(dets, fin, cwd, p_dir, pulse_shape=False, delayed=False, prompt=False, show_plots=True, save_plots=False, save_pickle=False)

    # comparison of ql for recoils along the a-axis
    if compare_a_axes:
        compare_a_axis_recoils(fin, dets, cwd, p_dir, plot_by_det=True, save_plots=False)

    # plot ratios
    if ratios_plot:
        plot_ratios(fin, dets, cwd, p_dir, pulse_shape=False, plot_fit_ratio=True)

    if adc_vs_cal:
        adc_vs_cal_ratios(fin, dets, cwd, p_dir, plot_fit_ratio=True)

    if acp_curves:
        plot_acp_lo_curves(fin, dets, cwd, p_dir, pulse_shape=False, br_only=True, plot_fit_data=False)
    
    # 3d plotting
    theta_n = [70, 60, 50, 40, 30, 20, 20, 30, 40, 50, 60, 70]
    phi_n = [180, 180, 180, 180, 180, 180, 0, 0, 0, 0, 0, 0]
    bvert_tilt = [0, 45, -45, 30, -30, 15, -15]
    cpvert_tilt = [0, 30, -30, 15, -15]

    ## crystal orientations ((a_x, a_y, a_z),(b_x, b_y, b_z),(c'_x, c'_y, c'_z))
    b_up = np.asarray(((-1,0,0), (0,-1,0), (0,0,1)))
    cp_up = np.asarray(((1,0,0), (0,0,1), (0,-1,0)))

    ## use to check orientations
    if scatter_11:
        scatter_check_3d(fin[0], fin[1], dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, beam_11MeV=True, matplotlib=False)
    if scatter_4:
        scatter_check_3d(fin[2], fin[3], dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, beam_11MeV=False, matplotlib=False)

    ## heat maps with data points
    if heatmap_11:
        plot_heatmaps(fin[0], fin[1], dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, beam_11MeV=True, 
                      plot_pulse_shape=True, multiplot=False, save_multiplot=False, show_delaunay=False)
    if heatmap_4:
        plot_heatmaps(fin[2], fin[3], dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, beam_11MeV=False, 
                      plot_pulse_shape=True, multiplot=False, save_multiplot=False, show_delaunay=False)

    ## heat maps with fitted data
    sin_fits = ['bvert_11MeV_sin_params.p', 'cpvert_11MeV_sin_params.p', 'bvert_4MeV_sin_params.p', 'cpvert_4MeV_sin_params.p']
    if fitted_heatmap_11:
        plot_fitted_heatmaps(sin_fits[0], sin_fits[1], dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, beam_11MeV=True, multiplot=False, save_multiplot=False)
    if fitted_heatmap_4:
        plot_fitted_heatmaps(sin_fits[2], sin_fits[3], dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, beam_11MeV=False, multiplot=False, save_multiplot=False)

    ## heat maps with data points
    if avg_heatmap_11:
        plot_avg_heatmaps(fin[0], fin[1], dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, beam_11MeV=True, 
                      plot_pulse_shape=True, multiplot=False, save_multiplot=False, show_delaunay=False)
    if avg_heatmap_4:
        plot_avg_heatmaps(fin[2], fin[3], dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, beam_11MeV=False, 
                      plot_pulse_shape=False, multiplot=False, save_multiplot=False, show_delaunay=False)

    # plots smoothed data from smooth_tilt()
    sin_fits = ['bvert_11MeV_sin_params_smoothed.p', 'cpvert_11MeV_sin_params_smoothed.p', 'bvert_4MeV_sin_params_smoothed.p', 'cpvert_4MeV_sin_params_smoothed.p']                     
    if smoothed_fitted_heatmap_11:
        plot_smoothed_fitted_heatmaps(sin_fits[0], sin_fits[1], dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, pulse_shape=True, 
                                      beam_11MeV=True, multiplot=False, save_multiplot=False)
    if smoothed_fitted_heatmap_4:
        plot_smoothed_fitted_heatmaps(sin_fits[2], sin_fits[3], dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, pulse_shape=True, 
                                      beam_11MeV=False, multiplot=False, save_multiplot=False)

    ## polar interpolation
    if polar_plots:
        polar_plot(fin[0], fin[1], dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, beam_11MeV=True)

    ## rbf interpolation
    if rbf_interp_heatmaps:
        rbf_interp_heatmap(fin[0], fin[1], dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, beam_11MeV=True, plot_pulse_shape=True, multiplot=False, save_multiplot=False)

    ## spherical harmonics fit to lo data
    if spherical_harmonics:
        sph_harm_fit(fin[0], fin[1], dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, beam_11MeV=True, plot_pulse_shape=False, multiplot=False, save_multiplot=False)

    if lsq_sph_biv_spline:
        lsq_sph_biv_spl(fin[0], fin[1], dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, beam_11MeV=True, plot_pulse_shape=False, multiplot=False, save_multiplot=False)

    if legendre_poly:
        legendre_poly_fit(fin[0], fin[1], dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, beam_11MeV=True, plot_pulse_shape=False, multiplot=False, save_multiplot=False)

    if lambertian_proj:
        lambertian(fin[0], fin[1], dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, beam_11MeV=True, pulse_shape=False)
        lambertian(fin[2], fin[3], dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, beam_11MeV=False, pulse_shape=False)

    if lambertian_smoothed:
        lambertian_smooth(sin_fits[0], sin_fits[1], (fin[0], fin[1]), dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, beam_11MeV=True, pulse_shape=True)
        lambertian_smooth(sin_fits[2], sin_fits[3], (fin[2], fin[3]), dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, beam_11MeV=False, pulse_shape=True)

if __name__ == '__main__':
    # check 3d scatter plots for both crystals
    scatter_11 = False
    scatter_4 = False

    # check lo for a specific tilt (sinusoids)
    check_tilt = False

    # smooth measured data
    smooth_tilt = False

    # compare a_axis recoils (all tilts measure ql along a-axis)
    compare_a_axes = False

    # plots a/c' and a/b ql or pulse shape ratios from 0deg measurements
    ratios_plot = False

    # analyze relative light output ratios agains calibrated data ratios (used to identify original calibration issues)
    adc_vs_cal = False

    # plot a, cp LO curves
    acp_curves = False

    # plot heatmaps with data points
    heatmap_11 = False 
    heatmap_4 = False

    # plot heatmaps with fitted data
    fitted_heatmap_11 = False
    fitted_heatmap_4 = False

    # plot heatmaps with data points
    avg_heatmap_11 = False 
    avg_heatmap_4 = False

    # plot smoothed measured data
    smoothed_fitted_heatmap_11 = False
    smoothed_fitted_heatmap_4 =  False

    # polar plots
    polar_plots = False

    # rbf interpolated heatmaps
    rbf_interp_heatmaps = False

    # fit using spherical hamonics
    spherical_harmonics = False

    # interpolation using least-squares bivariat spline approximation
    lsq_sph_biv_spline = False

    # legendre polynomial fit
    legendre_poly = False

    # Lambertian projection
    lambertian_proj = True
    lambertian_smoothed = False

    main()