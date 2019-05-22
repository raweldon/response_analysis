import numpy as np
import os
import pandas as pd
import re
import matplotlib.pyplot as plt

def pd_load(filename, p_dir):
    # converts pickled data into pandas DataFrame
    print '\nLoading pickle data from:\n', p_dir+filename
    data = pd.read_pickle(p_dir + filename)
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

def tilt_check(det_data, dets, tilts, pickle_name, cwd, p_dir, beam_11MeV, print_max_ql, get_a_data, pulse_shape, delayed, prompt, show_plots, save_plots, save_pickle):
    ''' Use to check lo, pulse shape (pulse_shape=True), or delayed pulse (delayed=True) for a given det and tilt
        Data is fitted with sinusoids
        Sinusoid fit parameters can be saved for later use by setting save_pickle=True
        a and c' axes directions are marked with scatter points
    '''
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
    a_axis_data.append(['crystal', 'energy', 'det', 'tilt', 'ql', 'abs_uncert', 'fit_ql'])
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

            elif delayed:
                data = det_df.qs_mean
                data_uncert = det_df.qs_abs_uncert
            
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

            else:
                data = det_df.ql_mean
                data_uncert = det_df.ql_abs_uncert

            # fit
            res = fit_tilt_data(data.values, angles, print_report=False)
            pars = res.best_values
            x_vals = np.linspace(0, 200, 100)
            x_vals_rad = np.deg2rad(x_vals)
            y_vals = sin_func(x_vals_rad, pars['a'], pars['b'], pars['phi'])
            sin_params.append([tilt, det, pars['a'], pars['b'], pars['phi']])

            name = re.split('\.|_', det_df.filename.iloc[0])  
            #print name[0] + '_' + name[1] + '_' + name[2] + '_' + name[4], max(data)
            # get lo along a, b, c' axes
            if a_axis_dir[d] in angles:
                a_axis_ql = data.iloc[np.where(angles == a_axis_dir[d])].values[0]
                a_axis_data.append([name[0], name[1], name[4], tilt, a_axis_ql, data_uncert.iloc[np.where(angles == a_axis_dir[d])].values[0], y_vals.max()])
            if cp_b_axes_dir[d] in angles:
                cp_b_ql = data.iloc[np.where(angles == cp_b_axes_dir[d])].values[0]
                cp_b_axes_data.append([name[0], name[1], name[4], tilt, cp_b_ql, data_uncert.iloc[np.where(angles == cp_b_axes_dir[d])].values[0], y_vals.min()])

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
        pickle.dump( sin_params, open( p_dir + name + '_sin_params.p', "wb" ) )
        print 'pickle saved to ' + p_dir + name + '_sin_params.p'

    if get_a_data:
        headers = a_axis_data.pop(0)
        return pd.DataFrame(a_axis_data, columns=headers), pd.DataFrame(cp_b_axes_data, columns=headers)

def main(cwd, p_dir, fin, dets, pulse_shape, tilts):
    for i, f in enumerate(fin):
        label_fit = ['', '', '', 'fit ratios']
        label = ['L$_a$/L$_c\'$', 'L$_a$/L$_b$', '', '']
        label_pat = ['', '', '', 'Schuster ratios']
        if '11' in f:
            beam_11MeV = True
            angles = [70, 60, 50, 40, 30, 20, 20, 30, 40, 50, 60, 70]
            p_erg = 11.325*np.sin(np.deg2rad(angles))**2
            ps_baseline_uncert = (0.001, 0.001, 0.001, 0.002, 0.005, 0.016, 0.016, 0.005, 0.002, 0.001, 0.001, 0.001) # qs uncer
        else:
            beam_11MeV = False
            angles = [60, 40, 20, 30, 50, 70]
            p_erg = 4.825*np.sin(np.deg2rad(angles))**2
            ps_baseline_uncert = (0.001, 0.001, 0.002, 0.003, 0.008, 0.027, 0.026, 0.008, 0.003, 0.002, 0.001) # qs uncert
        if 'bvert' in f:
            tilts = [0]
            color = 'r'
        else:
            tilts = [0]
            color = 'b'

        data = pd_load(f, p_dir)
        data = split_filenames(data)  
        a_axis_df, cp_b_axes_df = tilt_check(data, dets, tilts, f, cwd, p_dir, beam_11MeV, print_max_ql=False, 
                                             get_a_data=True, pulse_shape=pulse_shape, delayed=False, prompt=False, 
                                             show_plots=False, save_plots=False, save_pickle=False)
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
                    unc = np.sqrt(rat**2 * ((a_uncert/a_ql)**2 + (cp_b_uncert/cp_b_ql)**2)) # no baseline uncert
                    rat_fit = a_fit_ql/cp_b_fit_ql
                    ratio.append(rat)
                    uncert.append(unc)
                    fit_ratio.append(rat_fit)
                    shape = '^'
                else:
                    continue
    plt.show()

if __name__ == '__main__':
    cwd = os.getcwd()
    p_dir = cwd + '/pickles/'
    fin = ['bvert_11MeV.p', 'cpvert_11MeV.p', 'bvert_4MeV.p', 'cpvert_4MeV.p']
    dets = [4, 5, 6 ,7 ,8 , 9, 10, 11, 12, 13, 14, 15]
    tilts = [0] # only want a-cp and a-b ratios

    pulse_shape = False
    main(cwd, p_dir, fin, dets, pulse_shape, tilts)