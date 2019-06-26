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
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter
import lmfit
import pickle
import dask.array as da

def pd_load(filename, p_dir):
    # converts pickled data into pandas DataFrame
    print '\nLoading pickle data from:\n', p_dir + filename
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

def fit_tilt_data(data, angles, print_report):
        # sinusoid fit with lmfit
        angles = [np.deg2rad(x) for x in angles]
        gmodel = lmfit.Model(sin_func)
        params = gmodel.make_params(a=1, b=1, phi=0)
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
        out = open( p_dir + name + '_sin_params.p', "wb" )
        pickle.dump( sin_params, out)
        out.close()
        print 'pickle saved to ' + p_dir + name + '_sin_params.p'

    if get_a_data:
        headers = a_axis_data.pop(0)
        #pickle.dump( a_axis_data, open( p_dir + 'a_axis_data.p', 'wb'))
        return pd.DataFrame(a_axis_data, columns=headers), pd.DataFrame(cp_b_axes_data, columns=headers)

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

def compare_a_axis_recoils(fin, dets, cwd, p_dir, plot_by_det, save_plots):
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
    plt.show()

def plot_ratios(fin, dets, cwd, p_dir, pulse_shape, plot_fit_ratio):
    ''' Plots a/c' and a/b ql or pulse shape ratios
        Set pulse_shape=True for pulse shape ratios
        Uses tilt_check function to get data from dataframes
    '''
    for i, f in enumerate(fin):
        label_fit = ['', '', '', 'fit ratios']
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
        harmonics = 0
        for name, oc in zip(names, order_coeff):
            c = pars[name]
            harmonics += c*sph_harm(oc[1], oc[0], theta, phi)

        harmonics = harmonics.real
        return ql - harmonics

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
        tilts = np.concatenate([df_b_mapped.tilt.values, df_b_mapped_mirror.tilt.values, df_cp_mapped.tilt.values, df_cp_mapped_mirror.tilt.values])

        ## remove repeated points
        xyz = np.array(zip(x, y, z))
        xyz_u, indices = np.unique(xyz, axis=0, return_index=True)
        ql = ql[indices]
        tilts = tilts[indices]
        x, y, z = xyz_u.T

        ## convert back to spheical coords for sph harmonics fitting
        #theta = np.arccos(z)
        theta = np.arctan(np.sqrt(x**2 + y**2)/z)
        phi = np.arctan(y/x)
        ## check for nan in phi
        nans = np.argwhere(np.isnan(phi))
        for nan in nans:
            phi[nan] = phi[nan-1]

        # lmfit
        for i in range(4, 5):
            order = i
            names, order_coeff = get_coeff_names(order, central_idx_only=False)
            fit_params = add_params(names)
            fit_kws={'nan_policy': 'omit'}
            res = lmfit.minimize(minimize, fit_params, args=(ql, theta, phi, names, order_coeff), **fit_kws)
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
        #theta = np.arctan(np.sqrt(x**2 + y**2)/z)
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

        x = np.sin(theta)*np.cos(phi)
        y = np.sin(theta)*np.sin(phi)
        z = np.cos(theta)
        fig = mlab.figure(size=(400*2, 350*2)) 
        pts = mlab.points3d(x, y, z, ql, colormap='viridis', scale_mode='none', scale_factor=0.03)
        for x_val, y_val, z_val, ql_val, th, ph in zip(x, y, z, ql, theta, phi):
            if th==0 or ph==0:
                #mlab.text3d(x_val, y_val, z_val, str(round(ql_val,3)), scale=0.03, color=(0,0,0), figure=fig)
                mlab.text3d(x_val, y_val, z_val, str(round(th, 3)) + ', ' + str(round(ph, 3)), scale=0.03, color=(0,0,0), figure=fig)
                print th, ph, ql_val

        # interpolate 
        ## set interpolator object with mesh 
        knotst = np.linspace(0, np.pi, 31)
        knotsp = np.linspace(0, 2*np.pi, 31)
        lats, lons = np.meshgrid(knotst, knotsp)

        knotst[0] += 0.0001
        knotst[-1] -= 0.00001
        knotsp[0] += 0.00001
        knotsp[-1] -= 0.00001

        #lut = lsqsbs(theta, phi, ql, knotst, knotsp)
        #for t, p, q in sorted(zip(theta, phi, ql)):
        #    print t, p, q
        lut = ssbs(theta, phi, ql, s=1)
        ql_new = []
        for t, p in zip(theta, phi):
            ql_new.append(lut(t, p)[0][0])
        ql_new = np.array(ql_new)

        fig = mlab.figure(size=(400*2, 350*2)) 
        pts = mlab.points3d(x, y, z, ql_new, colormap='viridis', scale_mode='none', scale_factor=0.03)

        x_fine = np.sin(lats.ravel()) * np.cos(lons.ravel())
        y_fine = np.sin(lats.ravel()) * np.sin(lons.ravel())
        #for i, j, k in zip(np.sin(lats.ravel()), np.sin(lons.ravel()), np.sin(lats.ravel())*np.sin(lons.ravel())):
        #    print i, j, k
        #print '\n\n'
        z_fine = np.cos(lats.ravel())

        #for k1, k2, l in zip(lats.ravel(), lons.ravel(), lut(knotst, knotsp).ravel()):
        #    if k2 == 0 or (k2>np.pi-0.01 and k2<np.pi+0.01):
        #        print k1, k2, l

        # plot
        fig = mlab.figure(size=(400*2, 350*2)) 
        fig.scene.disable_render = True
        pts = mlab.points3d(x_fine, y_fine, z_fine, lut(knotst, knotsp).ravel(), colormap='viridis', scale_mode='none', scale_factor=0.03)

        for x_val, y_val, z_val, ql_val, th, ph in zip(x_fine, y_fine, z_fine, lut(knotst, knotsp).ravel(), lats.ravel(), lons.ravel()):
            #mlab.text3d(x_val, y_val, z_val, str(ql_val), scale=0.03, color=(0,0,0), figure=fig)
            if ph==0 or (ph>np.pi-0.01 and ph<np.pi+0.01):
                mlab.text3d(x_val, y_val, z_val, str(round(ql_val,3)), scale=0.03, color=(0,0,0), figure=fig)
                #mlab.text3d(x_val, y_val, z_val, str(round(th, 3)) + ', ' + str(round(ph, 3)), scale=0.03, color=(0,0,0), figure=fig)

        ## delaunay triagulation (mesh, interpolation)
        tri = mlab.pipeline.delaunay3d(pts)
        edges = mlab.pipeline.extract_edges(tri)

        tri_smooth = mlab.pipeline.poly_data_normals(tri) # smooths delaunay triangulation mesh
        surf = mlab.pipeline.surface(tri_smooth, colormap='viridis')
        
        mlab.axes(pts, xlabel='a', ylabel='b', zlabel='c\'')
        mlab.colorbar(surf, orientation='vertical') 
        mlab.view(azimuth=0, elevation=90, distance=7.5, figure=fig)  
        #mlab.title('Order =' + str(i))
        fig.scene.disable_render = False   

        x_fine = np.sin(lats) * np.cos(lons)
        y_fine = np.sin(lats) * np.sin(lons)
        z_fine = np.cos(lats)

        ql_new = []
        print lats.shape
        for t, p in zip(lats.ravel(), lons.ravel()):
            ql_new.append(lut(t, p)[0][0])
        ql_new = np.reshape(np.array(ql_new), (31, 31))

        fig = mlab.figure(size=(400*2, 350*2)) 
        pts = mlab.points3d(x_fine.ravel(), y_fine.ravel(), z_fine.ravel(), ql_new.ravel(), colormap='viridis', scale_mode='none', scale_factor=0.03)


        fig = mlab.figure(size=(400*2, 350*2)) 
        mlab.mesh(x_fine, y_fine, z_fine, scalars=ql_new, colormap='viridis')
        tri = mlab.pipeline.delaunay3d(pts)
        edges = mlab.pipeline.extract_edges(tri)

        tri_smooth = mlab.pipeline.poly_data_normals(tri) # smooths delaunay triangulation mesh
        surf = mlab.pipeline.surface(tri_smooth, colormap='viridis')
        
        mlab.axes(pts, xlabel='a', ylabel='b', zlabel='c\'')
        mlab.colorbar(surf, orientation='vertical') 
        mlab.show()

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
            tilt_check(data, dets, tilts, f, cwd, p_dir, beam_11MeV, print_max_ql=False, get_a_data=True, pulse_shape=False, 
                       delayed=False, prompt=False, show_plots=False, save_plots=False, save_pickle=True)

    # comparison of ql for recoils along the a-axis
    if compare_a_axes:
        compare_a_axis_recoils(fin, dets, cwd, p_dir, plot_by_det=True, save_plots=False)

    # plot ratios
    if ratios_plot:
        plot_ratios(fin, dets, cwd, p_dir, pulse_shape=True, plot_fit_ratio=False)

    if adc_vs_cal:
        adc_vs_cal_ratios(fin, dets, cwd, p_dir, plot_fit_ratio=True)
    
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
                      plot_pulse_shape=False, multiplot=False, save_multiplot=False, show_delaunay=False)
    if heatmap_4:
        plot_heatmaps(fin[2], fin[3], dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, beam_11MeV=False, 
                      plot_pulse_shape=True, multiplot=False, save_multiplot=False, show_delaunay=False)

    ## heat maps with fitted data
    sin_fits = ['bvert_11MeV_sin_params.p', 'cpvert_11MeV_sin_params.p', 'bvert_4MeV_sin_params.p', 'cpvert_4MeV_sin_params.p']
    if fitted_heatmap_11:
        plot_fitted_heatmaps(sin_fits[0], sin_fits[1], dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, beam_11MeV=True, multiplot=False, save_multiplot=False)
    if fitted_heatmap_4:
        plot_fitted_heatmaps(sin_fits[2], sin_fits[3], dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, beam_11MeV=False, multiplot=False, save_multiplot=False)

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

if __name__ == '__main__':
    # check 3d scatter plots for both crystals
    scatter_11 = False
    scatter_4 = False

    # check lo for a specific tilt (sinusoids)
    check_tilt = False

    # compare a_axis recoils (all tilts measure ql along a-axis)
    compare_a_axes = False

    # plots a/c' and a/b ql or pulse shape ratios from 0deg measurements
    ratios_plot = False

    # analyze relative light output ratios agains calibrated data ratios
    adc_vs_cal = False

    # plot heatmaps with data points
    heatmap_11 = False 
    heatmap_4 = False

    # plot heatmaps with fitted data
    fitted_heatmap_11 = False
    fitted_heatmap_4 = False

    # polar plots
    polar_plots = False

    # rbf interpolated heatmaps
    rbf_interp_heatmaps = False

    # fit using spherical hamonics
    spherical_harmonics = False

    # interpolation using least-squares bivariat spline approximation
    lsq_sph_biv_spline = True

    main()