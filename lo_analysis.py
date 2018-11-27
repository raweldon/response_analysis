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
from scipy.interpolate import griddata, Rbf, LinearNDInterpolator, RectSphereBivariateSpline
from scipy.spatial import Delaunay
import lmfit
import pickle

def pd_load(filename, p_dir):
    # converts pickled data into pandas DataFrame
    print '\nLoading pickle data from:\n', p_dir+filename
    data = pd.read_pickle(p_dir + filename)
    headers = data.pop(0)
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
            angles =    [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 190]
        else:
            rot_order = [12, 13, 5, 6, 0, 1, 2, 3, 7, 8, 9, 4, 10, 11]
            angles =    [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180]

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
        params = gmodel.make_params(a=1, b=0.1, phi=0)
        res = gmodel.fit(data, params=params, x=angles, nan_policy='omit')
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

def tilt_check(det_data, dets, tilts, pickle_name, p_dir, beam_11MeV, show_plots, save_pickle):
    ''' Use to check lo for a given det and tilt
        Data is fitted with sinusoids
        Sinusoid fit parameters can be saved for later use by setting save_pickle=True
    '''
    get_max_ql_per_det(det_data, dets, tilts)
 
    fig_no = [1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1]
    color = ['r', 'r', 'r', 'r', 'r', 'r', 'b', 'b', 'b', 'b', 'b', 'b']
    a_axis_dir = [20, 30, 40, 50, 60, 70, 110, 120, 130, 140, 150, 160] # angle for recoils along a-axis (relative)
    cp_b_axes_dir = [x + 90 for x in a_axis_dir[:6]] + [x - 90 for x in a_axis_dir[6:]] # angles for recoils along c' or b axes

    sin_params = []
    sin_params.append(['tilt', 'det', 'a', 'b', 'phi'])
    for tilt in tilts:
        print '\n'
        tilt_df = det_data.loc[(det_data.tilt == str(tilt))]

        for d, det in enumerate(dets):
            det_df = tilt_df[(tilt_df.det_no == str(det))]
            det_df, angles = order_by_rot(det_df, beam_11MeV)

            # fit
            res = fit_tilt_data(det_df.ql_mean.values, angles, print_report=False)
            pars = res.best_values
            x_vals = np.linspace(0, 200, 100)
            x_vals_rad = np.deg2rad(x_vals)
            y_vals = sin_func(x_vals_rad, pars['a'], pars['b'], pars['phi'])
            sin_params.append([tilt, det, pars['a'], pars['b'], pars['phi']])

            # plot same det angles together
            if show_plots:
                plt.figure(fig_no[d])
                plt.errorbar(angles, det_df.ql_mean, yerr=det_df.ql_abs_uncert.values, ecolor='black', markerfacecolor=color[d], fmt='o', 
                            markeredgecolor='k', markeredgewidth=1, markersize=10, capsize=1, label='det ' + str(det))
               
                # annotate
                for rot, ang, t in zip(det_df.rotation, angles, det_df.ql_mean):
                    plt.annotate( str(rot) + '$^{\circ}$', xy=(ang, t), xytext=(-3, 10), textcoords='offset points')
               
                # plot fit
                plt.plot(x_vals, y_vals, '--', color=color[d])
                
                # check a-axis recoils have max ql - 11 and 4 MeV look good
                name = re.split('\.|_', det_df.filename.iloc[0])  
                plt.scatter(angles[np.where(angles == a_axis_dir[d])], det_df.ql_mean.iloc[np.where(angles == a_axis_dir[d])], c='k', s=120, zorder=10, label='')
                plt.scatter(angles[np.where(angles == cp_b_axes_dir[d])], det_df.ql_mean.iloc[np.where(angles == cp_b_axes_dir[d])], c='k', s=120, zorder=10, label='')
                                   
                plt.xlim(-5, max(angles)+5)
                plt.ylabel('light output (MeVee)')
                plt.xlabel('rotation angle (degree)')
                name = name[0] + '_' + name[1] + '_' + name[2] + '_' + name[4]
                print name
                plt.title(name)
                plt.legend()
        plt.show()

    # save sinusoid fit to pickle
    if save_pickle:
        name = pickle_name.split('.')[0]
        pickle.dump( sin_params, open( p_dir + name + '_sin_params.p', "wb" ) )
        print 'pickle saved to ' + p_dir + name + '_sin_params.p'

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

def crystal_basis_vectors(theta,axis_up):
    # Rotation is counterclockwise about the a-axis (x-axis)
    theta = np.deg2rad(theta)
    rot_matix_x = np.asarray(((1,0,0),(0,np.cos(theta),np.sin(theta)),(0,-np.sin(theta),np.cos(theta))))
    rotated_axes = np.transpose(np.dot(rot_matix_x,np.transpose(axis_up)))  
    return rotated_axes

def map_3d(tilt, crystal_orientation, angles, theta_neutron, phi_neutron):
    # map points to sphere surface using rotation angle (angles), neutron scatter angle, and tilt angle

    basis_vectors = crystal_basis_vectors(tilt, crystal_orientation)
    thetap, phip = [], []
    for angle in angles:
        # code from C:\Users\raweldon\Research\TUNL\crystal_orientation\crystal_orientations_3d_plot_v8.py
        angle = np.deg2rad(angle)
        # counterclockwise rotation matrix about y
        rot_matrix_y = np.asarray(((np.cos(angle), 0, -np.sin(angle)), (0, 1, 0), (np.sin(angle), 0, np.cos(angle))))
        rot_orientation = np.transpose(np.dot(rot_matrix_y, np.transpose(basis_vectors)))
    
        # proton recoil
        theta_proton = np.deg2rad(theta_neutron) # proton recoils at 90 deg relative to theta
        phi_proton = np.deg2rad(phi_neutron + 180) # phi_proton will be opposite sign of phi_neutron
    
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
        dfs.append(update_df)
    return pd.concat(dfs)

def scatter_check_3d(fin1, fin2, dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, beam_11MeV):
    ''' Scatter plots of proton recoil trajectories
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

        fig = plt.figure(det)
        ax = fig.add_subplot(111, projection='3d')
        x1, y1, z1 = polar_to_cartesian(theta1, phi1, b_up, cp_up)
        x2, y2, z2 = polar_to_cartesian(theta2, phi2, cp_up, cp_up)

        # plot
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
            ax.set_title('11.3 MeV beam, det ' + str(det) + '  (' + str(theta_n) +'$^{\circ}$)')
        else:
            ax.set_title('4.8 MeV beam, det ' + str(det) + '  (' + str(theta_n) +'$^{\circ}$)')
        plt.tight_layout()
        plt.legend()
        plt.show()

def plot_heatmaps(fin1, fin2, dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, beam_11MeV, multiplot):
    ''' Plots the heatmap for a hemishpere of measurements
        Full sphere is made by plotting a mirror image of the hemiphere measurements
    '''
    data_bvert = pd_load(fin1, p_dir)
    data_bvert = split_filenames(data_bvert)
    data_cpvert = pd_load(fin2, p_dir)
    data_cpvert = split_filenames(data_cpvert)

    for d, det in enumerate(dets):       
        print 'det_no =', det, 'theta_n =', theta_n[d]
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

        x = np.concatenate((x_b, x_cp))
        y = np.concatenate((y_b, y_cp))
        z = np.concatenate((z_b, z_cp))
        ql = np.concatenate([df_b_mapped.ql_mean.values, df_b_mapped_mirror.ql_mean.values, df_cp_mapped.ql_mean.values, df_cp_mapped_mirror.ql_mean.values])

        # remove repeated points
        xyz = np.array(zip(x,y,z))
        xyz, indices = np.unique(xyz, axis=0, return_index=True)
        ql = ql[indices]
        x, y, z = xyz.T

        # points3d with delaunay filter - works best!!
        ## use for nice looking plots
        if multiplot:
            #            a   b  c'  nice 
            azimuth =   [0,  90, 0, 25]
            elevation = [90, 90, 0, 75]
            names = ['40deg_11MeV_a', '40deg_11MeV_b', '40deg_11MeV_c\'', '40deg_11MeV_nice']
            for i, (az, el) in enumerate(zip(azimuth, elevation)):
                fig = mlab.figure(size=(400*2, 350*2)) 
                fig.scene.disable_render = True
                pts = mlab.points3d(x, y, z, ql, color=(128./256,128./256,128./256), scale_mode='none', scale_factor=0.05)
                tri = mlab.pipeline.delaunay3d(pts)
                tri_smooth = mlab.pipeline.poly_data_normals(tri) # smooths delaunay triangulation mesh
                surf = mlab.pipeline.surface(tri_smooth, colormap='viridis')
                mlab.axes(pts, xlabel='a', ylabel='b', zlabel='c\'')
                mlab.colorbar(surf, orientation='vertical') 
                mlab.view(azimuth=az, elevation=el, distance=7.5, figure=fig)
                #if theta_n[d] == 40:
                #    print theta_n[d], names[i]
                #    mlab.savefig(cwd + '/' + names[i] + '.png')
                #    print names[i] + '.png saved'
                #mlab.clf()
                #mlab.close()

                for x_val, y_val, z_val, ql_val in zip(x, y, z, ql):
                    mlab.text3d(x_val, y_val, z_val, str(ql_val), scale=0.03, color=(0,0,0), figure=fig)
                fig.scene.disable_render = False

        ## use for analysis 
        else:
            max_idx = np.argmax(ql)
            print '\nql_max = ', ql[max_idx]

            fig = mlab.figure(size=(400*2, 350*2)) 
            fig.scene.disable_render = True
            pts = mlab.points3d(x, y, z, ql, colormap='viridis', scale_mode='none', scale_factor=0.03)

            # plot max ql point (red)
            mlab.points3d(x[max_idx], y[max_idx], z[max_idx], ql[max_idx], color=(1,0,0), scale_mode='none', scale_factor=0.03)
            
            # delaunay triagulation (mesh, interpolation)
            tri = mlab.pipeline.delaunay3d(pts)
            tri_smooth = mlab.pipeline.poly_data_normals(tri) # smooths delaunay triangulation mesh
            surf = mlab.pipeline.surface(tri_smooth, colormap='viridis')
            
            # ql vals
            for x_val, y_val, z_val, ql_val in zip(x, y, z, ql):
                mlab.text3d(x_val, y_val, z_val, str(ql_val), scale=0.03, color=(0,0,0), figure=fig)

            mlab.axes(pts, xlabel='a', ylabel='b', zlabel='c\'')
            mlab.colorbar(surf, orientation='vertical') 
            mlab.view(azimuth=0, elevation=0, distance=7.5, figure=fig)  
            fig.scene.disable_render = False            
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

def plot_fitted_heatmaps(fin1, fin2, dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, beam_11MeV):
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

        x = np.concatenate((x_b, x_cp))
        y = np.concatenate((y_b, y_cp))
        z = np.concatenate((z_b, z_cp))
        ql = np.concatenate([df_b_mapped.ql.values, df_b_mapped_mirror.ql.values, df_cp_mapped.ql.values, df_cp_mapped_mirror.ql.values])

        # remove repeated points
        xyz = np.array(zip(x,y,z))
        xyz, indices = np.unique(xyz, axis=0, return_index=True)
        ql = ql[indices]
        x, y, z = xyz.T

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

def main(check_tilt, scatter_11, scatter_4, heatmap_11, heatmap_4, fitted_heatmap_11):
    cwd = os.getcwd()
    p_dir = cwd + '/pickles/'
    fin = ['bvert_11MeV.p', 'cpvert_11MeV.p', 'bvert_4MeV.p', 'cpvert_4MeV.p']
    #fin = ['bvert_4MeV.p', 'cpvert_4MeV.p']
    dets = [4, 5, 6 ,7 ,8 ,9, 10, 11, 12, 13, 14, 15]

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
            tilt_check(data, dets, tilts, f, p_dir, beam_11MeV, show_plots=True, save_pickle=False)
    
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
        scatter_check_3d(fin[0], fin[1], dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, beam_11MeV=True)
    if scatter_4:
        scatter_check_3d(fin[2], fin[3], dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, beam_11MeV=False)

    ## heat maps with data points
    if heatmap_11:
        plot_heatmaps(fin[0], fin[1], dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, beam_11MeV=True, multiplot=False)
    if heatmap_4:
        plot_heatmaps(fin[2], fin[3], dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, beam_11MeV=False, multiplot=False)

    ## heat maps with fitted data
    sin_fits = ['bvert_11MeV_sin_params.p', 'cpvert_11MeV_sin_params.p', 'bvert_4MeV_sin_params.p', 'cpvert_4MeV_sin_params.p']
    if fitted_heatmap_11:
        plot_fitted_heatmaps(sin_fits[0], sin_fits[1], dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, beam_11MeV=True)
    if fitted_heatmap_4:
        plot_fitted_heatmaps(sin_fits[2], sin_fits[3], dets, bvert_tilt, cpvert_tilt, b_up, cp_up, theta_n, phi_n, p_dir, cwd, beam_11MeV=False)

if __name__ == '__main__':
    # check 3d scatter plots for both crystals
    scatter_11 = False
    scatter_4 = False

    # check lo for a specific tilt (sinusoids)
    check_tilt = False

    # plot heatmaps with data points
    heatmap_11 = False
    heatmap_4 = False

    # plot heatmaps with fitted data
    fitted_heatmap_11 = True
    fitted_heatmap_4 = False

    main(check_tilt, scatter_11, scatter_4, heatmap_11, heatmap_4, fitted_heatmap_11)