import numpy as np 
import matplotlib
matplotlib.use('TkAgg') # speeds up 3D rendering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
import re
import time
import os
from scipy.interpolate import griddata

def pd_load(filename, p_dir):
    # converts pickled data into pandas DataFrame
    print '\n', p_dir+filename
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

def tilt_check(det_data, dets, tilts, beam_11MeV):
    # check lo for a given det and tilt
    fig_no = [1, 2, 3,4 , 5, 6, 6, 5, 4, 3, 2, 1]
    color = ['r', 'r', 'r', 'r', 'r', 'r', 'b', 'b', 'b', 'b', 'b', 'b']
 
    for tilt in tilts:
        print '\n'
        tilt_df = det_data.loc[(det_data.tilt == str(tilt))]
        for d, det in enumerate(dets):
            #print tilt, det
            det_df = tilt_df[(tilt_df.det_no == str(det))]
            
            det_df, angles = order_by_rot(det_df, beam_11MeV)

            # plot same det angles together
            plt.figure(fig_no[d])
            plt.errorbar(angles, det_df.ql_mean, yerr=det_df.ql_abs_uncert.values, ecolor='black', markerfacecolor=color[d], fmt='o', 
                         markeredgecolor='k', markeredgewidth=1, markersize=10, capsize=1, label='det ' + str(det))
            for rot, ang, t in zip(det_df.rotation, angles, det_df.ql_mean):
                plt.annotate( str(rot) + '$^{\circ}$', xy=(ang, t), xytext=(-3, 10), textcoords='offset points')
            plt.xlim(-5, 185)
            plt.ylabel('light output (MeVee)')
            plt.xlabel('rotation angle (degree)')
            name = det_df.filename.iloc[0].split('.')[0]
            print name
            plt.title(name)
            plt.legend()
        plt.show()

def polar_to_cartesian(theta, phi):
    x = np.sin(theta)*np.cos(phi)
    y = np.sin(theta)*np.sin(phi)
    z = np.cos(theta)
    return x, y , z

def plot_3d(data1, data2, det, theta_n, beam_11MeV):
    # convert to cartesian
    theta1 = data1.theta.values
    phi1 = data1.phi.values
    theta2 = data2.theta.values
    phi2 = data2.phi.values

    fig = plt.figure(det)
    ax = fig.add_subplot(111, projection='3d')
    x1, y1, z1 = polar_to_cartesian(theta1, phi1)
    x2, y2, z2 = polar_to_cartesian(theta2, phi2)

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

def crystal_basis_vectors(theta,axis_up):
    ''' Rotation is counterclockwise about the a-axis (x-axis).
    '''    
    theta = np.deg2rad(theta)
    rot_matix_x = np.asarray(((1,0,0),(0,np.cos(theta),np.sin(theta)),(0,-np.sin(theta),np.cos(theta))))
    rotated_axes = np.transpose(np.dot(rot_matix_x,np.transpose(axis_up)))  
    return rotated_axes

def map_3d(data, det, tilts, crystal_orientation, theta_neutron, phi_neutron, beam_11MeV):

    det_df = data.loc[(data.det_no == str(det))]
    dfs=[]
    for t, tilt in enumerate(tilts):
        tilt_df = det_df.loc[(data.tilt == str(tilt))]
        tilt_df, angles = order_by_rot(tilt_df, beam_11MeV)

        basis_vectors = crystal_basis_vectors(tilt, crystal_orientation)
        thetap, phip = [], []
        for angle in angles:
            # code from C:\Users\raweldon\Research\TUNL\crystal_orientation\crystal_orientations_3d_plot_v8.py
            angle = np.deg2rad(angle)
            # counterclockwise rotation matrix about y
            rot_matrix_y = np.asarray(((np.cos(angle), 0, -np.sin(angle)), (0, 1, 0), (np.sin(angle), 0, np.cos(angle))))
            rot_orientation = np.transpose(np.dot(rot_matrix_y, np.transpose(basis_vectors)))
        
            # proton recoil
            theta_proton = np.deg2rad(90 - theta_neutron) # proton recoils at 90 deg relative to theta
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

        update_df = tilt_df.assign(phi = phip)
        update_df = update_df.assign(theta = thetap)
        dfs.append(update_df)
    return pd.concat(dfs)

def main(check_tilt, plot_11, plot_4):
    cwd = os.getcwd()
    p_dir = cwd + '/pickles/'
    fin = ['bvert_11MeV.p', 'cpvert_11MeV.p', 'bvert_4MeV.p', 'cpvert_4MeV.p']
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

            tilt_check(data, dets, tilts, beam_11MeV)
    
    # 3d plotting
    ## scatter plots
    theta_n = [70, 60, 50, 40, 30, 20, 20, 30, 40, 50, 60, 70]
    phi_n = [180, 180, 180, 180, 180, 180, 180, 0, 0, 0, 0, 0]
    bvert_tilt = [0, 45, -45, 30, -30, 15, -15]
    cpvert_tilt = [0, 30, -30, 15, -15]

    # crystal orientations ((a_x, a_y, a_z),(b_x, b_y, b_z),(c'_x, c'_y, c'_z))
    b_up = np.asarray(((-1,0,0), (0,-1,0), (0,0,1)))
    cp_up = np.asarray(((-1,0,0), (0,0,1), (0,-1,0)))

    if plot_11:
        data_bvert = pd_load(fin[0], p_dir)
        data_bvert = split_filenames(data_bvert)
        data_cpvert = pd_load(fin[1], p_dir)
        data_cpvert = split_filenames(data_cpvert)
        beam_11MeV = True
        for d, det in enumerate(dets):       
            df_b_mapped = map_3d(data_bvert, det, bvert_tilt, b_up, theta_n[d], phi_n[d], beam_11MeV) 
            df_cp_mapped = map_3d(data_cpvert, det, cpvert_tilt, cp_up, theta_n[d], phi_n[d], beam_11MeV)
            plot_3d(df_b_mapped, df_cp_mapped, det, theta_n[d], beam_11MeV)
        plt.show()

    if plot_4:
        data_bvert = pd_load(fin[2], p_dir)
        data_bvert = split_filenames(data_bvert)
        data_cpvert = pd_load(fin[3], p_dir)
        data_cpvert = split_filenames(data_cpvert)
        beam_11MeV = False
        for d, det in enumerate(dets):       
            df_b_mapped = map_3d(data_bvert, det, bvert_tilt, b_up, theta_n[d], phi_n[d], beam_11MeV=False) 
            df_cp_mapped = map_3d(data_cpvert, det, cpvert_tilt, cp_up, theta_n[d], phi_n[d], beam_11MeV=False)
            plot_3d(df_b_mapped, df_cp_mapped, det, theta_n[d], beam_11MeV)
        plt.show()

    ## heat maps
    data_bvert = pd_load(fin[0], p_dir)
    data_bvert = split_filenames(data_bvert)
    data_cpvert = pd_load(fin[1], p_dir)
    data_cpvert = split_filenames(data_cpvert)

    beam_11MeV = True
    for d, det in enumerate(dets):       
        df_b_mapped = map_3d(data_bvert, det, bvert_tilt, b_up, theta_n[d], phi_n[d], beam_11MeV) 
        df_b_mapped_mirror = map_3d(data_bvert, det, bvert_tilt, np.asarray(((1,0,0), (0,1,0), (0,0,-1))), theta_n[d], phi_n[d], beam_11MeV)
        df_cp_mapped = map_3d(data_cpvert, det, cpvert_tilt, cp_up, theta_n[d], phi_n[d], beam_11MeV)
        df_cp_mapped_mirror = map_3d(data_cpvert, det, cpvert_tilt, np.asarray(((1,0,0), (0,0,-1), (0,1,0))), theta_n[d], phi_n[d], beam_11MeV)

        ql = np.concatenate([df_b_mapped.ql_mean.values,df_cp_mapped.ql_mean.values, df_b_mapped_mirror.ql_mean.values,df_cp_mapped_mirror.ql_mean.values])
        theta = np.concatenate([df_b_mapped.theta.values,df_cp_mapped.theta.values, df_b_mapped_mirror.theta.values,df_cp_mapped_mirror.theta.values])
        phi = np.concatenate([df_b_mapped.phi.values,df_cp_mapped.phi.values, df_b_mapped_mirror.phi.values,df_cp_mapped_mirror.phi.values])
        # make grid
        step = 0.01
        u = np.linspace(theta.min(), theta.max(), len(ql)) # theta
        v = np.linspace(phi.min(), phi.max(), len(ql)) # phi
        X = np.outer(np.cos(v), np.sin(u))
        Y = np.outer(np.sin(v), np.sin(u))
        Z = np.outer(np.ones(np.size(v)), np.cos(u))        
        print len(X), len(ql)
        x, y, z = polar_to_cartesian(theta, phi)
        
        # scale ql between 0, 1
        ql = (ql - min(ql))/(max(ql) - min(ql))
        print max(ql), min(ql)

        ql_interp = griddata((x, y, z), ql, (X, Y, Z), method='nearest')
        print ql_interp
        heatmap = cm.ScalarMappable(cmap='viridis').to_rgba(ql_interp)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        #ax.scatter(x, y, z, c=ql, cmap=plt.hot(), zorder=100)
        ax.scatter(x, y, z, c='k', zorder=10)
        ax.plot_surface(X, Y, Z, alpha=1.0, zorder=0)
        ax.set_xlim([-1.1,1.1])
        ax.set_ylim([-1.1,1.1])
        ax.set_zlim([-1.1,1.1])
        ax.set_xlabel('a')
        ax.set_ylabel('b')
        ax.set_zlabel('c\'')
        ax.set_aspect('equal')
        #if beam_11MeV:
        #    ax.set_title('11.3 MeV beam, det ' + str(det) + '  (' + str(theta_n) +'$^{\circ}$)')
        #else:
        #    ax.set_title('4.8 MeV beam, det ' + str(det) + '  (' + str(theta_n) +'$^{\circ}$)')
        sm = cm.ScalarMappable(cmap=cm.viridis)
        sm.set_array(heatmap)
        plt.colorbar(sm)
        plt.tight_layout()       
        plt.show() 


if __name__ == '__main__':
    plot_11 = False
    plot_4 = False
    check_tilt = False
    
    main(check_tilt, plot_11, plot_4)