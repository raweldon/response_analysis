import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
import re
import time
import os

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
        rot_order = [12, 13, 5, 6, 0, 1, 2, 3, 7, 9, 9, 4, 10, 11]
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

def plot_3d(data, theta):
    # convert to cartesian
    tilt = theta
    theta = np.deg2rad([90 - t for t in theta])
    phi = np.deg2rad(data.phi.values)

    colors = cm.jet(np.linspace(0,1,len(tilt)))
    fig = plt.figure(0)
    ax = fig.add_subplot(111, projection='3d')
    for i, t in enumerate(theta):
        x = np.sin(t)*np.cos(phi)
        y = np.sin(t)*np.sin(phi)
        z = [np.cos(t)]*len(phi)

        # plot
        ax.scatter(x, y, z, label=tilt[i], c=colors[i])
    #ax.set_aspect('equal')
    plt.tight_layout()
    plt.legend()
    plt.show()


def map_3d(data, dets, tilts, beam_11MeV):
    det_ang_rel_a = [20, 30, 40, 50, 60, 70, 100, 120, 130, 140, 150, 160]
    for d, det in enumerate(dets):
        det_df = data.loc[(data.det_no == str(det))]
        dfs=[]
        for t, tilt in enumerate(tilts):
            tilt_df = det_df.loc[(data.tilt == str(tilt))]
            tilt_df, angles = order_by_rot(tilt_df, beam_11MeV)

            start = -det_ang_rel_a[d]
            if beam_11MeV:
                if len(tilt_df.filename) == 19:
                    phi = np.arange(start, start + 190, 10)
                else:
                    phi = np.arange(start, start + 200, 10)
            else:
                phi = [start, start+20, start+30, start+40, start+50, start+60, start+70, start+80, start+90, 
                       start+100, start+120, start+140, start+160, start+180]
        
            proton_phi = [90 - p for p in phi] # theta_p + theta_n = 90deg
            update_df = tilt_df.assign(phi = proton_phi)
            dfs.append(update_df)

        full_df = pd.concat(dfs)
        plot_3d(full_df, tilts)
            
            


def main():
    cwd = os.getcwd()
    p_dir = cwd + '/pickles/'
    fin = ['bvert_11MeV.p', 'cpvert_11MeV.p', 'bvert_4MeV.p', 'cpvert_4MeV.p']
    #fin = ['cpvert_4MeV.p']
    dets = [4, 5, 6 ,7 ,8 ,9, 10, 11, 12, 13, 14, 15]

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

        #tilt_check(data, dets, tilts, beam_11MeV)
        map_3d(data, dets, tilts, beam_11MeV)

if __name__ == '__main__':
    main()