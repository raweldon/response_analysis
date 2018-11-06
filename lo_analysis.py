import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import re
import time

def pd_load(filename, p_dir):
    # converts pickled data into pandas DataFrame
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

def tilt_check(det_data, det, tilt, beam_11MeV):
    # check lo for a given det and tilt
    det_df = det_data.loc[(det_data.det_no == str(det))]
    tilt_df = det_df[(det_df.tilt == str(tilt))]

    # order by correct rot angle
    if beam_11MeV:
        if len(tilt_df.filename) == 19:
            rot_order = [15, 16, 17, 18, 6, 7, 0, 1, 2, 3, 4, 8, 9, 10, 5, 11, 12, 13, 14]
            angles = np.arange(0, 190, 10)
        else:
            rot_order = [15, 16, 17, 18, 19, 6, 7, 0, 1, 2, 3, 4, 8, 9, 10, 5, 11, 12, 13, 14]
            angles = np.arange(0, 200, 10)
    else:
        rot_order = [15, 16, 17, 18, 6, 7, 0, 1, 2, 3, 4, 8, 9, 10, 5, 11, 12, 13, 14]
        angles = np.arange(0, 190, 10) 

    tilt_df = tilt_df.assign(rot_order = rot_order)

    tilt_df = tilt_df.sort_values('rot_order')

    plt.figure()
    plt.errorbar(angles, tilt_df.ql_mean, yerr=tilt_df.ql_abs_uncert.values, ecolor='black', markerfacecolor='none', fmt='o', 
                 markeredgecolor='red', markeredgewidth=1, markersize=10, capsize=1)
    for rot, ang, t in zip(tilt_df.rotation, angles, tilt_df.ql_mean):
        plt.annotate( rot, xy=(ang, t), xytext=(-3, 10), textcoords='offset points')
    plt.xlim(-5, 185)
    plt.ylabel('light output (MeVee)')
    plt.xlabel('rotation angle (degree)')
    name = tilt_df.filename.iloc[0].split('.')[0]
    plt.title(name)
    
def main():
    p_dir = 'C:/Users/raweldon/Research/TUNL/git_programs/response_analysis/pickles/'
    fin = ['bvert_11MeV.p', 'cpvert_11MeV.p', 'bvert_4MeV.p', 'cpvert_4MeV.p']
    dets = [4, 5, 6 ,7 ,8 ,9, 10, 11, 12, 13, 14, 15]
    tilts = [0, 45, -45, 30, -30, 15, -15]

    for f in fin:
        if '11' in f:
            beam_11MeV = True
        else:
            beam_11MeV = False

        data = pd_load(f, p_dir)
        data = split_filenames(f)

        for tilt in tilts:
            for det in dets:
                print tilt, det
                tilt_check(f, det, tilt, beam_11MeV)
            plt.show()

if __name__ == '__main__':
    main()