import pandas as pd
import os
from lo_analysis import pd_load, split_filenames, order_by_rot

cwd = os.getcwd()
p_dir = cwd + '/pickles/'
fin = ['bvert_11MeV.p', 'cpvert_11MeV.p', 'bvert_4MeV.p', 'cpvert_4MeV.p']
dets = [4, 5, 6 ,7 ,8 ,9, 10, 11, 12, 13, 14, 15]

for f in fin:
    data = pd_load(f, p_dir)
    data = split_filenames(data)
    tmp = f.split('.')
    if '11' in f:
        beam_11MeV = True
    else:
        beam_11MeV = False
    if 'bvert' in f:
        tilts = [0, 45, -45, 30, -30, 15, -15]
    else:
        tilts = [0, 30, -30, 15, -15]
    # reorder dataframes
    dfs=[]
    for tilt in tilts:
        tilt_df = data.loc[(data.tilt == str(tilt))]
        for d, det in enumerate(dets):
            #print tilt, det
            det_df = tilt_df[(tilt_df.det_no == str(det))]
            det_df, angles = order_by_rot(det_df, beam_11MeV)
            dfs.append(det_df)
    result = pd.concat(dfs)
    csv_data = result.to_csv(tmp[0] + '.csv')