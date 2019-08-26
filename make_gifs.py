import imageio
import glob
import os
import numpy as np

cwd = os.getcwd()
files_11 = glob.glob(cwd + '/figures/lambert/*lo_11*')
files_4 = glob.glob(cwd + '/figures/lambert/*lo_4*')

files = sorted(files_4)[::-1] + sorted(files_11)[::-1]
angles = np.array([20, 30, 40, 50, 60, 70])
energies = [4.83*np.sin(np.deg2rad(angles))**2] + [11.33*np.sin(np.deg2rad(angles))**2]

files = [x for _, x in sorted(zip(energies, files))]

print files

