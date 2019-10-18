import imageio
import glob
import os
import numpy as np

cwd = os.getcwd()
pulse_shape = False

if pulse_shape:
    files_11 = glob.glob(cwd + '/figures/lambert/*shape_11*')
    files_4 = glob.glob(cwd + '/figures/lambert/*shape_4*')
else:
    files_11 = glob.glob(cwd + '/figures/lambert/*lo_11*')
    files_4 = glob.glob(cwd + '/figures/lambert/*lo_4*')


files = sorted(files_4)[::-1] + sorted(files_11)[::-1]
angles = np.array([20, 30, 40, 50, 60, 70])
energies = list(4.83*np.sin(np.deg2rad(angles))**2) + list(11.33*np.sin(np.deg2rad(angles))**2)

files = [x for _, x in sorted(zip(energies, files))]
energies = sorted(energies)

print('\nimages for gif:')
for i in files:
    print( i) 

images=[]
for f in files:
    images.append(imageio.imread(f))

if pulse_shape:
    imageio.mimsave(cwd + '/figures/lambert/pulse_shape.gif', images, fps=0.5)
    print('\nfile saved to /figures/lambert/pulse_shape.gif')   
else:
    imageio.mimsave(cwd + '/figures/lambert/light_output.gif', images, fps=0.5)
    print('\nfile saved to /figures/lambert/light_output.gif')   