''' Plots proton recoil vectors measured over a hemisphere.

    Lab frame coordinates are:
      neutron beam travels along z axis. y rotated to -y.
      Theta and phi defined normally such that y=(theta=90,phi=90) x=(theta=90,phi=0)
                                                          
              z
              ^
              |                        ------->z
              |                      - |
              ------>y  ------>     -  |
             -                     x   y
            -
           x

    Crystal coordinates defined such that theta_p is 0 along the c' axis and 90 at the b axis.
      Phi_p is 0 at the a-axis. a,b,c' is x,y,z.

                    c'
                    ^
                    |                                                
                    |
                    -------->b
                   -
                  -
                 a
      

'''

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # can't tell if this is needed or not
from matplotlib import cm
import csv

def crystal_basis_vectors(theta,axis_up):
    ''' Calculates basis vectors for tilted crystals.
        Rotation is counterclockwise about the a-axis (x-axis).
    '''    
    theta = np.deg2rad(theta)
    rot_matix_x = np.asarray(((1,0,0),(0,np.cos(theta),np.sin(theta)),(0,-np.sin(theta),np.cos(theta))))
    rotated_axes = np.transpose(np.dot(rot_matix_x,np.transpose(axis_up)))  
    return rotated_axes
    
def polar_plot(thetap, phip, tilt, energy, plot_separate):
        r = np.sqrt(1 - np.multiply(np.cos(thetap),np.cos(thetap)))
        if plot_separate == False:
            plt.figure(100)
        else:
            plt.figure()
        ax = plt.subplot(111, projection='polar')
        ax.plot(phip, r, 'o',label=tilt)
        ax.set_rmax(1.1)
        ax.legend(loc=2)
        plt.title(str(energy[index])+' MeV proton recoil')

def plot_3d_vectors(ax3d, x, thetap_all, xdata, colors, text):
    for p, thetap in enumerate(thetap_all[x]):
        ax3d.text(xdata[p],y_data[x][p],z_data[x][p], p, size=10, zorder=100, color='k') 
        if text == True:
            # label each point (theta_p,phi_p)
            ax3d.text(xdata[p],y_data[x][p],z_data[x][p], '($%d\degree, %d\degree$)' % (np.rad2deg(thetap_all[x][p]),np.rad2deg(phip_all[x][p])), 
                      size=10, zorder=100, color='k') 
        # draw vectors
        ax3d.quiver(0,0,0,x_data[x][p],y_data[x][p],z_data[x][p],pivot='tail',arrow_length_ratio=0.05,color=colors[x],linestyle='--',alpha=0.3)
        ax3d.quiver(0,0,0,-x_data[x][p],-y_data[x][p],-z_data[x][p],pivot='tail',arrow_length_ratio=0.05,color=colors[x],linestyle='--',alpha=0.3)
       
def plot_3d(crystal_tilt_angles,x_data,y_data,z_data,crystal,vectors,text):
    ''' Plot in 3d
        set vectors == True to plot vector lines
        set text == True to write theta, phi coordinates for each vector
    '''
    matplotlib.rcParams.update({'font.size': 18})
    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection='3d')
    colors = cm.jet(np.linspace(0,1,len(crystal_tilt_angles)))
    
    for x, xdata in enumerate(x_data):  
        ax3d.scatter(x_data[x],y_data[x],z_data[x],s=40,c=colors[x],label=str(crystal_tilt_angles[x])+'$\degree$') # s is marker size
        if vectors == True:
            plot_3d_vectors(ax3d, x, thetap_all, xdata, colors, text)

    ax3d.set_xlim([-1.1,1.1])
    ax3d.set_ylim([-1.1,1.1])
    ax3d.set_zlim([-1.1,1.1])
    ax3d.set_xlabel('a')
    ax3d.set_ylabel('b')
    ax3d.set_zlabel('c\'')
    ax3d.set_aspect("equal")
    ax3d.set_title(str(theta_n)+'$\degree$ neutron scatter, '+crystal)
    plt.tight_layout()
    plt.legend()

# measurement information
#angles = np.arange(0,180,10) # 11MeV beam rot angles
angles = [0, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180] # 4 MeV beam rot angles
theta_neutron = [70, 60, 50, 40, 30, 20, 20, 30, 40, 50, 60, 70]
phi_neutron = [180, 180, 180, 180, 180, 180, 0, 0, 0, 0, 0, 0]
#theta_neutron = [70,70]#, 20, 30, 40]
#phi_neutron = [180,0]#, 0, 180, 180]
beam_energy = 11.33 # MeV
energy = [round(np.sin(np.deg2rad(x))**2 * beam_energy,2) for x in theta_neutron]
#crystal_tilt_angles = np.arange(0,180,45)

# crystal orientations ((a_x, a_y, a_z),(b_x, b_y, b_z),(c'_x, c'_y, c'_z))
a_up = np.asarray(((0,-1,0), (0,0,1), (-1,0,0)))
b_up = np.asarray(((-1,0,0), (0,-1,0), (0,0,1)))
cp_up = np.asarray(((1,0,0), (0,0,1), (0,-1,0)))

crystal_orientations = [a_up,b_up,cp_up]

theta_phi, p_data = [], []
p_data.append(['theta', 'phi', 'rot_anlge', 'tilt', 'theta_n'])
for index, theta_n in enumerate(theta_neutron): 
    for co,crystal_orientation in enumerate(crystal_orientations):
        # define crystal tilt angles for each crystal
        if np.array_equal(cp_up, crystal_orientation) == True:
            crystal_tilt_angles = np.arange(-45,45,15)
        elif np.array_equal(b_up, crystal_orientation) == True:
            crystal_tilt_angles = np.arange(-45,45,15)
        else:
            crystal_tilt_angles = [0]
            
        thetap_all=[]; phip_all=[]
        for tilt in crystal_tilt_angles:
            # calculate basis vectors
            basis_vectors = crystal_basis_vectors(tilt,crystal_orientation)
            
            # get theta_p and phi_p for crystal rotation orientations
            thetap=[]; phip=[]
            for angle in angles:
    #            print '\nangle =', angle
                angle = np.deg2rad(angle)
                # counterclockwise rotation matrix about y
                rot_matrix_y = np.asarray(((np.cos(angle), 0, -np.sin(angle)), (0, 1, 0), (np.sin(angle), 0, np.cos(angle))))
                rot_orientation = np.transpose(np.dot(rot_matrix_y, np.transpose(basis_vectors)))
            
                # proton recoil
                theta_proton = np.deg2rad(theta_n) # proton recoils at 90 deg relative to theta
                phi_proton = np.deg2rad(phi_neutron[index] + 180) # phi_proton will be opposite sign of phi_neutron
            
                # cartesian vector    
                x_p = np.sin(theta_proton)*np.cos(phi_proton)
                y_p = np.sin(theta_proton)*np.sin(phi_proton)    
                z_p = np.cos(theta_proton)
                
                p_vector = np.asarray((x_p, y_p, z_p))
                
                #get theta_p
                p_vector_dot_cp = np.dot(p_vector,rot_orientation[2]) # a.b=||a||*||b|| cos(theta)
                theta_p = np.rad2deg(np.arccos(p_vector_dot_cp))
    #            print '  theta_p =',theta_p
                
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
    
                p_data.append([np.deg2rad(theta_p), np.deg2rad(phi_p), angle, tilt, theta_n])

                thetap.append(np.deg2rad(theta_p))
                phip.append(np.deg2rad(phi_p)) 

    #            polar_plot(thetap,phip,tilt,energy,plot_separate=True)                
            
            thetap_all.append(thetap)
            phip_all.append(phip)
                
#            polar_plot(thetap,phip,tilt,energy,plot_separate=False)
    #        
        #plot 3d
        # cartesian coordinates sign dependent on the crystal orientation   
        if np.array_equal(cp_up, crystal_orientation) == True:
            x_data = np.multiply(np.sin(thetap_all),np.cos(phip_all))
            y_data = np.multiply(np.sin(thetap_all),np.sin(phip_all))
            z_data = np.cos(thetap_all)
#            crystal = 'c\'-axis up'
            crystal = 'crystal 3'
            cp_3d = [x_data,y_data,z_data,crystal]
        if np.array_equal(b_up, crystal_orientation) == True:
            x_data = -np.multiply(np.sin(thetap_all),np.cos(phip_all))
            y_data = -np.multiply(np.sin(thetap_all),np.sin(phip_all))
            z_data = np.cos(thetap_all)
#            crystal = 'b-axis up'
            crystal = 'crystal 1'
            b_3d = [x_data,y_data,z_data,crystal]
        if np.array_equal(a_up, crystal_orientation) == True:
            x_data = -np.multiply(np.sin(thetap_all),np.cos(phip_all))
            y_data = np.multiply(np.sin(thetap_all),np.sin(phip_all))
            z_data = -np.cos(thetap_all)
            crystal = 'a-axis up'
            a_3d = [x_data,y_data,z_data,crystal]
#        plot_3d(crystal_tilt_angles,x_data,y_data,z_data,crystal,vectors=True,text=False)
        
    #plot all crystals
    text = True
    fig = plt.figure()
    ax3d = fig.add_subplot(111, projection='3d')
    colors = ['r','b','g']
    f = open('meas_points.txt','wb')
    writer = csv.writer(f, delimiter=' ')
    
    count = 0
    for i,axis in enumerate([b_3d,cp_3d]):
        ax3d.scatter(axis[0],axis[1],axis[2],s=40,c=colors[i],label=axis[3],alpha=0.4) # s is marker size
        for x, xdata in enumerate(axis[0]):
            for p, thetap in enumerate(thetap_all[x]):
                writer.writerow([axis[3], crystal_tilt_angles[x], angles[p], round(np.rad2deg(thetap_all[x][p]),3), round(np.rad2deg(phip_all[x][p]),3)])
                #ax3d.text(axis[0][x][p],axis[1][x][p],axis[2][x][p], '($%d^{\circ}$, $%d^{\circ}$)' % (np.rad2deg(thetap_all[x][p]),np.rad2deg(phip_all[x][p])), 
                #          size=10, zorder=100, color='k') 
                #print '($%d^{\circ}$, $%d^{\circ}$)' % (np.rad2deg(thetap_all[x][p]),np.rad2deg(phip_all[x][p]))
                # draw vectors
                ax3d.quiver(0,0,0,axis[0][x][p],axis[1][x][p],axis[2][x][p],pivot='tail',arrow_length_ratio=0.05,color=colors[i],linestyle='--',alpha=0.3)
                ax3d.quiver(0,0,0,-axis[0][x][p],-axis[1][x][p],-axis[2][x][p],pivot='tail',arrow_length_ratio=0.05,color=colors[i],linestyle='--',alpha=0.3)
                count +=1
    
    ax3d.set_xlim([-1.1,1.1])
    ax3d.set_ylim([-1.1,1.1])
    ax3d.set_zlim([-1.1,1.1])
    ax3d.set_xlabel('\na')
    ax3d.set_ylabel('\nb')
    ax3d.set_zlabel('c\'  ')
    ax3d.set_aspect("equal")
    ax3d.set_title(theta_n)
#    ax3d.set_title(r'$\theta =$'+str(theta_n)+'$\degree, \phi =$'+str(phi_neutron[index])+'$\degree$ neutron scatter, '+crystal)
    plt.tight_layout()
    plt.legend()
    plt.show()
