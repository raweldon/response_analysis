# response_analysis notes

* 11/13/2018     
    * finally have a working version
    * uses mayavi.mlab.points3d with a delaunay filter for the interpolation
    * got invdisttree.py but did not end up using it (issues with idw method)

* 11/14/2018
    * combined all pickles into an excel sheet (pickels.xlsx)
    * use this sheet to check good and bad data points     

* 11/20/2018
    * added sinusoidal fit to tilt data

* 11/26/2018
    * added 3d plot of sinusoidal fit

* 11/27/18
    * used tilt_check function to locate max ql for bvert 0deg tilt
        * performed for both 11 MeV and 4 MeV
        * max is consistently near the a-axis (+-5deg)
    * when plotting data in 3d there appears to be an offset of ~45deg relative to the neutron scatter angle for the max ql from the a-axis
        * e.g. for the 70deg backing detector the max is ~25deg away from the a-axis, for the 40deg detector it is ~5deg away from the a-axis
        * bl ql maxes are all in a-c' plane
        * br ql maxes are mostly not in a-c' plane
        * FIXED:
            * changed theta_proton = np.deg2rad(90 - theta_neutron) (line 213) to theta_proton = np.deg2rad(theta_neutron)
            * 90 - theta_neutron give incorrect result -- Why??? Does it matter??

* 11/28/2018
    * haven't figured out why map_3d works with theta_proton=theta_neutron and not with theta_proton = 90-theta_neutron
        * letting this go for now - look into it later
    * added compare_a_axis_recoils function 
        * LO for all tilts along a-axis show good agreement (close to expected uncertainty on the measurement)
            * relative uncerts at 11MeV <1.3%
            * relative uncerts at 4 MeV <3.2%

* 11/29/18
    * added pulse shape plots
        * issues with 3d mapping are evident
            * plots look good at 11 MeV but are messed up at 4 MeV
    * added plot of pulse shape ratios - trend is oppposite what was reported by schuster

* 11/30/18
    * cleaned up some plots to show results
    * added prompt and delayed plots to tilt_check 
        * trend like ql
        * not very interesting and does not give any insight into qs/ql
            