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
    * issues with 3d mapping for 4 MeV recoils? 
            
* 12/3/18
    * 3d mapping issue is related to x-y-z a-b-c' relationship
        * all points are already 90 - theta_n due to the cyrstalline/lab-frame mapping 
        * e.g. the mapping expects the a-axis in-line with the first rotation measurement while for the experiment we had c' in-line with the first rotation measurement
    * added rotation angle plotting to scatter_check_3d; includes plotting with mayavi or matplotlib (no text)

* 12/4/18
    * added polar plots of data with rbf interpolation

* 12/5/18
    * implemented rbf interpolation on spherical data - looks ok

* 12/11/18
    * pulse shape is reverse of what was reported by Patricia due to difference in psd calculation
        * Pat: short gate near end of pulse
        * Us: short gate at beginning of pulse
    * May have found that we did save pulse shape data -- we did

* 1/25/2019
    * updated pulse shape analysis to be comparable to Patricia
        * qs/ql --> 1-qs/ql
        * updated pulse shape ratio plots to include basline uncert (uncomment code to use) 
            * rough estimate, representative not accurate
    * added pulse shape analysis to multiplot
    * updated plot_ratios
    * new branch for testing minimization

* 4/9/2019
    * new lo calibration data added (new hot spot data)

* 5/22/2019
    * added adc_vs_cal_ratios function to analyze issues with new calibration
        * turns out the linear calibration m*x+b changes the ratio due to the intercept term (large effect at low light output)
        * reformulated according to dietze c(E_e - E_0) where (E_e - E_0) is the *effective* electron light output and E_0 accounts for quenching 
    * uba_pmt_vs_new.py - compares june 2016 data with current data 
        * looks at effect of saturation -- large
    
* 6/22/2019
    * added spherical hamonics fit to light output data
    * not working great...
    * need to determine how many terms are needed for the data

* 6/26/2019
    * got SmoothSphereBivariateSpline working -- looks great!!
    * Note: LSQSphereBivariateSpline and SmoothSphereBivariateSpline are the same if you do not use weights
    * does not work for lo values < 1
        * 20 and 30 deg lo, psd
        * will scaling correct it?
    * fixed pickle issue with pulls from github
        * pandas.read_pickle() did not properly assess how to read the pickles
        * corrected by adding:
            'with open(path_to_file, 'r') as f:
                data = pickle.load(f)         '

* 8/6/2019
    * added plot_acp_lo_curves function
    * updated compare_a_axis_recoils function to show rel diff between bvert and cpvert a-axis measurements
        * cpvert is systematically lower -- why? 

* 8/11/2019
    * added smoothing branch
        * Issue with BR vs BL anisotropy measurements (check_tilt function)
            * The anisotropy magnitude measured with BR dets is less than BL
            * There is an angular offset between the BR and BL anistoropy measurements
                * 10 deg for 0deg tilt, small (~5 deg or less) for other tilts
                * Only thing that changed between tilts was the detector position
            * Partially due to BR detectors not being level with the target det (i.e. an offset in phi?)?
                * BR LO is systematically high for EJ-228 measurements
                * LO can be higher or lower than that measured in the ac' plane depending on the offset in phi (seems like our offset is negative)
                
* 8/12/2019
    * created new functions smoothing_tilt and plot_smoothed_fitted_heatmaps for smoothed data
    * abandoned BR data - anisotropy is systematically lower than BL data - backing dets must have been off zero degree plane          

          