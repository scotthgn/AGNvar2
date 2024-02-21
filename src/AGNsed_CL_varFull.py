#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 15:11:22 2023

@author: Scott Hagen

AGNSED model, including reflected wind emisison from Cloudy, for the full
variability

Models the underlying continuum following the prescription in AGNSED/RELAGN
(Kubota & Done 2018; Hagen & Done 2023b). The calculates the contribution
to the observed emission from the intrinsic SED reflecting/re-processing off
of a wind - calculated using Cloudy (Ferland et al. 2017)

The model can then calculate the full variability signature of the system.
It starts by generating realisations of mass-accretion rate fluctuations
propagating down through the accretion flow (Ingram & Done 2011, 2012;
Ingram & Van der Klis 2013). This provides mass-accretion rate time-series
for each annulus within the flow, which are then used to calculate the 
modulation to the emission from each annulus. This also takes into account
the change in seed photon luminosity seen by the hot flow, allowing us to
model the spectral pivoting in the hard Compton emission

Now that the code knows the intrinsic variability it will calculate the
reverberation signal from both the disc and the outflowing wind - giving
an overall picture of the variability (to our current knowledge...!)

There are then methods for extracting light-curve realisations in any band,
power-spectra, time-lags, etc.
"""

import numpy as np
import os
import glob
import astropy.units as u

from scipy.interpolate import interp1d
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

from _AGN_objectHandler import AGNobject
from _CLOUDY_objectHandler import CLOUDY_object
from _Spectral_Calculator import _SpecCalc
from _PropFluc_Calculator import _PropFluc



class AGNsed_CL_fullVar(AGNobject):
    
    dr_dex = 36 #Spacing in radial disc grid
    
    dcos_thw = 0.01 #Spacing in wind grid
    dphi_w = 0.01
    
    def __init__(self,
                 M = 1e8, 
                 dist = 100, 
                 log_mdot = -1,
                 astar = 0,
                 cos_inc = 0.9,
                 tau_t = 1,
                 kTe_warm = 0.2,
                 gamma_warm = 2.5,
                 r_hot = 10,
                 r_warm = 2000,
                 log_rout = -1,
                 hmax = 10,
                 redshift = 0,
                 N=2**14,
                 dt=0.1*24*3600):
        
        """
    
        Parameters
        ----------
        M : float
            Black hole mass - Units : Msol.
            Defualt : 1e8
        dist : float
            Co-Moving distnace - Units : Mpc.
            Default : 100
        log_mdot : float
            Log (dimensionless) mass accretion rate - Units : Mdot/Mdot_edd 
            Defualt : -1
        astar : float
            Dimensionless black hole spin.
            Default : 0
        cos_inc : float
            Cosine of inclination, measured from z-axis with disc in x-y plane.
            Defualt : 0.9
        tau_t : float
            Thomson optical dpeth for the hot Comptonisation region. Assumed to 
            remain constant throughout. Is used when calculating the hot Compton
            electron temperature. If you need inspiration for what to use, then
            X-ray observations suggest tau ~ 0.5 - 2 (e.g Zdziarski et al. 1997)
            Units : Dimensionless.
            Defualt : 1
        kTe_warm : float
            Warm Compton electron temperature. Sets high energy roll-over for
            warm Comptonisation region - Units : keV
            Default : 0.2
        gamma_warm : float
            Spectral index for warm Comptonisation region. Remains fixed throughout
            Default : 2.5
        r_hot : float
            Outer radius of hot Compton X-ray corona - Units : Rg.
            If this is negative then sets to r_isco
            If this is greater than r_warm then sets to r_warm
            Default : 10
        r_warm : float
            Outer radius of warm Compton region - Units : Rg.
            If this is negative then sets to r_isco
            If this is greater than r_out then sets to r_out
            Default : 20
        log_rout : float
            log outer disc radius - Units : Rg
            If this is negative then sets to self-gravity radius (Laor & Netzer 1989)
            Default : -1 (i.e sets to self-gravity)
        hmax : float
            Maximal scale height of X-ray corona - Rg.
            If this is greater than r_hot then sets to r_hot
            Default ; 10
        redshift : float
            Redshift to source
            Default : 0
        N : int
            Number of time-steps in series
            Default 2**14
        dt : float
            Time step in time-sereis
            Units : s
            Default : 0.1*24*3600 (i.e 0.1 days)
        
        Returns
        -------
        None.

        """
        
        super().__init__(M, dist, log_mdot, astar, redshift)
        
        #Read remaining parameters (non-switching parameters)
        self.cos_inc, self.inc = cos_inc, np.arccos(cos_inc) #inc in radians
        self.tau_t = tau_t
        self.kTw = kTe_warm
        self.gamma_w = gamma_warm
        
        #Checking switching parameters
        if log_rout < 0:
            self.rout = self.r_sg
        else:
            self.rout = 10**(log_rout)
            
        if r_hot <= self.risco:
            self.rh = self.risco
        elif r_hot >= self.rout:
            raise ValueError('r_hot >= r_out!!!')
        else:
            self.rh = r_hot
        
        if r_warm <= r_hot or r_warm <= self.risco:
            self.rw = self.rh
        elif r_warm >= self.rout:
            self.rw = self.rout
        else:
            self.rw = r_warm
        
        
        if hmax >= self.rh:
            self.hmax = self.rh
        else:
            self.hmax = hmax
            
            
        #Making radial grids
        
        self.logr_ad_bins, ddex_ad = self._make_rbins_even(np.log10(self.rw), 
                                                           np.log10(self.rout)) #outer disc
        self.logr_wc_bins, ddex_wc = self._make_rbins_even(np.log10(self.rh), 
                                                           np.log10(self.rw)) #warm Compton
        self.logr_hc_bins, ddex_hc = self._make_rbins_even(np.log10(self.risco), 
                                                           np.log10(self.rh)) #hot compton    
        

        self.dlogr_ad = 1/ddex_ad
        self.dlogr_wc = 1/ddex_wc
        self.dlogr_hc = 1/ddex_hc
        
        #Initiating SED_calculator, Propfluc calculator, and Cloudy handler
        self.sed = _SpecCalc(self)
        self.propfluc = _PropFluc(self, N=N, dt=dt)
        self.cloudy = CLOUDY_object()



    ###########################################################################
    #---- Intrinsic SED and variability
    ###########################################################################
    
    def make_intrinsicSED(self, reprocess=True):
        """
        Calculates the intrinsic SED of the system (i.e disc+warm Compton+hot Compton).
        i.e no variability, re-processing, or wind contributions

        Returns
        -------
        None.

        """
        self.Lx = self.sed.calc_Ldiss() + self.sed.calc_Lseed()
        
        if hasattr(self, 'Lnu_disc'):
            pass
        else:
            self.Lnu_disc = self.sed.disc_spec(reprocess=reprocess)[:, 0]
        
        if hasattr(self, 'Lnu_warm'):
            pass
        else:
            self.Lnu_warm = self.sed.warm_spec(reprocess=reprocess)[:, 0]
        
        if hasattr(self, 'Lnu_hot'):
            pass
        else:
            self.Lnu_hot = self.sed.hot_spec()[:, 0]
        
        self._add_SEDcomponent(self.Lnu_disc, 'disc')
        self._add_SEDcomponent(self.Lnu_warm, 'warm')
        self._add_SEDcomponent(self.Lnu_hot, 'hot')
        
        self._SED_rep = reprocess
        self.Lnu_intrinsic = self.Lnu_disc + self.Lnu_warm + self.Lnu_hot
        return self.Lnu_intrinsic
    
    
    def _gene_LseedFM(self, xt_dsc, xt_wrm, ts):
        """
        Shifts the mass-accretion rate time-series from the frame of each 
        individual annulus to that of the hot corona; assuming light 
        travel-time to the outer edge of the corona

        Parameters
        ----------
        xt_dsc : 2D-array
            Mass-accretion rate time-series for each annulus in the disc region
        xt_wrm : 2D-array
            Mass-accretion rate time-series for each annulus in the warm region
        ts : array
            Time array

        Returns
        -------
        None.

        """
        
        #Doing light-travel time from geometric midpoint
        rmd = 10**(self.logr_ad_bins[:-1] - self.dlogr_ad)
        rmw = 10**(self.logr_wc_bins[:-1] - self.dlogr_wc)
        
        tau_d = ((rmd - self.rh)*self.Rg)/self.c #time delay from each annulus in disc region
        tau_w = ((rmw - self.rh)*self.Rg)/self.c #Time delay from each annulus in warm region
        
        xtd_interp = interp1d(ts, xt_dsc, kind='linear', axis=0,
                              bounds_error=False, fill_value=1)
        xtw_interp = interp1d(ts, xt_wrm, kind='linear', axis=0,
                              bounds_error=False, fill_value=1)
        
        #first doing contribution from disc
        for i, tau in enumerate(tau_d):
            xti = xtd_interp(ts - tau)[:, i]

            if i == 0:
                fm_cor = xti
            else:
                fm_cor = np.column_stack((fm_cor, xti))
        
        #and now from the warm region
        for i, tau in enumerate(tau_w):
            xti = xtw_interp(ts -tau)[:, i]
            if len(tau_d) == 0 and i == 0:
                fm_cor = xti
            else:
                fm_cor = np.column_stack((fm_cor, xti))
    
        return fm_cor
    
    
    
    def _gene_discReverbLXarr(self, Lx_intrp, logr_bins, dlogr):
        """
        Takes an X-ray light-curve - LX - and applies light-travel time
        delay to each radial annulus along the disc

        Parameters
        ----------
        Lx : scipy interpolation object
            Interpolated light-curve of X-ray corona

        Returns
        -------
        Lx_arr : 2D-array
            X-ray light-curve as seen by each radial annulus

        """
        
        rmd = 10**(logr_bins[:-1] - dlogr/2)
        Lx_arr = np.ndarray((len(self.propfluc.ts), len(rmd)))
        for i, r in enumerate(rmd):
            tau_r = self._tau_cor2dsc(r)
            
            Lxr = Lx_intrp(self.propfluc.ts[:, np.newaxis] - tau_r)
            Lxr = np.mean(Lxr, axis=-1)
            
            Lx_arr[:, i] = Lxr
            
        return Lx_arr
    
    def _tau_cor2dsc(self, r):
        """
        Does the time-delay for an annulus r from the corona to the disc
        / warm corona

        Parameters
        ----------
        r : float or int
            Disc radius - Units : Rg

        Returns
        -------
        tau_cd : array
            Time-delay for range of azimuths within annulus

        """
        
        phis = np.arange(0, 2*np.pi, 0.01)
        tau_cd = np.sqrt(r**2 + self.hmax**2) + self.hmax*self.cos_inc
        tau_cd -= r*np.cos(phis)*np.sin(self.inc)
        tau_cd *= (self.Rg/self.c)
        
        return tau_cd
    
    
    
    def evolve_intrinsicSED(self, reverberate=True, reverb_only=False,
                            reprocess=True, fpd=None, fpw=None, fph=None):
        """
        Evolves the SED by first generating a realisation of the propagating
        fluctuations, and then calculaing the spectrum
        
        Note, all variability time-scales and such must be set with
        self.propfluc.... (see documentation)
        
        Parameters
        ----------
        reverberate : bool
            Switches disc reverberation on/off (true/false)
            The default is True (on)
        reverb_only : bool
            Switch to only include disc reverberation in output variability
            i.e if True, then propagating fluctuations will only be used to 
            generetate mdot time series in disc and corona - however in the
            output SED the disc mdot(t) will be set to 1 (so intrinsically
            static)
            To be used if you want a pure reverberation model
            The defualt is False (off)
        reprocess : bool
            Turns on/off (true/false) disc re-processing. Only matters when
            reverberate is set to false, as this allows for a non-time varying
            reprocessed component in the SED (for intrinsic only runs)
            If reverberate = True, then reprocess will always be True!!
        
        
        Returns
        -------
        None.

        """
        
        #Generating mdot realisations
        xt_dsc, xt_wrm, xt_hot, dttot = self.propfluc.gene_mdotRealisation(fpd, fpw, fph)
        fm_seed = self._gene_LseedFM(xt_dsc, xt_wrm, self.propfluc.ts)
        
        setattr(self, 'xt_hot', xt_hot)
        setattr(self, 'xt_wrm', xt_wrm)
        
        
        #Calculating spectral parameters
        Lseed = self.sed.calc_Lseed(fm_seed)
        Ldiss = self.sed.calc_Ldiss(xt_hot)
        gamma_hs = self.sed.calc_gammah(Ldiss, Lseed)
        kths = self.sed.calc_kTe(gamma_hs)
        ktseeds = self.sed.calc_kTseed(fm_seed[:, -1])
        
        #Getting reverberation time-scales
        Lxs = Ldiss + Lseed #Total x-ray lum
        if reverberate:
            Lxs_interp = interp1d(self.propfluc.ts, Lxs, kind='linear',
                                  bounds_error=False, fill_value=np.mean(Lxs))
            Lx_dscarr = self._gene_discReverbLXarr(Lxs_interp, self.logr_ad_bins,
                                                   self.dlogr_ad)
            Lx_wrmarr = self._gene_discReverbLXarr(Lxs_interp, self.logr_wc_bins, 
                                                   self.dlogr_wc)
            reprocess = True #reprocess always on if reverberating!
        else:
            Lx_dscarr = None #setting None allows for non varying re-processed component in SEDs
            Lx_wrmarr = None
            if hasattr(self, 'Lx'): #needs to exist for non-varying component
                pass
            else:
                self.Lx = self.sed.calc_Lseed() + self.sed.calc_Ldiss()
        
        #if reverb only turning off disc variability
        if reverb_only:
            xt_dsc = np.ones(np.shape(xt_dsc))
            xt_wrm = np.ones(np.shape(xt_wrm))
        else:
            pass
        
        
        #Now calculating variable spectral components
        #disc
        if len(self.logr_ad_bins) > 1:
            Ldisc_var = self.sed.disc_spec(xt_dsc, reprocess=reprocess, Lx=Lx_dscarr)
        else:
            Ldisc_var = np.zeros((len(self.Egrid), len(self.propfluc.ts)))
        
        #warm compton
        if len(self.logr_wc_bins) > 1:
            Lwarm_var = self.sed.warm_spec(xt_wrm, reprocess=reprocess, Lx=Lx_wrmarr)
        else:
            Lwarm_var = np.zeros((len(self.Egrid), len(self.propfluc.ts)))
        
        #hot compton
        if len(self.logr_hc_bins) > 1:
            Lhot_var = self.sed.hot_spec(xt_hot, gamma=gamma_hs, kte=kths,
                                         kts=ktseeds, Lseed=Lseed)
        else:
            Lhot_var = np.zeros((len(self.Egrid), len(self.propfluc.ts)))
        
        self.Lintrinsic_var = Ldisc_var + Lwarm_var + Lhot_var
        
        
        #For Cloudy purposes later!!
        self.Lintrinsic_min = np.amin(self.Lintrinsic_var, axis=-1)
        self.Lintrinsic_max = np.amax(self.Lintrinsic_var, axis=-1)
        
        #Adding to VARdict
        self._add_varComponent(Ldisc_var, 'disc')
        self._add_varComponent(Lwarm_var, 'warm')
        self._add_varComponent(Lhot_var, 'hot')
        
        #Adding additional time-series if necessary
        self.parvar_dict = {'gamma_hot':gamma_hs,
                            'kTe_hot':kths,
                            'kTseed_hot':ktseeds,
                            'Lseed_hot':Lseed,
                            'Ldiss_hot':Ldiss}
        
        self._reverb = reverberate #flag for use when extracting LCs later!!
        
        self.Ltot_var = self.Lintrinsic_var
        self.dttot = dttot
        return self.Lintrinsic_var, dttot
    
    
    
    
    ###########################################################################
    #---- Wind - Methods for calculating the observerd free-bound emission
    #     Note, these work on the basis that Cloudy has already been run
    #     (see next section) and the output loaded
    ###########################################################################
    
    
    def make_windSED(self):
        """
        Takes the Cloudy output and creates the reflected wind component of
        the SED

        Returns
        -------
        None.

        """
        
        if hasattr(self, '_ref_emiss_mean'):
            pass
        else:
            print('Error! No mean Cloudy SEDs loaded!!'
                  ' Either load Cloudy output using .loadCL_run (make sure'
                  ' you do this for the output corresponding to the currently'
                  ' set wind parameters) or generate new Cloudy output using'
                  ' .runCLOUDYmod')
            
            exit()
        
        #multiply by half total solid angle, as only see wind launched from side of disc facing obs
        Lref_tot = self._ref_emiss_mean * 2*np.pi*self.fcov
        self._add_SEDcomponent(Lref_tot, 'wind_ref')
        return Lref_tot
    
    
    
    def evolve_windSED(self, Ncpu='max-1'):
        """
        Calculates the time-dependent wind SED emission. Note - the Cloudy
        simulatin must have been RUN and LOADED for this method to work!!!
        
        Note - This assumes that the time-delay from the warm Corona/disc region
        is more or less the same as from the hot corona
        (yes, technically not really the case - however most of the variability
         should arise from this region, so it works as an approximation)
        
        Parameters
        ----------
        Ncpu : str or int
            Number of cpu cores to use
            If int : treated as explicit number of CPU cores
                    Note, if more than max available cores, then will reduce
                    to maximum available on system
            
            If 'max' : Will use system max 
            
            If 'max-<int>' : Will use system max - int (e.g 1 or 2 etc)
                    Note, if int >= max, will default to 1 core as cannot use
                    negative number of cores...
            
            The default is 'max-1'

        Returns
        -------
        None.

        """
        
        if hasattr(self, '_ref_emiss_min'):
            pass
        else:
            print('Error! No variable Cloudy SEDs loaded!!'
                  ' Either load Cloudy output using .loadCL_run (make sure'
                  ' you do this for the output corresponding to the currently'
                  ' set wind parameters) or generate new Cloudy output using'
                  ' .runCLOUDYmod')
            
            exit()
            
        #If old save file this wont have been stored...
        #This only exists so I don't need to re-run the earlier models..!
        if hasattr(self, 'Lintrinsic_min'):
            pass
        else:
            self.Lintrinsic_min = np.amin(self.Lintrinsic_var, axis=-1)
            self.Lintrinsic_max = np.amax(self.Lintrinsic_var, axis=-1)
        
        #Also need to check if Lnu_intrinsic exists - for when time delay greater than t!!
        if hasattr(self, 'Lnu_intrinsic'):
            pass
        else:
            self.make_intrinsicSED(reprocess=self._reverb)
        

        didxs = np.arange(0, len(self.dw_mids), 1)
 
        print('Evolving wind...')
        #Setting up pool object
        Ncpu = self._getNcpu(Ncpu)
        pool = Pool(Ncpu, maxtasksperchild=1)
        Lref_var = np.zeros((len(self.Egrid), len(self.propfluc.ts)))
        for lann in tqdm(pool.imap_unordered(self._calc_windLum_ann, didxs),
                         total=len(didxs)):
            Lref_var += lann
        
        
        self.Lref_var = Lref_var
        self._add_varComponent(Lref_var, 'wind_ref')
        
        return self.Lref_var

    
    def _calc_windLum_ann(self, didx):
        """
        Calculates the wind luminosity from one annulus

        Parameters
        ----------
        didx : int
            Current radial index for wind

        Returns
        -------
        None.

        """
        
        phi_w_mids = self.phi_w_bins[:-1] + self.dphi_w/2
        dOmega = self.dcos_thw * self.dphi_w
        
        Lr_ann = np.zeros((len(self.Egrid), len(self.propfluc.ts)))
        for j, phi in enumerate(phi_w_mids):
            ref_grd = self._get_emiss_t_grd(didx, phi)
            lr = ref_grd * dOmega
            Lr_ann += lr
        
        return Lr_ann
        
        
    def _get_emiss_t_grd(self, didx, phi):
        """
        Calculates the time-dependent emissivity from a Cloudy grid point
        in the wind

        Parameters
        ----------
        didx : int
            The current radial index for the wind grid
        phi : float
            Azimuthal midtpoint in grid

        Returns
        -------
        ref_emiss : 2D-array
            Time-dependent reflected emissivity for grid point
            Has shape (len(Egrid), len(ts))

        """
        
        #Caclulating SED seen by wind at time t
        tauw = self._tau_wnd(didx, phi)
        tw = self.propfluc.ts - tauw
        tmax = tw[-1]
        
        idx_close = np.argmin(abs(self.propfluc.ts - tauw))
        if tauw >= self.propfluc.ts[idx_close]:
            idx1 = idx_close
            idx2 = idx_close + 1
            
        else:
            idx1 = idx_close-1
            idx2 = idx_close
            
        intrp_fac = (tauw-self.propfluc.ts[idx2])/(self.propfluc.ts[idx1] - self.propfluc.ts[idx2])
        
        idx_i = np.argwhere(self.propfluc.ts <= tmax)

        Lw_in = self.Lintrinsic_var[:, idx_i] * intrp_fac
        Lw_in += (1-intrp_fac) * self.Lintrinsic_var[:, idx_i+1]
        Lw_in = Lw_in[:, :, 0] #reducing dimensions (last one is introduced by argwhere and is useless!)
        
        Lw_in_m = np.ndarray((len(self.Egrid), len(tw[tw<=0])))
        Lw_in_m[:, :] = self.Lnu_intrinsic[:, np.newaxis]
        
        Lw_in = np.hstack((Lw_in_m, Lw_in))
        
        
        #Calculating emissivity at grid at time t using log-linear interpolation
        Sfac = (Lw_in - self.Lintrinsic_max[:, np.newaxis])
        Sfac /= (self.Lintrinsic_min[:, np.newaxis] - self.Lintrinsic_max[:, np.newaxis])
        Sfac = np.ma.masked_invalid(Sfac) #Masking Nan values (usually at edge where we have divide by 0!)
        
        ref_emiss = Sfac * self._ref_emiss_min[:, np.newaxis]
        ref_emiss += (1-Sfac)*self._ref_emiss_max[:, np.newaxis]
        
        return ref_emiss
    
    

    def _tau_wnd(self, didx, phi_w):
        """
        Calculates the time-delay to a wind grid point as seen by the observer

        Parameters
        ----------
        didx : int
            Current idx in dw_mids (distance to wind)
        phi_w : float or array
            Azimuthal coordinate - Units : rad

        Returns
        -------
        tau_wnd : array
            Time delay to wind grid point

        """
        
        rw = self.rw_mids[didx]
        hw = self.hw_mids[didx]

        
        tau_wnd = np.sqrt(rw**2 + (hw)**2)
        tau_wnd += (-hw)*self.cos_inc
        tau_wnd -= rw * np.sin(self.inc)*np.cos(phi_w)
        tau_wnd *= (self.Rg/self.c)

        return tau_wnd
    
    
    def defineWindGeom(self, r_l=300, alpha_l=70, fcov=0.8):
        """
        Defines the wind geometry used when calculating the Cloudy SEDs and
        time-delays
        
        Will check that the input covering fraction can be reached for given 
        radius and launch angle, then generates a grid in theta and phi defining
        the bi-conical wind.
        
        If the deisred fcov cannot be reached for input r_l and alpha_l, then
        the code will increase alpha_l until fcov is reached. A warning will
        also be printed!

        Parameters
        ----------
        r_l : float or int
            Launch radius - Units : Rg
            The default is 300
        alpha_l : float or int
            Launch angle - Units : Deg
            (See Fig. A1 and A2 in Hagen & Done 2023a)
            The default is 80
        fcov : float
            Total wind covering fraction, i.e taking into account the wind
            launched from BOTH sides of the disc. The effective covering fraction,
            that an observer sees is simply half of this
            Units : Omega/4pi (i.e 0<fcov<1)
            The default is 0.8

        Returns
        -------
        None.

        """
        
        alpha_l = self._checkAndAdjust_windAngle(alpha_l, fcov)
        self.fcov = fcov
        self.alpha_l = alpha_l
        self.r_l = r_l

        #generating wind grids
        #remember fcov = cos(theta_m)
        self.cos_th_bins = np.arange(0, fcov+self.dcos_thw, self.dcos_thw) #bin edges
        self.phi_w_bins = np.arange(0, 2*np.pi+self.dphi_w, self.dphi_w) #bin edges
        
        #Getting midpoint radii, height and distance for each bin
        th_mid = np.arccos(self.cos_th_bins[:-1] + self.dcos_thw/2)
        
        self.rw_mids = self.r_l * np.tan(np.deg2rad(self.alpha_l))
        self.rw_mids /= (np.tan(np.deg2rad(self.alpha_l)) - np.tan((np.pi/2) - th_mid))
        
        self.hw_mids = self.rw_mids * np.tan((np.pi/2) - th_mid)
        self.dw_mids = np.sqrt(self.rw_mids**2 + self.hw_mids**2)
        
        #Getting the disatnce used in Cloudy calculation
        #For now setting as middle distance
        self.DW_calc = min(self.dw_mids) + 0.5*(max(self.dw_mids)-min(self.dw_mids))
        self.DW_calc *= self.Rg
    
    ###########################################################################
    #---- BLR - Methods for calculating line emission from a BLR
    ###########################################################################
    
    def make_lineSED(self):
        """
        Takes the Cloudy line output (if run) and calculates the line 
        contribution to the SED

        Returns
        -------
        None.

        """
    
        if hasattr(self, '_line_emiss_mean'):
            pass
        else:
            print('Error! No mean Cloudy line emission loaded!'
                  ' Either load Cloudy output using .loadCL_run (make sure'
                  ' you do this for the output corresponding to the currently'
                  ' set wind parameters) or generate new Cloudy output using'
                  ' .runCLOUDYmod')
            exit()
            
        #multiply by half solid andle
        Lline_tot = self._line_emiss_mean * 2*np.pi*self.fcov_blr
        self._add_SEDcomponent(Lline_tot, 'wind_line')
        return Lline_tot
    
    
    def defineBLRGeom(self, r_l=5000, alpha_l=70, fcov=0.3):
        """
        Defines the wind geometry used when calculating the Cloudy SEDs and
        time-delays
        
        Will check that the input covering fraction can be reached for given 
        radius and launch angle, then generates a grid in theta and phi defining
        the bi-conical wind.
        
        If the deisred fcov cannot be reached for input r_l and alpha_l, then
        the code will increase alpha_l until fcov is reached. A warning will
        also be printed!

        Parameters
        ----------
        r_l : float or int
            Launch radius - Units : Rg
            The default is 300
        alpha_l : float or int
            Launch angle - Units : Deg
            (See Fig. A1 and A2 in Hagen & Done 2023a)
            The default is 80
        fcov : float
            Total wind covering fraction, i.e taking into account the wind
            launched from BOTH sides of the disc. The effective covering fraction,
            that an observer sees is simply half of this
            Units : Omega/4pi (i.e 0<fcov<1)
            The default is 0.8

        Returns
        -------
        None.

        """
        
        alpha_l = self._checkAndAdjust_windAngle(alpha_l, fcov)
        self.fcov_blr = fcov
        self.alpha_l_blr = alpha_l
        self.r_l_blr = r_l

        #generating wind grids
        #remember fcov = cos(theta_m)
        self.cos_th_bins_blr = np.arange(0, fcov+self.dcos_thw, self.dcos_thw) #bin edges
        self.phi_w_bins_blr = np.arange(0, 2*np.pi+self.dphi_w, self.dphi_w) #bin edges
        
        #Getting midpoint radii, height and distance for each bin
        th_mid = np.arccos(self.cos_th_bins_blr[:-1] + self.dcos_thw/2)
        
        self.rw_mids_blr = self.r_l_blr * np.tan(np.deg2rad(self.alpha_l_blr))
        self.rw_mids_blr /= (np.tan(np.deg2rad(self.alpha_l_blr)) - np.tan((np.pi/2) - th_mid))
        
        self.hw_mids_blr = self.rw_mids_blr * np.tan((np.pi/2) - th_mid)
        self.dw_mids_blr = np.sqrt(self.rw_mids_blr**2 + self.hw_mids_blr**2)
        
        #Getting the disatnce used in Cloudy calculation
        #For now setting as middle distance
        self.DW_calc_blr = min(self.dw_mids_blr) + 0.5*(max(self.dw_mids_blr)-min(self.dw_mids_blr))
        self.DW_calc_blr *= self.Rg
    
    
    ###########################################################################
    #---- Wind/Cloudy - Methods for running and handling Cloudy stuff
    ###########################################################################
    
   
    
    
    def runCLOUDYmod(self, log_hden=12, log_Nh=23, mode='mean', component='wind',
                     outdir='', flabel='', iterate=False):
        """
        Sets up and runs Cloudy
        
        If running in var mode, then evolve_intrinsic_spec needs to be run FIRST!

        Parameters
        ----------
        log_hden : TYPE, optional
            DESCRIPTION. The default is 12.
        log_Nh : TYPE, optional
            DESCRIPTION. The default is 23.
        mode : {'mean', 'var'}
            mean => single CL run on mean intrinsic SED
            var => Runs CL for time-dependent intrinsic SED. Same number of
                outputs as points in time-series used to generate initial SEDs
        component : {'wind', 'blr'}
            Which component to run for (i.e inner wind or outer BLR)
            Note - If BLR iterate should be True, as needed for line emission
        outdir : str
            Output directory
        Ncpu : str or int
            Number of cpu cores to use
            If int : treated as explicit number of CPU cores
                    Note, if more than max available cores, then will reduce
                    to maximum available on system
            
            If 'max' : Will use system max 
            
            If 'max-<int>' : Will use system max - int (e.g 1 or 2 etc)
                    Note, if int >= max, will default to 1 core as cannot use
                    negative number of cores...
            
            The default is 'max-1'
        flabel : str
            Additional label string to add as prefix onto output files
            This exists for the purpose of differntiating files when running
            batch jobs on, e.g, cosma
        iterate : int or bool
            Determines whether to iterate the cloudy run - note this can 
            significantly increase run time, and is only really necessary when
            including line emission
            Options:
                True : iterates to convergence
                False : does not iterate
                int : iterates <n> times where <n> is in
            The defauls is False
            
            
        Returns
        -------
        None.

        """
    
        if 'run' in os.listdir():
            pass #to avoid file clashes when running multiple models simoultaneously
        else:
            self.cloudy.gene_runfile() #used to execute the cloudy run
        
        #output directory
        if outdir == '':
            outdir = f'CL_run_loghden{log_hden}_logNh{log_Nh}_alpha{self.alpha_l}'
            outdir += f'_rl{self.r_l}_fcov{self.fcov}'
        else:
            pass
        
        if flabel == '':
            pass
        else:
            flabel = flabel+'_'
        
        #Checks if wind geometry has been defined
        if hasattr(self, 'rw_mids') and component.lower()=='wind':
            pass
        elif hasattr(self, 'rw_mids_blr') and component.lower()=='blr':
            pass
        else:
            print('Warning!!')
            print('-----------------------------------------------------------')
            print('Wind geometry not defined')
            print('Will use default values of: ')
            print('    r_l = 300')
            print('    alpha_l = 70')
            print('    fcov = 0.8')
            print()
            print('You can set these yourself using the .defineWindGeom method')
            print('-----------------------------------------------------------')
            print()
            
            self.defineWindGeom()
            self.defineBLRGeom()
        
        
        if mode == 'mean':
            if hasattr(self, 'Lnu_intrinsic'):
                pass
            else:
                self.make_intrinsicSED()
            
            print('Running Cloudy for mean')
            self._geneCL_SEDfiles(log_hden, log_Nh, self.Lnu_intrinsic, 
                                  simname=f'{flabel}Lmean_{component}_run', outdir=outdir,
                                  iterate=iterate)
            
            self.loadCL_run(outdir, which='mean', component=component)
        
        elif mode == 'var':
            if hasattr(self, 'Lintrinsic_var'):
                pass
            else:
                print('No variable intrinsic spectrum!!!! \n'
                      'Run .evolve_intrinsicSED FIRST!')
                exit()
            
            
            #If old save file this wont have been stored...
            #This only exists so I don't need to re-run the earlier models..!
            if hasattr(self, 'Lintrinsic_min'):
                pass
            else:
                self.Lintrinsic_min = np.amin(self.Lintrinsic_var, axis=-1)
                self.Lintrinsic_max = np.amax(self.Lintrinsic_var, axis=-1)
            
            Lnu_min = self.Lintrinsic_min
            Lnu_max = self.Lintrinsic_max
            
            print('Running Cloudy for Lnu_min')
            self._geneCL_SEDfiles(log_hden, log_Nh, Lnu_min, simname=f'{flabel}Lmin_run', 
                                  outdir=outdir, iterate=iterate)
            print()
            
            print('Running Cloudy for Lnu_max')
            self._geneCL_SEDfiles(log_hden, log_Nh, Lnu_max, simname=f'{flabel}Lmax_run',
                                  outdir=outdir, iterate=iterate)
            print()
            
            self.loadCL_run(outdir, which='min', component=component)
            self.loadCL_run(outdir, which='max', component=component)
    
    
    
    def loadCL_run(self, outdir, which='mean', component='wind'):
        """
        Loads and rebins the Cloudy simulation output
        Also subtracts out line emission, and sets the emission to 
        Luminsoity per steradian (ie divide by solid angle)
            (i.e ergs/str/s/Hz)
        
        Stores as class attribute. No need for user to use this method, as
        called automatically upon executing runCLOUDYmod. However, in case
        you forgot to save the class instance, and don't want to re-run 
        Cloudy...

        Parameters
        ----------
        outdir : str
            Directory containing Cloudy simulation
        which : {'mean', 'min', 'max'}, optional
            Which simulation run to load.
            'mean' used the mean intrinsic SED (so not time-dependent)
            'min' used the minimum from the .evolve_intrinsic_spec output
            'max' used the max from .evolve_intrinsic_spec
            Note, 'mean' will only exist if .runCLOUDYmod has been run with
            mode='mean', while 'min' and 'max' will only exist if run with
            mode='var'
            The default is 'mean'.
        component : {'wind', 'BLR'}
            Which component to load
            'wind' : Free-bound continuum from a wind (CL files with wind suffix)
            'BLR' : line emission from the BLR (CL files with blr suffix)
            The default is 'continuum'
        

        Returns
        -------
        None.

        """
        
        
        cl = glob.glob(f'{outdir}/*L{which}_{component}_run.con')
        if component.lower() == 'wind':
            if len(cl) == 0:
                #for backwards compatibility!
                cl = glob.glob(f'{outdir}/*L{which}_run.con')
            
            Omega = 4*np.pi*self.fcov
        
        elif component.lower() == 'blr':
            Omega = 4*np.pi*self.fcov_blr
        
        else:
            raise ValueError('component must be wind or blr!!')
            
            
        #Loading data files
        nu_cl, Lref, Lline = np.loadtxt(cl[0], usecols=(0, 5, 7), unpack=True)
        Lref = Lref-Lline
        
        #Omega = 4*np.pi*self.fcov #solid angle
        Lref /= Omega #Luminsoty per unit steradian (ergs/s/str)
        Lref /= nu_cl #ergs/s/str/Hz
        
        Lline_bnd = self._rebin_line(nu_cl, Lline/nu_cl)
        Lline_bnd /= Omega #Line luminsity per unit steradian (ergs/s/Hz/st)


        #Rebinning onto same grid as rest of code
        Lref_int = interp1d(nu_cl, Lref)
        Lref_bnd = Lref_int(self.nu_grid)

        
        #Storing as attribute
        if component.lower == 'wind':
            setattr(self, f'_ref_emiss_{which}', Lref_bnd)
        else:
            pass
        
        if component.lower() == 'blr':
            setattr(self, f'_line_emiss_{which}', Lline_bnd)
        else:
            pass
    
    
    def _rebin_line(self, nu_cl, Lline):
        """
        Re-bins cloudy line emission onto same grid as rest of model
        Slightly more complicated due to the fact that lines are intrinsically
        narrow (i.e delta function).

        Parameters
        ----------
        nu_cl : array
            Cloudy frequency grid
        Lline : array
            Line luminosity o Cloudy grid
            Units : ergs/s/Hz

        Returns
        -------
        L_bnd : array
            Line luminosity binned
            Units : ergs/s/Hz

        """
        
        #first creating bin edges on model gridding
        dnus = np.diff(self.nu_grid)
        dnus = np.append(dnus, dnus[-1])
        nu_l_edg = self.nu_grid - 0.5*dnus
        nu_r_edg = self.nu_grid + 0.5*dnus
        
        #getting bin widths on cloudy grid
        dnu_cl = np.diff(nu_cl)
        dnu_cl = np.append(dnu_cl, dnu_cl[-1])
        
        #converting to photons per bin
        ph_line = Lline/(nu_cl*self.h) #photons/s/Hz
        ph_line *= dnu_cl #photons/s (per bin)
        
        #re-binning
        ph_bnd = np.zeros(len(dnus))
        for i, dnu_i in enumerate(dnus):
            idx_in = np.argwhere(np.logical_and(nu_cl >= nu_l_edg[i],
                                                nu_cl < nu_r_edg[i]))[:, 0]
            
            ph_bnd[i] = np.sum(ph_line[idx_in])/dnu_i #ph/s/Hz
            
        #converting to luminosity
        L_bnd = ph_bnd * self.nu_grid * self.h #ergs/s/Hz
        return L_bnd
    
    
    def _geneCL_SEDfiles(self, log_hden, log_Nh, Lnu, simname, outdir='',
                         iterate=False, component='wind'):
        """
        For each theta grid in the wind, runs a Cloudy simulation

        Parameters
        ----------
        log_hden : float or int
            log Hydrogen number density
            Units : cm^-3
        log_Nh : float or int
            log Hydrogen column density
            Units : cm^-2
        Lnu : array
            Input SED (this will typically be mean, min, or max SEDs)
            Units : ergs/s/Hz
        simname : str
            Name for CL simulation output files. This will give following
            format
                output files: <simname>_ridx<r_idx>.out (or .con, .ovr, etc)
                in SED : <simname>.sed
        outdir : str
            Output directory
            If '' will simply default to 'CL_run_loghden{log_hden}_logNh{logNh}'
            The default is ''
        iterate : int or bool
            Determines whether to iterate the cloudy run - note this can 
            significantly increase run time, and is only really necessary when
            including line emission
            Options:
                True : iterates to convergence
                False : does not iterate
                int : iterates <n> times where <n> is in
            The defauls is False
        component : {'wind', 'blr'}
            Whether to calculate for inner wind or blr

        Returns
        -------
        None.

        """
        
        if component.lower() == 'wind':
            fcov = self.fcov
            dw_calc = self.DW_calc
        
        elif component.lower() == 'blr':
            fcov = self.fcov_blr
            dw_calc = self.DW_calc_blr
        
        
        self.cloudy.geneCL_SEDfile(self.nu_grid, Lnu, sedname=simname)
        Ltot = np.trapz(Lnu, self.nu_grid)
        
        self.cloudy.geneCL_infile(log_hden, log_Nh, fcov, np.log10(dw_calc),
                                  Ltot, sedname=simname, fname=simname, iterate=iterate)

        os.system(f'./run {simname}')
        
    
        #if outdir does not exist need to make it!
        if outdir in os.listdir():
            pass
        else:
            os.system(f'mkdir {outdir}')
        
        os.system(f'mv {simname}* {outdir}')
    
    
    
        
        
    def _checkAndAdjust_windAngle(self, alpha_l, fcov):
        """
        Checks and adjusts wind launch angle accordingly, to get desired
        covering fraction

        Parameters
        ----------
        alpha_l : float or int
            Launch angle - Units : Deg
            (See Fig. A1 and A2 in Hagen & Done 2023a)
        fcov : float
            Total wind covering fraction, i.e taking into account the wind
            launched from BOTH sides of the disc. The effective covering fraction,
            that an observer sees is simply half of this
            Units : Omega/4pi (i.e 0<fcov<1)

        Returns
        -------
        alpha_l : float or int
            Launch angle (potentially corrected) - Units : deg

        """
        
        theta_m = np.arccos(fcov)
        alpha_rad = np.deg2rad(alpha_l)
        
        if alpha_rad <= (np.pi/2)-theta_m:
            print('Desired covering fraction cannot be reached for input launch'
                  f' angle {alpha_l} deg')
            alpha_l = np.rad2deg((np.pi/2)-theta_m) + 1
            print(f'New launch angle is {alpha_l} deg')
        else:
            pass
        
        return alpha_l



    ###########################################################################
    #---- Other routines
    ###########################################################################
    
    def set_radialResolution(self, Ndex):
        """
        Re-sets the radial binning within the code
        
        This is a de-bugging aide!!!!

        Parameters
        ----------
        Ndex : float or int
            Number of bins per radial decade

        Returns
        -------
        None.

        """
        
        self.dr_dex = Ndex
        self.__init__(self.M, self.D, np.log10(self.mdot), self.a,
                      self.cos_inc, self.tau_t, self.kTw, self.gamma_w, 
                      self.rh, self.rw, np.log10(self.rout), self.hmax, self.z,
                      self.propfluc.N, self.propfluc.dt)
        
        
    
    def _getNcpu(self, Ncpu):
        """
        Takes user Ncpu input and checks against system to see if works!

        Parameters
        ----------
        Ncpu : str or int
            Number of cpu cores to use
            If int : treated as explicit number of CPU cores
                    Note, if more than max available cores, then will reduce
                    to maximum available on system
            
            If 'max' : Will use system max 
            
            If 'max-<int>' : Will use system max - int (e.g 1 or 2 etc)
                    Note, if int >= max, will default to 1 core as cannot use
                    negative number of cores...
            
            The default is 'max-1'

        Returns
        -------
        Ncpu : int
            Number of CPU cores, corrected for current system!

        """
        
        #Getting number of CPU cores to use
        Nmax = cpu_count() #max number on system
        if isinstance(Ncpu, int):
            if Ncpu < 0:
                Ncpu = 1
            elif Ncpu > Nmax:
                Ncpu = Nmax
            else:
                pass
        
        elif isinstance(Ncpu, str):
            if Ncpu.__contains__('max'):
                if Ncpu.__contains__('-'):
                    _, sub = Ncpu.split('-')
                    Ncpu = Nmax - int(sub)
                else:
                    Ncpu = Nmax
            
            else:
                raise ValueError('Wrong str format for Ncpu!!!'
                                 'Must be max or max-<int>')
        
        else:
            raise ValueError('Ncpu MUST be int or str!!!')
        
        return Ncpu
    


if __name__ == '__main__':
    agn = AGNsed_CL_fullVar()
    
    agn.defineWindGeom()
    agn.loadCL_run('CL_tstRun', which='min')
    




    
    