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
                 redshift = 0):
        
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
        self.propfluc = _PropFluc(self, N=2**14, dt=0.1*24*3600)
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
    
    
    
    def evolve_intrinsicSED(self, reverberate=True, fpd=1, fpw=10, fph=10):
        """
        Evolves the SED by first generating a realisation of the propagating
        fluctuations, and then calculaing the spectrum
        
        Note, all variability time-scales and such must be set with
        self.propfluc.... (see documentation)

        Returns
        -------
        None.

        """
        
        #Generating mdot realisations
        self.verboseprint('Generating time-series')
        xt_dsc, xt_wrm, xt_hot, dttot = self.propfluc.gene_mdotRealisation(fpd, fpw, fph)
        fm_seed = self._gene_LseedFM(xt_dsc, xt_wrm, self.propfluc.ts)
        
        setattr(self, 'xt_hot', xt_hot)
        setattr(self, 'xt_wrm', xt_wrm)
        
        
        #Calculating spectral parameters
        self.verboseprint('Calculating spectral parameters')
        Lseed = self.sed.calc_Lseed(fm_seed)
        Ldiss = self.sed.calc_Ldiss(xt_hot)
        gamma_hs = self.sed.calc_gammah(Ldiss, Lseed)
        kths = self.sed.calc_kTe(gamma_hs)
        ktseeds = self.sed.calc_kTseed(fm_seed[:, -1])
        
        #Getting reverberation time-scales
        Lxs = Ldiss + Lseed #Total x-ray lum
        Lxs_interp = interp1d(self.propfluc.ts, Lxs, kind='linear',
                              bounds_error=False, fill_value=np.mean(Lxs))
        Lx_dscarr = self._gene_discReverbLXarr(Lxs_interp, self.logr_ad_bins,
                                               self.dlogr_ad)
        Lx_wrmarr = self._gene_discReverbLXarr(Lxs_interp, self.logr_wc_bins, 
                                               self.dlogr_wc)
        
        self.verboseprint('Calculating SEDs')
        #Now calculating variable spectral components
        if len(self.logr_ad_bins) > 1:
            self.verboseprint('Disc component...')
            Ldisc_var = self.sed.disc_spec(xt_dsc, reprocess=reverberate, Lx=Lx_dscarr)
        else:
            Ldisc_var = np.zeros((len(self.Egrid), len(self.propfluc.ts)))
        self.verboseprint('Warm Compton component')
        Lwarm_var = self.sed.warm_spec(xt_wrm, reprocess=reverberate, Lx=Lx_wrmarr)
        self.verboseprint('Hot Compton component')
        Lhot_var = self.sed.hot_spec(xt_hot, gamma=gamma_hs, kte=kths,
                                     kts=ktseeds, Lseed=Lseed)
        
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
        return self.Lintrinsic_var, dttot
    
    
    
    
    ###########################################################################
    #---- Wind/Cloudy - Methods for calculating the observerd emission
    #     Note, these work on the basis that Cloudy has already been run
    #     (see next section) and the output loaded
    ###########################################################################
    
    def _windSED_fromEmiss(self, ref_emiss):
        """
        Takes the wind emissivity calculated by cloudy, and applies wind
        geometry in order to calculate observed SED

        Parameters
        ----------
        ref_emiss : 2D array
            Wind emissivity for each calculated cloudy grid

        Returns
        -------
        Lref_tot : 1D-array
            Reflected wind SED component, as seen by an observer

        """
        
        phi_w_mids = self.phi_w_bins[:-1] + self.dphi_w/2
        cos_bet = self._windVisibility(phi_w_mids)
        
        for i, dw in enumerate(self.dw_mids):
            dA = self.dcos_thw * self.dphi_w * (dw*self.Rg)**2 #grid surface area
            
            Lr = ref_emiss[:, i] * dA * (cos_bet[:, np.newaxis]/0.5)
            Lr = np.sum(Lr, axis=0)
            
            if i == 0:
                Lref_tot = Lr
            else:
                Lref_tot = Lref_tot + Lr
        
        return Lref_tot
    
    
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
        
        Lref_tot = self._windSED_fromEmiss(self._ref_emiss_mean)
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
        pool = Pool(Ncpu)
        Lref_var = np.zeros((len(self.Egrid), len(self.propfluc.ts)))
        for lann in pool.imap_unordered(self._calc_windLum_ann, didxs):
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
        cos_bet = self._windVisibility(phi_w_mids)
        dA = self.dcos_thw * self.dphi_w * (self.dw_mids[didx]*self.Rg)**2
        
        Lr_ann = np.zeros((len(self.Egrid), len(self.propfluc.ts)))
        for j, phi in tqdm(enumerate(phi_w_mids), total=len(phi_w_mids), desc=f'Calculting for didx {didx}'):
            ref_grd = self._get_emiss_t_grd(didx, phi)
            lr = ref_grd * dA * (cos_bet[j]/0.5)
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
        
        
        #Calculating emissivity at grid at time t
        Sfac = (Lw_in - self.Lintrinsic_max[:, np.newaxis])
        Sfac /= (self.Lintrinsic_min[:, np.newaxis] - self.Lintrinsic_max[:, np.newaxis])
        Sfac = np.ma.masked_invalid(Sfac) #Masking Nan values (usually at edge where we have divide by 0!)
        
        ref_emiss = Sfac * self._ref_emiss_min[:, didx, np.newaxis]
        ref_emiss += (1-Sfac)*self._ref_emiss_max[:, didx, np.newaxis]
        
        return ref_emiss
    
    
    def _windVisibility(self, phi_w_mids):
        """
        Calculates the wind visibility for each azimuth as seen by the observer

        Parameters
        ----------
        phi_w_mids : array or float
            The azimuths

        Returns
        -------
        cos_bet : array or float
            The wind visibility

        """
        
        if self.alpha_l < 90:
            cos_bet = -np.cos(phi_w_mids)*np.tan(np.deg2rad(self.alpha_l))*np.sin(self.inc)
            cos_bet += self.cos_inc
            cos_bet /= ((np.tan(np.deg2rad(self.alpha_l)))**2 + 1)
        
        else:
            #if alpha_l=90 => cylindrical geometry!!
            cos_bet = -np.cos(phi_w_mids)*np.sin(self.inc)
        
        if isinstance(cos_bet, np.ndarray):
            cos_bet[cos_bet <= 0] = 0 #setting 0 visibility if looking through point
        else:
            if cos_bet <= 0:
                cos_bet = 0
            else:
                pass
        
        return cos_bet
    
    
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

        #tau_wnd = np.sqrt(rw**2 + (hw - self.hmax)**2)
        #tau_wnd += (self.hmax - hw)*self.cos_inc
        
        tau_wnd = np.sqrt(rw**2 + (hw)**2)
        tau_wnd += (-hw)*self.cos_inc
        tau_wnd -= rw * np.sin(self.inc)*np.cos(phi_w)
        tau_wnd *= (self.Rg/self.c)

        return tau_wnd
    
    
    ###########################################################################
    #---- Wind/Cloudy - Methods for running and handling Cloudy stuff
    ###########################################################################
    
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
    
    
    def runCLOUDYmod(self, log_hden=12, log_Nh=23, mode='mean', outdir='',
                     Ncpu='max-1'):
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
            
        
        Returns
        -------
        None.

        """
        
        self.cloudy.gene_runfile() #used to execute the cloudy run
        
        #output directory
        if outdir == '':
            outdir = f'CL_run_loghden{log_hden}_logNh{log_Nh}_alpha{self.alpha_l}'
            outdir += f'_rl{self.r_l}_fcov{self.fcov}'
        else:
            pass
        
        #Checks if wind geometry has been defined
        if hasattr(self, 'rw_mids'):
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
        
        if mode == 'mean':
            if hasattr(self, 'Lnu_intrinsic'):
                pass
            else:
                self.make_intrinsicSED()
            
            print('Running Cloudy for mean')
            self._geneCL_SEDfiles(log_hden, log_Nh, self.Lnu_intrinsic, 
                                  simname='Lmean_run', Ncpu=Ncpu, outdir=outdir)
            
            self.loadCL_run(outdir, which='mean')
        
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
            self._geneCL_SEDfiles(log_hden, log_Nh, Lnu_min, 
                                  simname='Lmin_run', Ncpu=Ncpu, outdir=outdir)
            print()
            
            print('Running Cloudy for Lnu_max')
            self._geneCL_SEDfiles(log_hden, log_Nh, Lnu_max, 
                                  simname='Lmax_run', Ncpu=Ncpu, outdir=outdir)
            print()
            
            self.loadCL_run(outdir, which='min')
            self.loadCL_run(outdir, which='max')
    
    
    
    def loadCL_run(self, outdir, which='mean'):
        """
        Loads and rebins the Cloudy simulation output
        Also subtracts out line emission, and sets the emission to 
        Luminsoity per unit surface area
            (i.e ergs/cm2/s)
        
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

        Returns
        -------
        None.

        """
        
        for ridx, dw in enumerate(self.dw_mids):
            nu_cl, Lrefi, Llinei = np.loadtxt(f'{outdir}/L{which}_run_ridx{ridx}.con',
                                            usecols=(0, 5, 7), unpack=True)
            Lrefi = Lrefi-Llinei
            
            Acl = 4*np.pi*self.fcov * (dw*self.Rg)**2 #Cloud surface area
            
            Lrefi /= Acl #Luminsoty per unit area (ergs/s/cm2)
            
            if ridx == 0:
                Lref_all = Lrefi/nu_cl
            else:
                Lref_all = np.column_stack((Lref_all, Lrefi/nu_cl))
        
        #Rebinning onto same grid as rest of code
        Lref_int = interp1d(nu_cl, Lref_all, axis=0)
        Lref_bnd = Lref_int(self.nu_grid)
        
        #Storing as attribute
        setattr(self, f'_ref_emiss_{which}', Lref_bnd)
    
    
    def _geneCL_SEDfiles(self, log_hden, log_Nh, Lnu, simname, Ncpu='max-1',
                         outdir=''):
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
        
        outdir : str
            Output directory
            If '' will simply default to 'CL_run_loghden{log_hden}_logNh{logNh}'
            The default is ''

        Returns
        -------
        None.

        """
        
        Ncpu = self._getNcpu(Ncpu)
        self.cloudy.geneCL_SEDfile(self.nu_grid, Lnu, sedname=simname)
        Ltot = np.trapz(Lnu, self.nu_grid)
        
        pool = Pool(Ncpu)
        pfunc = partial(self._exeCL_run, log_hden=log_hden, log_Nh=log_Nh,
                        Ltot=Ltot, sedname=simname, fprefix=simname)
        
        ridxs = np.arange(0, len(self.rw_mids), 1) #iterable
        for _ in tqdm(pool.imap_unordered(pfunc, ridxs), total=len(ridxs)):
            pass
        
        #if outdir does not exist need to make it!
        if outdir in os.listdir():
            pass
        else:
            os.system(f'mkdir {outdir}')
        
        os.system(f'mv {simname}* {outdir}')
    
    
    def _exeCL_run(self, r_idx, log_hden, log_Nh, Ltot, sedname,
                   fprefix):
        """
        Generates CL input file and executes (as seperate routine to make
        multiprocessing easier)

        Parameters
        ----------
        r_idx : int
            Index in rw_mids
        log_hden : float or int
            log Hydrogen number density
            Units : cm^-3
        log_Nh : float or int
            log Hydrogen column density
            Units : cm^-2
        Ltot : float
            Integrates SED luminsity
            Units : ergs/s
        sedname : str
            Name of cloudy .sed file to use
        fprefix : str
            Prefix for output file names
            i.e output files will have following format:
                {fprefix}_ridx{r_idx}.in or .con, etc

        Returns
        -------
        None.

        """
        
        dw = self.dw_mids[r_idx]
        log_Dw = np.log10(dw * self.Rg) #cm
        self.cloudy.geneCL_infile(log_hden, log_Nh, self.fcov, log_Dw, Ltot, 
                                  sedname, fname=f'{fprefix}_ridx{r_idx}')
        
        os.system(f'./run {fprefix}_ridx{r_idx}')
        
    
        
        
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
                      self.rh, self.rw, np.log10(self.rout), self.hmax, self.z)
        
        
    
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
    






    
    