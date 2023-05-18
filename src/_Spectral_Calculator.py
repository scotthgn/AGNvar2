#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 14:02:28 2023

@author: Scott Hagen

Contains methods that will calculate and return a spectrum for a given 
AGNobject
"""

import numpy as np
import astropy.units as u

from myNTHCOMP import mdonthcomp


class _SpecCalc:
    
    """
    Containts methods for calculating spectra from an AGNobject.
    
    NOTE! AGNobj MUST be a subclass of AGNobject from _AGN_objectHandler, 
    since it must contain information about radial grids and system
    energetics!
    All AGNobj class methods used here are found in _AGN_objectHandelr.py
    
    NOTE! Assumes radial grids go from out and in (as this is easier for the
    propagation parts!)
    
    """
    
    def __init__(self, AGNobj):
        """
        

        Parameters
        ----------
        AGNobject : AGNobject class
            AGNobject from _AGN)objectHandler, containing a desired model.
            Must include basic BH paramters (i.e Mass, mdot, spin) and radial
            grids, following same naming convention as used throughout this
            code. i.e:
                standard disc region: logr_ad_bins
                warm Compton region: logr_wc_bins
                hot Compton region: logr_hc_bins

        Returns
        -------
        None.

        """
        
        #read input object
        self._agn = AGNobj
    
    
    
    
    
    ###########################################################################
    #---- Standard disc region
    ###########################################################################
    
    def bb_radiance(self, T):
        """
        Calculates the thermal emission for a given temperature T

        Parameters
        ----------
        T : float OR array
            Temperature - Units : K

        Returns
        -------
        \pi * Bnu : 2D-array, shape=(len(nus), len(Ts))
            Black-body emission spectrum - Units : erg/s/cm^2/Hz

        """
        pre_fac = (2 * self._agn.h * self._agn.nu_grid[:, np.newaxis]**3)/(self._agn.c**2)
        exp_fac = np.exp((self._agn.h * self._agn.nu_grid[:, np.newaxis])/(self._agn.k_B * T)) - 1
        Bnu = pre_fac / exp_fac
        
        return np.pi * Bnu
    
    
    
    def _disc_annuli(self, r, dr, fmd=1, reprocess=False, Lx=None):
        """
        Calculates the emission from an annulus within the standard disc region

        Parameters
        ----------
        r : float
            Geometric midpoint in radial bin - Units : Rg.
        dr : float
            Bin width (linear) - Units : Rg.
        fmd : array, optional
            Mass-accretion rate time-series at r, Normalised such that the mean
            is 1.
            If float or int, then assumed constant
            The defialt is float(1)
        reprocess : Bool, optional
            Switch for whether or not to include re-proccesing
        Lx : array, optional
            ONLY usefull if reprocess=True. Sets the (time-depnt) luminosity
            of the X-ray corona.
            If None, then X-ray corona assumed to be constant at Ldiss
            Units : ergs/s

        Returns
        -------
        Lnu_ann : 2D-array, shape=(len(Egrid), len(fmd))
            
        
        """
                
        T4_ann = self._agn.calc_Tnt(r, fmd)
        if reprocess == True:
            if Lx is None:
                Lx = self._agn.Lx
                
            T4_ann = T4_ann + self._agn.calc_Trep(r, Lx)
        
        bb_ann = self.bb_radiance(T4_ann**(1/4))
        Lnu_ann = 4*np.pi*r*dr * bb_ann * self._agn.Rg**2 #erg/s/Hz

        return Lnu_ann
    
    
    def disc_spec(self, fmd=1, reprocess=False, Lx=None):
        """
        Calculates the total spectrum from the standard disc region

        Parameters
        ----------
        fmd : array, optional
            Mass-accretion rate time-series at r, Normalised such that the mean
            is 1.
            If 2D-array - then assumed to correspond to the radially dependent
                mass-accretion rate time-series. In this case the shape must be:
                shape(fmd) = (len(ts), len(rbin_edges)-1)
                i.e each column contains a time-series, and there are the same
                number of columns as radial bins
            If 1D-array it is assumed that the mass-accretion rate across the 
                entire disc region varies according to fmd
            If float or int, assumed constant
            The default is float(1)
        reprocess : Bool, optional
            Switch for whether or not to include re-proccesing
        Lx : array, optional
            ONLY usefull if reprocess=True. Sets the (time-depnt) luminosity
            of the X-ray corona.
            If None, then X-ray corona assumed to be constant at Ldiss
            Units : ergs/s

        Returns
        -------
        Lnu_tot : array
            Output spectrum for standard disc region - Units : erg/s/Hz.
            Has shape (len(Egrid), len(fmd)) (if fmd 2D, then len(fmd[:, 0]))

        """
        
        #Checking input time-series to ensure same lengths!!!
        self._check_timeseries(fmd, Lx)
                
        #Calculating disc emission if this region exists!
        if len(self._agn.logr_ad_bins) > 1:
            for i in range(len(self._agn.logr_ad_bins) - 1):
                dr_bin = 10**self._agn.logr_ad_bins[i] - 10**self._agn.logr_ad_bins[i+1]
                rmid = 10**(self._agn.logr_ad_bins[i] - self._agn.dlog_r/2)
 
                if isinstance(fmd, (float, int)) or fmd.ndim==1:
                    fmdi = fmd
                else:
                    fmdi = fmd[:, i]
                
                if isinstance(Lx, (float, int)) or Lx is None or Lx.ndim==1:
                    Lxi = Lx
                else:
                    Lxi = Lx[:, i]
               
                Lnu_r = self._disc_annuli(rmid, dr_bin, fmdi, reprocess, Lxi)
                if i == 0:
                    Lnu_tot = Lnu_r
                else:
                    Lnu_tot += Lnu_r
        
        else:
            Lnu_tot = np.zeros((len(self._agn.nu_grid), 1))
        
        return Lnu_tot
    
    
    
    ###########################################################################
    #---- Warm Compton region
    ###########################################################################
    
    def _gene_CompInArrs(self, gammas, ktes, ktss):
        """
        Checks if any of the input paramters for nthcomp are time-series,
        and adjusts the remaining accordingly

        Parameters
        ----------
        gammas : array
            Spectral indexes.
        ktes : array
            Electron temperatures
            Units : keV
        ktss : array
            Seed photon temperatures
            Units : keV

        Returns
        -------
        gammas : array
            Spectral indexes.
        ktes : array
            Electron temperatures
            Units : keV
        ktss : array
            Seed photon temperatures
            Units : keV

        """
    
        
        #Now if any are time-series!!
        if len(gammas)==1 and len(ktes)==1 and len(ktss)==1:
            pass #no time sereis
        else:
            tsize = max(len(gammas), len(ktes), len(ktss))
            
            if len(gammas) == 1:
                gammas = np.full(tsize, gammas[0])
            
            if len(ktes) == 1:
                ktes = np.full(tsize, ktes[0])
            
            if len(ktss) == 1:
                ktss = np.full(tsize, ktss[0])
        
        
        return gammas, ktes, ktss
        
        
    
    
    def _warm_annuli(self, r, dr, gamma, ktw, fmd=1, reprocess=False, Lx=None):
        """
        Calculates the emission from an annulus within the warm Compton region

        Parameters
        ----------
        r : float
            Geometric midpoint in radial bin - Units : Rg.
        dr : float
            Bin width (linear) - Units : Rg.
        gamma : array
            Spectral index (can be time-series)
        ktw : array
            Electron temperature for warm region (can be time-series)
        fmd : array, optional
            Mass-accretion rate time-series at r, Normalised such that the mean
            is 1.
            If float or int, then assumed constant
            The defialt is float(1)
        reprocess : Bool, optional
            Switch for whether or not to include re-proccesing
        Lx : array, optional
            ONLY usefull if reprocess=True. Sets the (time-depnt) luminosity
            of the X-ray corona.
            If None, then X-ray corona assumed to be constant at Ldiss
            Units : ergs/s

        Returns
        -------
        Lnu_ann : 2D-array, shape=(len(Egrid), len(fmd))
        
        """
        
        
        T4_ann = self._agn.calc_Tnt(r, fmd)
        if reprocess == True:
            if Lx is None:
                Lx = self._agn.Lx
                
            T4_ann = T4_ann + self._agn.calc_Trep(r, Lx)
        
        kTann = self._agn.k_B * T4_ann**(1/4) #ergs
        kTann = (kTann * u.erg).to(u.keV).value #keV for nthcomp
        
        if isinstance(kTann, np.ndarray):
            pass
        else:
            kTann = np.array([kTann])
        
        gammas, ktws, kTann = self._gene_CompInArrs(gamma, ktw, kTann)
        ph_nth = mdonthcomp(self._agn.Egrid, gammas, ktws, kTann)
        #converting to erg/s/Hz
        ph_nth = (ph_nth * u.erg/u.s/u.keV).to(u.erg/u.s/u.Hz, 
                                            equivalencies=u.spectral()).value
        
        
        
        norm = self._agn.sigma_sb * T4_ann * 4*np.pi*r*dr * self._agn.Rg**2
        radiance = np.trapz(ph_nth, self._agn.nu_grid, axis=0)
        
        Lnu_ann = norm * (ph_nth/radiance)
        return Lnu_ann
    
    
    
    def warm_spec(self, fmd=1, reprocess=False, Lx=None, gamma=None, ktw=None):
        """
        Calculates the total spectrum from the warm Compton region

        Parameters
        ----------
        fmd : array, optional
            Mass-accretion rate time-series at r, Normalised such that the mean
            is 1.
            If 2D-array - then assumed to correspond to the radially dependent
                mass-accretion rate time-series. In this case the shape must be:
                shape(fmd) = (len(ts), len(rbin_edges)-1)
                i.e each column contains a time-series, and there are the same
                number of columns as radial bins
            If 1D-array it is assumed that the mass-accretion rate across the 
                entire warm region varies according to fmd
            If float or int, assumed constant
            The default is float(1)
        reprocess : Bool, optional
            Switch for whether or not to include re-proccesing
        Lx : array, optional
            ONLY usefull if reprocess=True. Sets the (time-depnt) luminosity
            of the X-ray corona.
            If None, then X-ray corona assumed to be constant at Ldiss
            Units : ergs/s
            The default is None
        gamma : array, optional
            The spectral index time-series for the warm-Comptonisation region
            If 2D-array - then assumed to be a radially dependent time-series
                for the spectral index. Shape must then be:
                shape(gamma) = (len(ts), len(rbin_edges)-1)
            If 1D-array - then assumed to be time-sereis across the entire
                warm Compton region (i.e no radial dependence)
            If None, then assumed constant as gamma_warm (input in model class)
            The default is None
        ktw : array, optional
            The electron temperature time-series for the warm-Comptonisation region
            If 2D-array - then assumed to be a radially dependent time-series
                for the electron temperature. Shape must then be:
                shape(ktw) = (len(ts), len(rbin_edges)-1)
            If 1D-array - then assumed to be time-sereis across the entire
                warm Compton region (i.e no radial dependence)
            If None, then assumed constant as kTe_warm (input in model class)
            Units : keV
            The default is None

        Returns
        -------
        Lnu_tot : array
            Output spectrum for warm Compton region - Units : erg/s/Hz.
            Has shape (len(Egrid), len(fmd)) (if fmd 2D, then len(fmd[:, 0]))

        """

        #Checking input time-series to ensure same lengths!!!
        self._check_timeseries(fmd, Lx, 'fmd', 'Lx')
        self._check_timeseries(fmd, gamma, 'fmd', 'gamma')
        self._check_timeseries(fmd, ktw, 'fmd', 'ktw')
        self._check_timeseries(gamma, Lx, 'gamma', 'Lx')
        self._check_timeseries(ktw, Lx, 'gamma', 'Lx')
        self._check_timeseries(ktw, gamma, 'gamma', 'ktw')
        
        
        #Making gamma and ktw array inputs (makes life easier later...)
        if isinstance(gamma, np.ndarray):
            pass
        else:
            if gamma is None:
                gamma = self._agn.gamma_w
            gamma = np.array([gamma])
        
        if isinstance(ktw, np.ndarray):
            pass
        else:
            if ktw is None:
                ktw = self._agn.kTw
            ktw = np.array([ktw])
        
    
        #Calculating warm spec if it exists!
        if len(self._agn.logr_wc_bins) > 1:
            for i in range(len(self._agn.logr_wc_bins) - 1):
                dr_bin = 10**self._agn.logr_wc_bins[i] - 10**self._agn.logr_wc_bins[i+1]
                rmid = 10**(self._agn.logr_wc_bins[i] - self._agn.dlog_r/2)

                if isinstance(fmd, (float, int)) or fmd.ndim==1:
                    fmdi = fmd
                else:
                    fmdi = fmd[:, i]
            
                if gamma.ndim==1:
                    gammai = gamma
                else:
                    gammai = gamma[:, i]
            
                if ktw.ndim==1:
                    ktwi = ktw
                else:
                    ktwi = ktw[:, i]
                
                if isinstance(Lx, (int, float)) or Lx is None or Lx.ndim==1:
                    Lxi = Lx
                else:
                    Lxi = Lx[:, i]
              
                Lnu_r = self._warm_annuli(rmid, dr_bin, gammai, ktwi,fmdi, 
                                      reprocess, Lxi)
                if i == 0:
                    Lnu_tot = Lnu_r
                else:
                    Lnu_tot += Lnu_r
        
        else:
            Lnu_tot = np.zeros((len(self._agn.nu_grid), 1))
        
        return Lnu_tot
    
    
    
    ###########################################################################
    #---- Hot Compton region
    ###########################################################################
    
    def _hot_annuli(self, r, dr, gamma, kte, kts, fmd=1, Lseed=None):
        """
        Calculates the spectrum from an annulus within the hot Compton region

        Parameters
        ----------
        r : float
            Geometric midpoint in radial bin - Units : Rg.
        dr : float
            Bin width (linear) - Units : Rg.
        gamma : array
            Spectral index (can be time-series).
        kte : array
            Electron temperature (can be time-series).
            Units : keV
        kts : array
            Seed photon temperature (can be time-series).
            Units : keV
        fmd : array, optional
            Time series of mass accretion rate within hot flow at r
            If float or int, then assumed constant
            The default is int(1)

        Returns
        -------
        Lnu_ann : 2D-array, shape=(len(Egrid), len(fmd))

        """
        
        #Power dissipated in corona
        T4_ann = self._agn.calc_Tnt(r, fmd)
        Ld_ann = self._agn.sigma_sb * T4_ann * 4*np.pi*r*dr * self._agn.Rg**2 #erg/s

        
        #Seed photon power
        if Lseed is None:
            Lseed = self.calc_Lseed()
        
        Ls_ann = Lseed/(len(self._agn.logr_hc_bins) - 1) #Need to divide by number of annuli
        
        #Calculating spectral shape
        gammas, ktes, ktss = self._gene_CompInArrs(gamma, kte, kts)
        ph_nth = mdonthcomp(self._agn.Egrid, gammas, ktes, ktss)
        #converting to erg/s/Hz
        ph_nth = (ph_nth * u.erg/u.s/u.keV).to(u.erg/u.s/u.Hz, 
                                            equivalencies=u.spectral()).value
        
        #normalising by luminsoity in annulus
        norm = Ld_ann + Ls_ann
        radiance = np.trapz(ph_nth, self._agn.nu_grid, axis=0)
        
        Lnu_ann = norm * (ph_nth/radiance)
        return Lnu_ann
    
    
    
    def hot_spec(self, fmd=1, gamma=None, kte=None, kts=None, Lseed=None):
        """
        Calculates the emission from the Hot Compton region

        Parameters
        ----------
        fmd : array, optional
            Mass-accretion rate time-series at r, Normalised such that the mean
            is 1.
            If 2D-array - then assumed to correspond to the radially dependent
                mass-accretion rate time-series. In this case the shape must be:
                shape(fmd) = (len(ts), len(rbin_edges)-1)
                i.e each column contains a time-series, and there are the same
                number of columns as radial bins
            If 1D-array it is assumed that the mass-accretion rate across the 
                entire warm region varies according to fmd
            If float or int, assumed constant
            The default is float(1)
        gamma : array, optional
            The spectral index time-series for the warm-Comptonisation region
            If 2D-array - then assumed to be a radially dependent time-series
                for the spectral index. Shape must then be:
                shape(gamma) = (len(ts), len(rbin_edges)-1)
            If 1D-array - then assumed to be time-sereis across the entire
                hot Compton region (i.e no radial dependence)
            If None, then assumed constant as gamma_h (either given as input
                or calculated upon intitiation)
            The default is None
        kte : array, optional
            The electron temperature of the hot Compton region
            If 2D-array - then assumed to be radially dependent time-series
                Shape must then be: shape(kte) = (len(ts), len(rbin_edges)-1)
            If 1D-array - the assumed to be time-series across the entire
                hot Compton region (i.e no radial dependence)
            If None, then assumed constant as kTh (either given as input
                or calculated upon initiation, depending on the model)
            Units : keV
            The default is None.
        kts : array, optional
            The seed-photon temperature
            If 2D-array - then assumed to be radially dependent time-series
                Shape must then be: shape(kte) = (len(ts), len(rbin_edges)-1)
            If 1D-array - the assumed to be time-series across the entire
                hot Compton region (i.e no radial dependence)
            If None, then assumed constant as kTseed (calculated upon initiation)
            Units : keV
            The default is None.
        Lseed : array or float, optional
            Seed photon luminosity
            If array, then assumed to be a time-series
            If float or int assumed to be constant
            If None, then uses value calculated upon initiating the model
            The default is None.

        Returns
        -------
        Lnu_tot : array
            Output spectrum for hot Compton region - Units : erg/s/Hz.
            Has shape (len(Egrid), len(fmd)) (if fmd 2D, then len(fmd[:, 0]))

        """
       
        #Performing input checks and passing to arrays
        fmd, gamma, kte, kts, Lseed = self._hot_checks(fmd, gamma, kte, kts, Lseed)

        #Now calculating spectrum if hot region exists
        if len(self._agn.logr_hc_bins) > 1:
            for i in range(len(self._agn.logr_hc_bins) - 1):
                dr_bin = 10**self._agn.logr_hc_bins[i] - 10**self._agn.logr_hc_bins[i+1]
                rmid = 10**(self._agn.logr_hc_bins[i] - self._agn.dlog_r/2)
   
                #Checking dimensions
                if fmd.ndim == 1:
                    fmdi = fmd
                else:
                    fmdi = fmd[:, i]
            
                if gamma.ndim == 1:
                    gammai = gamma
                else:
                    gammai = gamma[:, i]
            
                if kte.ndim == 1:
                    ktei = kte
                else:
                    ktei = kte[:, i]
            
                if kts.ndim == 1:
                    ktsi = kts
                else:
                    ktsi = kts[:, i]
            
                if Lseed.ndim == 1:
                    Lseedi = Lseed
                else:
                    Lseedi = Lseed[:, i]
            
            
                #Getting spec
                Lnu_r = self._hot_annuli(rmid, dr_bin, gammai, ktei, ktsi,
                                     fmdi, Lseedi)
            
                if i == 0:
                    Lnu_tot = Lnu_r
                else:
                    Lnu_tot += Lnu_r
        
        else:
            Lnu_tot = np.zeros((len(self._agn.nu_grid), 1))
        
        return Lnu_tot
    
    
    
    def _hot_checks(self, fmd, gamma, kte, kts, Lseed):
        """
        Simply performs checks on input paramters for the hot flow, and passes
        to array for simplicity when performing calculations

        Parameters
        ----------
        fmd : TYPE
            DESCRIPTION.
        gamma : TYPE
            DESCRIPTION.
        kte : TYPE
            DESCRIPTION.
        kts : TYPE
            DESCRIPTION.
        Lseed : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        #Checking input time-series to ensure same lengths!!!
        self._check_timeseries(fmd, gamma, 'fmd', 'gamma')
        self._check_timeseries(fmd, kte, 'fmd', 'kte')
        self._check_timeseries(fmd, kts, 'fmd', 'kts')
        self._check_timeseries(fmd, Lseed, 'fmd', 'Lseed')
        self._check_timeseries(gamma, kte, 'gamma', 'kte')
        self._check_timeseries(gamma, kts, 'gamma', 'kts')
        self._check_timeseries(gamma, Lseed, 'gamma', 'Lseed')
        self._check_timeseries(kte, kts, 'kte', 'kts')
        self._check_timeseries(kte, Lseed, 'kte', 'Lseed')
        self._check_timeseries(kts, Lseed, 'kte', 'Lseed')
        
        
        #Making everything array input to make life easier
        if isinstance(fmd, np.ndarray):
            pass
        else:
            fmd = np.array([fmd])
        
        if isinstance(gamma, np.ndarray):
            pass
        else:
            if gamma is None:
                gamma = self.calc_gammah()
            gamma = np.array([gamma])

        if isinstance(kte, np.ndarray):
            pass
        else:
            if kte is None:
                kte = self.calc_kTe(self.calc_gammah())
            kte = np.array([kte])
        
        if isinstance(kts, np.ndarray):
            pass
        else:
            if kts is None:
                kts = self.calc_kTseed()
            kts = np.array([kts])
        
        if isinstance(Lseed, np.ndarray):
            pass
        else:
            if Lseed is None:
                Lseed = self.calc_Lseed()
            Lseed = np.array([Lseed])
        
        
        return fmd, gamma, kte, kts, Lseed
    
    
    ###########################################################################
    #---- Hot flow properties
    #     These go into the hot_spec and _hot_annuli calculations!
    ###########################################################################
    

    def calc_Ldiss(self, fmd=1):
        """
        Calculates the power dissipated in the hot-corona

        Parameters
        ----------
        fmd : array, optional
            Mass-accretion rate time-series at r, Normalised such that the mean
            is 1.
            NOTE: This is the time-series as seen by the hot corona!
            i.e does not necessarily correspond to the mdot fluctuations seen
            by the UV disc
            If 2D-array - then assumed to correspond to the radially dependent
                mass-accretion rate time-series. In this case the shape must be:
                shape(fmd) = (len(ts), len(rbin_edges)-1)
                i.e each column contains a time-series, and there are the same
                number of columns as radial bins
            If 1D-array it is assumed that the mass-accretion rate across the 
                entire hot region varies according to fmd
            If float or int, assumed constant
            The default is 1.
            
        Returns
        -------
        Ldiss : array
            Total luminosity dissipated within the corona
            Units : erg/s

        """

        drs = 10**self._agn.logr_hc_bins[:-1] - 10**self._agn.logr_hc_bins[1:]
        rms = 10**(self._agn.logr_hc_bins[:-1] - self._agn.dlog_r/2)
        
        #T4s = self.calc_Tnt(rms, fmd)
        if isinstance(fmd, (int, float)) or fmd.ndim==2:
            T4s = self._agn.calc_Tnt(rms, fmd)
        
        #Need to explicitly define axis when two 1D-arrays so python doesnt get confused!
        else:
            T4s = self._agn.calc_Tnt(rms, fmd[:, np.newaxis])


        Ldiss = np.sum(self._agn.sigma_sb*T4s * 4*np.pi*rms*drs * self._agn.Rg**2, axis=-1)
        return Ldiss
            
        
        
    def calc_Lseed(self, fmd=1):
        """
        Calculates the seed photon luminosity seen by the hot corona
        
        NOTE! Although this method can return time-series, it does NOT calculate
        the time-delay. It is assumed that any delay between respective disc
        annuli is already encoded within fmd.
        Use calc_seedDelay method to create correctly shifted fmd time-series
        
        Parameters
        ----------
        fmd : array, optional
            Mass-accretion rate time-series at r, Normalised such that the mean
            is 1.
            NOTE: This is the time-series of the UV disc as seen by the Corona!
            i.e does not necessarily correspond to the mdot fluctuations within
            the corona (because its the disc fluctuations that set Lseed!!!)
            If 2D-array - then assumed to correspond to the radially dependent
                mass-accretion rate time-series. In this case the shape must be:
                shape(fmd) = (len(ts), len(logr_grd)-1)
                i.e each column contains a time-series, and there are the same
                number of columns as radial bins
            If 1D-array it is assumed that the mass-accretion rate across the 
                entire disc/warm region varies according to fmd AND has the
                same light-travel time to the corona
            If float or int, assumed constant
            The default is 1.

        
        Returns
        -------
        Lseed : array
            Total seed photon luminosity as seen by the corona
            Units : ergs/s

        """
        
        
        if isinstance(fmd, (int, float)) or fmd.ndim == 1:
            xt_ad = fmd
            xt_wc = fmd
            
        else:
            #if 2d then need to split up into disc and warm region
            idx_split = len(self._agn.logr_ad_bins) - 1
            xt_ad = fmd[:, :idx_split]
            xt_wc = fmd[:, idx_split:]
            
        
        Ls_ad = self._lseed_region(self._agn.logr_ad_bins, self._agn.dlogr_ad,
                                   xt_ad)
        Ls_wc = self._lseed_region(self._agn.logr_wc_bins, self._agn.dlogr_wc,
                                   xt_wc)
        Lseed = Ls_ad + Ls_wc
        
        return Lseed
 
    
    def _lseed_region(self, logr_bins, dlogr, fmd):
        """
        Calculates the seed photon luminosity contribution from a region
        (e.g disc or warm corona)

        Parameters
        ----------
        logr_bins : array
            Radial bin edges for region
        dlogr : float
            Radial bin spacing
        fmd : array
            Mass accretion rate variations within region

        Returns
        -------
        None.

        """
        
        drs = 10**(logr_bins[:-1]) - 10**(logr_bins[1:])
        rms = 10**(logr_bins[:-1] - dlogr/2)
            
        if isinstance(fmd, (int, float)) or fmd.ndim==2:
            T4s = self._agn.calc_Tnt(rms, fmd)
        else:
            T4s = self._agn.calc_Tnt(rms, fmd[:, np.newaxis])
        
        th0 = np.arcsin(self._agn.hmax/rms)
        cov_frac = (1/np.pi) * (th0 - 0.5*np.sin(2*th0))

        Ls_all = self._agn.sigma_sb*T4s * 4*np.pi*rms*drs * cov_frac * self._agn.Rg**2
        Lseed_region = np.sum(Ls_all, axis=-1)
        return Lseed_region
        
    
    def calc_gammah(self, Ldiss=None, Lseed=None):
        """
        Calculates spectral index for hot compton region
        
        Uses Eqn. 14 in Beloborodov (1999) [is also seen in Kubota & Done 2018]
        

        Parameters
        ----------
        Ldiss : float or array, optional
            Power dissipated through the accretion flow within the hot Compton
            region.
            IF None, then uses Ldiss calculated for input accretion params. 
            The default is None.
        Lseed : float or array, optional
            Seed photon luminosity seen by the hot corona
            If None, then uses Lseed calculated for input accretion params
            The default is None.

        Returns
        -------
        gamma_h : float or array
            Spectral index of hot Compton region

        """
        if Ldiss is None:
            Ldiss = self.calc_Ldiss()
            
        if Lseed is None:
            Lseed = self.calc_Lseed()
        
        gamma_h = (7/3) * (Ldiss/Lseed)**(-0.1)
        return gamma_h
    
    
    def calc_kTe(self, gamma):
        """
        Calculates the electron temperature for a givn photon index.
        
        Follows Beloborodov (1999) to determine first the Compton y parameter
        through:
                y = (4*gamma/9)**(-9/2)
        
        Then uses the relation (again in Beloborodov 1999) between the y-parameter
        and electron temperature to calculate the temperature.
        This assumes a constant Thomson optical depth (given when initiating 
        the model!!)

        Parameters
        ----------
        gamma : float or array
            Spectral index for Comptonisation.

        Returns
        -------
        kTe : float or array
            Electron temperature for Comptonisation - Units : keV.

        """
        
        #There are two classes of model in self._agn
        #One where you pass thompson optical depth - in which case kTe must
        #be calculated.
        #And one where kTe is passed directly for the hot region as kTh
        
        if hasattr(self._agn, 'tau_t'):
            y = ((4*gamma)/9)**(-9/2) #Compton y-param
        
            #Dimensionless electron temperature
            theta_e = 4*y/(self._agn.tau_t*(self._agn.tau_t+1))
            theta_e += 1
            theta_e = np.sqrt(theta_e)
            theta_e *= (1/8)
            theta_e -= (1/8)

            #Converting to physical uits
            kTe = theta_e * self._agn.m_e * self._agn.c**2 #erg
            kTe = (kTe * u.erg).to(u.keV).value #keV
        
        else:
            kTe = self._agn.kTh
            
            if isinstance(gamma, np.ndarray):
                kTe = np.full(np.shape(gamma), kTe)
            else:
                pass
        
        return kTe
    
    
    def calc_kTseed(self, fmd=1, gamma=None):
        """
        Calculates the seed photon temperature, for seed photons entering the
        hot Compton region
        These are assumed to be dominated by the inner edge of the disc, hence
        only considers emission originating from r_hot

        Parameters
        ----------
        fmd : array, optional
            Mass-accretion rate time-series at r_hot, Normalised such that the mean
            is 1.
            NOTE: This is the time-series of the inner-edge of the UV disc as 
            seen by the Corona! Hence, should have any potential time-delay
            encoded within it!!
            i.e does not necessarily correspond to the mdot fluctuations within
            the corona (because its the disc fluctuations that set Lseed!!!)
            If 1D-array it is assumed that this is a time-series
            If float or int, assumed constant
            The default is 1.
        gamma : float or array
            Spectral index of warm Comptonisation region, inner edge
            If float, then assumed constant
            If array, the asuumed time-series
            If None, then assumed self.gamma_w from self._agn

        Returns
        -------
        kT_seed : array
            Seed-photon temperature for the Hot-Compton region
            Units : keV

        """
        if gamma is None:
            gamma = self._agn.gamma_w
        
        T4_edge = self._agn.calc_Tnt(self._agn.rh, fmd) #Disc temp at inner edge
        Tedge = T4_edge**(1/4)
        
        kT_seed = self._agn.k_B * Tedge #ergs
        kT_seed = (kT_seed * u.erg).to(u.keV).value #to keV
        if hasattr(self._agn, 'rw') and self._agn.rw != self._agn.rh:
            #If warm compton region exists, then include compton y-param
            ysb = (gamma * (4/9))**(-4.5)
            kT_seed *= np.exp(ysb)
        
        else:
            pass
        
        return kT_seed
    
    
            
    
    ###########################################################################
    #---- Checking methods
    ###########################################################################
    
    
    def _check_timeseries(self, in1, in2, name1='in1', name2='in2'):
        """
        Checks that two input time-series to spec methods are compatible with one
        another. i.e not different lengths!!!

        Parameters
        ----------
        in1 : array
            Input 1
            This can be 2D
        in2 : array
            Inpu2
            This can be 2D

        Returns
        -------
        None.

        """
        
        #Checking input time-series to ensure same lengths!!!
        if isinstance(in1, np.ndarray) and isinstance(in2, np.ndarray):
            if in1.ndim == 1 and in2.ndim == 1:
                if len(in1) == 1 or len(in2) == 1:
                    pass
                elif len(in1) != len(in2):
                    raise ValueError(f'{name1} and {name2} have different lengths!'
                                     'These must be the same!!!')
                else:
                    pass
            
            elif in1.ndim == 2 and in2.ndim == 1:
                if len(in2) == 1:
                    pass
                elif len(in1[:, 0]) != len(in2):
                    raise ValueError(f'{name1} and {name2} have different lengths!'
                                     'These must be the same!!!')
                else:
                    pass
            
            elif in1.ndim == 1 and in2.ndim == 2:
                if len(in1) == 1:
                    pass
                elif len(in1) != len(in2[:, 0]):
                    raise ValueError(f'{name1} and {name2} have different lengths!'
                                     'These must be the same!!!')
                else:
                    pass
    
            elif in1.ndim == 2 and in2.ndim == 2:
                if len(in1[:, 0]) != len(in2[:, 0]):
                    raise ValueError(f'{name1} and {name2} have different lengths!'
                                     'These must be the same!!!')
                else:
                    pass
    
    