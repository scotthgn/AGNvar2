#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 14:21:11 2023

@author: Scott Hagen


Contains methods that will calculate the model power-spectrum for inward
propogating mass-accretion rate fluctuations, for each radial annulus as well
as total.
Additionally, contains methods for creating realisations of mdot time-series
from these model power spectra - for each radial annulus.

"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from tqdm import tqdm

class _PropFluc:
    
    def __init__(self,
                 AGNobj,
                 N,
                 dt):
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
        N : int
            Number of points in time-series
        dt : float
            Time-step in time series
            Units : s
        
        Returns
        -------
        None.

        """
        
        #read input object and pars
        self._agn = AGNobj
        self.N = N
        self.dt = dt
        
        
        #generating frequency and time grids
        self._gene_ffreq()
        self.ts = np.arange(0, self.N*self.dt, self.dt)
    
    
    
    ###########################################################################
    #---- Model power spectra and variability
    ###########################################################################
    
    def set_generativeVar(self, rv_max=-1, Fvd=0.1, Fvw=0.5, Fvh=0.01, Bd=0.05, 
                          Bw=0.001, Bh=20, md=0.5, mw=-3/2, mh=1, hpfrac=1):
        """
        Sets the variability within each region and generates generateve
        frequency arrays

        Parameters
        ----------
        rv_max : float, optional
            Sets the maximum radius that generates variability
            If -ve, then the entire flow is assumed to be varying
            The default is -1
            Units : Rg
        Fvd : float, optional
            Fractional variability per radial decade in the disc region.
            The default is 0.1.
        Fvw : float, optional
            Fractional variability per radial decade in the warm Compton region. 
            The default is 0.2.
        Fvh : float, optional
            Fractional variability per radial decade in the hot Compton region. 
            The default is 0.5.
        Bd : float, optional
            Generative frequency amplitude in the disc region. 
            The default is 0.05.
        Bw : float, optional
            Generative frequency amplitude in the warm region. 
            The default is 0.5.
        Bh : float, optional
            Generative frequency amplitude in the hot region. 
            The default is 2.
        md : float, optional
            Generative frequency power-law index in the disc region. 
            The default is 0.1.
        mw : float, optional
            Generative frequency power-law index in the warm region.
            The default is 1.
        mh : float, optional
            Generative frequency power-law index in the hot region. 
            The default is 1.
        hpfrac : float, optional
            Fraction of mdot fluctuations that propagate into the hot corona
            This essentially re-scales the variance in the warm/disc power
            spectrum before propogating into the hot corona
            The defaults is 1
        
        Returns
        -------
        None.

        """
        
        self.Fvd = Fvd
        self.Fvw = Fvw
        self.Fvh = Fvh
        
        if rv_max <= 0:
            self.rv_max = self._agn.rout
        else:
            self.rv_max = rv_max
        
        self.hpfrac = hpfrac
        
        if len(self._agn.logr_ad_bins) > 1:
            rdm = 10**(self._agn.logr_ad_bins[:-1] - self._agn.dlogr_ad/2)
            self.fg_ad = self._fgen(rdm, Bd, md)
        else:
            self.fg_ad = None
        
        if len(self._agn.logr_wc_bins) > 1:
            rwm = 10**(self._agn.logr_wc_bins[:-1] - self._agn.dlogr_wc/2)
            self.fg_wc = self._fgen(rwm, Bw, mw)
        else:
            self.fg_wc = None
        
        if len(self._agn.logr_hc_bins) > 1:
            rhm = 10**(self._agn.logr_hc_bins[:-1] - self._agn.dlogr_hc/2)
            self.fg_hc = self._fgen(rhm, Bh, mh)
        else:
            self.fg_hc = None
            
    
    def gene_mdotPspec(self):
        """
        Generates model power-spectrum of the mass-acretion rate time-series
        at each annulus (from outer to inner!)
        
        Returns
        -------
        None.

        """
        
        if hasattr(self, 'Fvw'):
            pass
        else:
            print('Variability pars have not been set yet!!! Using default '
                  'values. Set these yourself with .set_generativeVar()')
            self.set_generativeVar()
        
        
        AC = None
        #Disc region
        if self.fg_ad is None:
            self.pow_dsc_iann = np.zeros((len(self.fs), 1))
            self.pow_dsc_tann = np.zeros((len(self.fs), 1))
            self.ACdsc = np.zeros(len(self.fs))
        else:
            AC, self.pow_dsc_iann, self.pow_dsc_tann = self._mod_regionPspec(
                AC, self.fg_ad, self._agn.dlogr_ad, self.Fvd, 1)
            self.ACdsc = AC
        
        #warm region
        if self.fg_wc is None:
            self.pow_wrm_iann = np.zeros((len(self.fs), 1))
            self.pow_wrm_tann = np.zeros((len(self.fs), 1))
            self.ACwrm = np.zeros(len(self.fs))
        else:
            AC, self.pow_wrm_iann, self.pow_wrm_tann = self._mod_regionPspec(
                AC, self.fg_wc, self._agn.dlogr_wc, self.Fvw, 1)
            self.ACwrm = AC
        
        #Hot region
        if self.fg_hc is None:
            self.pow_hot_iann = np.zeros((len(self.fs), 1))
            self.pow_hot_tann = np.zeros((len(self.fs), 1))
            self.AChot = np.zeros(len(self.fs))
        else:
            AC, self.pow_hot_iann, self.pow_hot_tann = self._mod_regionPspec(
                AC, self.fg_hc, self._agn.dlogr_hc, self.Fvh, self.hpfrac)
            self.AChot = AC
            
    
        return self.ACdsc, self.ACwrm, self.AChot
    
    
    def _mod_regionPspec(self, AC, fgs, dlogr, Fv, pfrac):
        """
        Calculates the power spec from a single spectral region

        Parameters
        ----------
        AC : array
            Input power-spectrum
            This will be the output from the inner radius of the previous
            region
        fgs : array
            Generative frequencies
        dlogr : float
            Radial grid spacing
        Fv : float
            Fractional variability per radial decade
        pfrac : float
            Fraction of variability that passes from one region to the next

        Returns
        -------
        AC : array
            Current model power-spectrum
        pow_iann : 2D-array
            Intrinsic power-spectrum at each radius within region
        pow_tann : 2D-array
            Total power-spectrum at each radius within region

        """
        
        pow_iann = np.ndarray((len(self.fs), len(fgs))) #intrinsic to annulus
        pow_tann = np.ndarray((len(self.fs), len(fgs))) #total at annulus
        
        Ndec = 1/dlogr
        var = (Fv/np.sqrt(Ndec))**2
        amp = var/(self.dt * self.N * np.pi)
        for i, fg in enumerate(fgs):
            Anew = self._Lorentzian(fg, amp)
            pow_iann[:, i] = Anew

            if i == 0 and AC is None:
                AC = Anew
            elif i == 0 and AC is not None:
                XJ = fftconvolve(Anew, AC, mode='same')
                AC = np.sqrt(pfrac) * XJ - np.sqrt(pfrac) * Anew + Anew
            else:
                AC = fftconvolve(Anew, AC, mode='same')
            
            pow_tann[:, i] = AC
            
            
        return AC, pow_iann, pow_tann
            
        
    
    
    ###########################################################################
    #---- Generating mdot realisations
    ###########################################################################
    
    
    def gene_mdotRealisation(self, fpd, fpw, fph):
        """
        Generates realisations of the mass-accretion rate fluctuations at
        each annulus
        
        Also does the propogation
        
        Note - must do one at a time - because if you try to do several
        thousand realisations simoultaneously you quickly run out of ram...!
        (or at least I do)
        
        Parameters
        ----------
        fpd : int or float
            Propogation factor for disc (i.e how much faster than the generative
            time-scale do fluctuations propogate downwards)
            This is such that fprop = fpd * fgen
        fpw : int or float
            Propogation factor for warm region
        fph : int or float
            Propogation factor for hot region

        Returns
        -------
        None.

        """
        #Uses the model power-spectra to generate realisations
        if hasattr(self, 'AChot'):
            pass
        else:
            self.gene_mdotPspec()
        
        xt_dsc_iann = self._TimmerKoenig(self.pow_dsc_iann)
        xt_wrm_iann = self._TimmerKoenig(self.pow_wrm_iann)
        xt_hot_iann = self._TimmerKoenig(self.pow_hot_iann)
        
        setattr(self, 'xt_dsc_iann', xt_dsc_iann)
        setattr(self, 'xt_wrm_iann', xt_wrm_iann)
        setattr(self, 'xt_hot_iann', xt_hot_iann)

        #Disc region
        xt_tot = None
        dti = None
        dttot = 0
        if self.fg_ad is None:
            xt_dsc_tann = xt_dsc_iann
        else:
            xt_dsc_tann, dti, dttot = self._prop_xt(xt_tot, dti, xt_dsc_iann, 
                                        fpd*self.fg_ad, self._agn.logr_ad_bins, 
                                        self._agn.dlogr_ad) 
            xt_tot = xt_dsc_tann[:, -1]
        
        #warm region
        if self.fg_wc is None:
            xt_wrm_tann = xt_wrm_iann
            xt_tot = None
        else:
            xt_wrm_tann, dti, dttot_wc = self._prop_xt(xt_tot, dti, xt_wrm_iann, 
                                        fpw*self.fg_wc, self._agn.logr_wc_bins, 
                                        self._agn.dlogr_wc)
            xt_tot = xt_wrm_tann[:, -1]
            dttot += dttot_wc
            
        #hot region
        if self.fg_hc is None:
            xt_hot_tann = xt_hot_iann
            xt_tot = None
        else:
            if self.hpfrac == 1 or xt_tot is None:
                pass
            else:
                xt_tot = self._reScale_xtvar(xt_tot, self.hpfrac)
            
            xt_hot_tann, dti, dttot_hc = self._prop_xt(xt_tot, dti, xt_hot_iann, 
                                    fph*self.fg_hc, self._agn.logr_hc_bins, 
                                    self._agn.dlogr_hc)
                
            dttot += dttot_hc
        
        if np.amin(xt_dsc_tann) <= 0:
            print('ARRRRGH! Too much variability in standard dsic. i.e Fvd'
                  ' too big!!! mdot has gone negative - Reduce Fvd!')
            print()
            print('Exiting program - too much variability')
            exit()
        
        if np.amin(xt_wrm_tann) <= 0:
            print('ARRRRGH! Too much variability in warm Compton region. i.e Fvw'
                  ' too big!!! mdot has gone negative - Reduce Fvw!')
            print()
            print('Exiting program - too much variability')
            exit()
        
        if np.amin(xt_hot_tann) <= 0:
            print('ARRRRGH! Too much variability in hot Compton region. i.e Fvh'
                  ' too big!!! mdot has gone negative - Reduce Fvh')
            print()
            print('Exiting program - too much variability')
            exit()
        
        return xt_dsc_tann, xt_wrm_tann, xt_hot_tann, dttot
    
    
    
    def _prop_xt(self, xt_tot, dti, xt_iann, fprp, log_rbins, dlog_r):
        """
        Propogates the mdot time-series through a region in the flow

        Parameters
        ----------
        xt_tot : array or None
            The current mdot from the previous region
        dti : float or None
            Propogation time from previous region
        xt_iann : 2D-array
            Intrinsic mdot for each annulus in region
        fprp : 1D-array
            Propogation frequency for each annulus in region
        log_rbins : 1D-array
            Log radial bin edges for region
        dlog_r : float
            Radial bin width (in log)

        Returns
        -------
        None.

        """
        
        #rmids = 10**(log_rbins[:-1] - dlog_r/2)
        #drs = 10**(log_rbins[:-1]) - 10**(log_rbins[1:])      
        
        if xt_tot is None:
            xt_interp = interp1d(self.ts, xt_iann[:, 0], kind='linear',
                                 bounds_error=False, fill_value=1)
        else:
            xt_interp = interp1d(self.ts, xt_tot, kind='linear',
                                 bounds_error=False, fill_value=1)

        dttot = 0.
        res_warning = False
        for i, fp in enumerate(fprp):
            if fp == 0:
                if i == 0:
                    xt_tann = xt_iann[:, i]
                else:
                    xt_tann = np.column_stack((xt_tann, xt_iann[:, i]))
            else:
                if dti is None:
                    xt_anni = xt_iann[:, i]
                else:
                    xt_anni = xt_interp(self.ts - dti) * xt_iann[:, i]
            
                #Updating interpolated xt
                xt_interp = interp1d(self.ts, xt_anni, kind='linear',
                                 bounds_error=False, fill_value=1)
            
                if i == 0:
                    xt_tann = xt_anni
                else:
                    xt_tann = np.column_stack((xt_tann, xt_anni))
            
                dti = dlog_r * (1/fp)
                dttot += dti
            
                if dti > self.dt and res_warning==False:
                    print()
                    print('-------------------------------------------------------')
                    print('WARNING!!! Sampling frequency is higher than propagation'
                          ' frequency!!! Either increase radial grid resolution,'
                          f' currently {self._agn.dr_dex} bins per decade, or '
                          f' reduce sampling time, currently {self.dt} s')
                    
                    print()
                    print(f'Minimum needed resolution: {1/(fp*self.dt)} bins per decade')
                    print()
                    print('-------------------------------------------------------')
                
                
                    res_warning = True
            
        
        
        return xt_tann, dti, dttot
        
    
        
        
    
    
    ###########################################################################
    #---- Time scales
    ###########################################################################
    
    def _fgen(self, r, B, m):
        """
        Generative frequency at some radius r

        Parameters
        ----------
        r : float or array
            Radius considered
            Units : Rg
        B : float
            Power-law amplitude
        m : float
            Power-law index

        Returns
        -------
        fgen : float or array
            Generative frequency at r
            Units : Hz

        """
        
        fgen = (1/(2*np.pi)) * r**(-3/2) #Keplerian frequency in c/Rg
        fgen *= B * r**(-m)
        fgen *= (self._agn.c/self._agn.Rg) #converting to Hz
        
        #now checking that below rv_max
        fgen[r > self.rv_max] = 0
        
        return fgen
    
    
    
    ###########################################################################
    #---- Background methods
    ###########################################################################
    
    def _gene_ffreq(self):
        """
        Generates an evenly spaced grid in Fourier frequency based off the 
        class input parameters N (number of points in time-series) and dt
        (time-step)
        
        Stores the grid as a class attribute

        """
        
        if self.N % 2 == 0:
            #Generating -ve AND +ve Fourier frequencies
            self.fs = np.linspace(-(0.5*self.N)/(self.N*self.dt),
                                  (0.5*self.N)/(self.N*self.dt), int(self.N))
            
            #inserting a 0 (DC component) between -ve and +ve fourier frequencies!!!
            self.fs = np.concatenate((self.fs[:(int(self.N)//2)], np.zeros(1), 
                                      self.fs[(int(self.N)//2):]))
        
        else:
            self.fs = np.linspace(-0.5*(self.N-1)/(self.N*self.dt),
                                  0.5*(self.N-1)/(self.N*self.dt), int(self.N))
            
            self.fs = np.concatenate((self.fs[self.fs<0], np.zeros(1),
                                      self.fs[self.fs>0]))

    
    def _Lorentzian(self, fv, amp):
        """
        Creates a zero centered Lorentzian as a function of Fourier frequency

        Parameters
        ----------
        fv : float
            Width of the Lorentzian.
        amp : float
            Amplitude.

        Returns
        -------
        lz : array
            The Lorentzian...

        """
        lz = amp * (fv/(fv**2 + self.fs**2))
        lz[self.fs==0] = 1 #setting DC component to 1 - hardwires mean=1
        return lz
    
    
    
    
    def _TimmerKoenig(self, PS):
        """
        Generates a realisation of a time-series following the method of
        Timmer & Koenig (1995), for each radial annulus in PS
        

        Parameters
        ----------
        PS : 2D-array
            Input power-spectrum for each radial annulus

        Returns
        -------
        xt : 2D-array
            Output time-series for each radial annulus


        """
        
        #initiating fourier transfomr array - only using +ve frequencies
        FT = np.zeros(np.shape(PS[self.fs>0, :]), dtype=complex)
        
        #Drawing realisation from Gaussian distribution
        FT.real = np.random.normal(0, 1, np.shape(PS[self.fs>0, :]))
        FT.imag = np.random.normal(0, 1, np.shape(PS[self.fs>0, :]))
        
        FT *= np.sqrt(0.5 * PS[self.fs>0, :]) #modulating by power-spec
        
        #Since LC always real, if N even then f_nyquist must be real
        if self.N % 2 == 0:
            FT.imag[-1, :] = 0
        
        #creating time-series through inverse fft
        xt = np.fft.irfft(FT, self.N, norm='forward', axis=0) + 1
        return xt
    
    
    def _reScale_xtvar(self, xt, frac):
        """
        Re-scales array containing time-sereies to new varaince
        This is so we can consider case where not all variability propagates
        into the hot corona

        Parameters
        ----------
        xt : 2D-array
            Contains times-series for each annulus from some region
        frac : float
            Fraction to scale the variance by
            

        Returns
        -------
        None.

        """
        var = np.var(xt, axis=0)
        var_new = frac*var
        
        xt_new = (xt - np.mean(xt, axis=0))/np.sqrt(var)
        xt_new *= np.sqrt(var_new)
        xt_new += 1
        return xt_new
        
        
    
    
    ###########################################################################
    #---- Plotting output - for de-bugging and testing reasons!!
    ###########################################################################
    
    def plot_fgen(self):
        """
        Plots the generative frequency as function of radius

        Returns
        -------
        None.

        """
        
        if hasattr(self, 'fg_ad'):
            pass
        else:
            self.set_generativeVar()
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        
        rmd = 10**(self._agn.logr_ad_bins[:-1] - self._agn.dlogr_ad)
        rmw = 10**(self._agn.logr_wc_bins[:-1] - self._agn.dlogr_wc)
        rmh = 10**(self._agn.logr_hc_bins[:-1] - self._agn.dlogr_hc)
        
        if self.fg_ad is None:
            pass
        else:
            ax.loglog(rmd, self.fg_ad, color='red')
        
        if self.fg_wc is None:
            pass
        else:
            ax.loglog(rmw, self.fg_wc, color='green')
        
        if self.fg_hc is None:
            pass
        else:
            ax.loglog(rmh, self.fg_hc, color='blue')
        
        ax.set_xlabel('Radius, r   (Rg)')
        ax.set_ylabel(r'$f_{gen}$   (Hz)')
        
        plt.show()        
    
    def plot_modPspec(self):
        """
        Plots model power-spectrum and its components

        Returns
        -------
        None.

        """
        
        if hasattr(self, 'AChot'):
            pass
        else:
            self.gene_mdotPspec()
        
        
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        
        norm = 2*self.N*self.dt
        
        if self.fg_ad is None:
            pass
        else:
            for i in range(len(self.fg_ad)):
                ax.loglog(self.fs, norm*self.fs*self.pow_dsc_iann[:, i], ls='-.', color='red')
            ax.loglog(self.fs, norm*self.fs*self.ACdsc, color='red')
        
        if self.fg_wc is None:
            pass
        else:
            for i in range(len(self.fg_wc)):
                ax.loglog(self.fs, norm*self.fs*self.pow_wrm_iann[:, i], ls='-.', color='green')
            ax.loglog(self.fs, norm*self.fs*self.ACwrm, color='green')
        
        if self.fg_hc is None:
            pass
        else:
            for i in range(len(self.fg_hc)):
                ax.loglog(self.fs, norm*self.fs*self.pow_hot_iann[:, i], ls='-.', color='blue')
            ax.loglog(self.fs, norm*self.fs*self.AChot, color='blue')
        
        
        ax.set_xlabel('Frequency, f   (Hz)')
        #ax.set_ylabel(f'fP(f)   (2T/$\mu^{2}$) $(\sigma/\mu)^{2}$')
        ax.set_ylabel(f'fP(f)   $(\sigma/\mu)^{2}$')
        plt.show()
        
    
    def plot_PowerSpecRealisations(self, Nrel):
        """
        Takes the mdot realisations and generates a power spectrum in order
        to compare to the model.
        This is simply a de-bugging aid and works as a sanity check!!
        
        Returns
        -------
        None.

        """
        if hasattr(self, 'AChot'):
            pass
        else:
            self.gene_mdotPspec()
        
        
        
        #Generating the averaged power-spectrum from the xt-realisations
        print('Averaging all model realisations for power-sepctrum')
        for i in tqdm(range(Nrel)):
            _, _, xt_hot, dttot = self.gene_mdotRealisation(fpd=1, fpw=1, fph=1)
            xt_i = xt_hot[:, -1]
            
            FTi = np.fft.rfft(xt_i, self.N, norm='forward')
            FTi = np.concatenate((np.conj(FTi[1:]), FTi))
            
            if i == 0:
                pow_tot = np.conj(FTi) * FTi
            else:
                pow_tot += np.conj(FTi) * FTi
            
        pow_tot /= Nrel #power-spectrum averaged over all realisations
        
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)
        
        ax.loglog(self.fs, self.fs*pow_tot, color='green')
        ax.loglog(self.fs, self.fs*self.AChot, color='k')
        
        #finding limits
        Tact = (self.N*self.dt) - dttot
        ax.axvline(1/Tact, ls='-.', color='red')
        fmax = 1/(2*self.dt)
        ax.axvline(fmax/2, ls='-.', color='red')
        
        fobs_lim = 1/(2*1*24*3600)
        ax.axvline(fobs_lim, ls='-.', color='blue')
        
        plt.show()
        
    
    