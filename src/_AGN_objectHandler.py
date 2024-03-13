#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 15:21:30 2022

@author: Scott Hagen

This handles all the generic AGN attributes that all codes need, regardless of
whether one is using only reverberation or considering full variability, etc.

Contains all the useful accretion information (like eddington ratios, 
efficiency, r_isco, etc...). Also contains the methods for calculating 
the Novikov-Thorne temperature, black-body spectra, nthcomp spectra, etc

Finally, all the methods for handling units are contained within here too

This code is intended as a base object, that handles all the usefull
things a user may need irregardless of model/physical set-up, but doesn't
necesssarily want to deal with themselves. Essentially, this will be incorporated
into all the other models, hence the user should NEVER have to deal with or
import this particular module
"""

import numpy as np
import astropy.constants as const
import astropy.units as u
import warnings
import pickle

from scipy.interpolate import interp1d
from astropy.io import fits




class AGNobject:
    """
    Main AGN object handler
    
    Default units are ALWAYS cgs and in luminosity
    """
    
    #Defining default energy grid
    Emin = 1e-4 #keV
    Emax = 1e4 #keV
    numE = 1000
    
    default_units = 'cgs'
    units = 'cgs'
    as_flux = False
    
    dr_dex = 100 #default radial spacing - N bins per decade
    A = 0.3 #default disc Albedo
    
    #Stop all the run-time warnings (we know why they happen - doesn't affect the output!)
    warnings.filterwarnings('ignore') 
    verbose = False
    
    def __init__(self,
                 M,
                 dist,
                 log_mdot,
                 astar,
                 redshift):
        """
        Initiates AGN object

        Parameters
        ----------
        M : float
            AGN mass - Units : Msol.
        dist : float
            Co-Moving distance - Units : Mpc.
        log_mdot : float
            log of mass accretion rate - Units : Mdot/Mdot_edd.
        astar : float
            Black-hole spin.
        redshift : float
            Redshift.

        
        """
        
        #Read parameters
        self.M = float(M)
        self.D, self.d = float(dist), float((dist*u.Mpc).to(u.cm).value)
        self.mdot = float(10**(log_mdot))
        self.a = float(astar)
        self.z = float(redshift)
    
        #Initiating relevant constants
        self._set_constants()
        
        #Energy/frequency grid
        self.Egrid = np.geomspace(self.Emin, self.Emax, self.numE)
        self._Egrid_default = np.geomspace(1e-4, 1e4, 1000)
        self.nu_grid = (self.Egrid * u.keV).to(u.Hz,
                                equivalencies=u.spectral()).value
        self._nu_grid_default = (self._Egrid_default * u.keV).to(u.Hz,
                                equivalencies=u.spectral()).value
        self.wave_grid = (self.Egrid * u.keV).to(u.AA,
                                equivalencies=u.spectral()).value
        
        self.E_obs = self.Egrid/(1+self.z)
        self.nu_obs = self.nu_grid/(1+self.z)
        self.wave_obs = self.wave_grid * (1+self.z)
        
        self.dlog_r = 1/self.dr_dex
        
        #Calculating relevant attributes
        self._calc_risco()
        self._calc_r_selfGravity()
        self._calc_Ledd()
        self._calc_efficiency()
        
        #physical conversion factors
        self.Mdot_edd = self.L_edd/(self.eta * self.c**2) #g/s
        self.Rg = (self.G * self.M)/(self.c**2) #cm
        
        
        #Initiating component dictionaries
        self.SEDdict = {} #will contain SED components (time independent!!)
        self.VARdict = {} #Will contain time-dependent SED components
        
        
    
    def _set_constants(self):
        """
        Sets the physical constants in cgs units as object attributes

        """
        self.G = (const.G * const.M_sun).to(u.cm**3 / u.s**2).value #cm^3/Msol/s^2
        self.sigma_sb = const.sigma_sb.to(u.erg/u.s/u.K**4/u.cm**2).value 
        self.c = const.c.to(u.cm/u.s).value #speed of light, cm/s
        self.h = const.h.to(u.erg*u.s).value #Plank constant, erg s
        self.k_B = const.k_B.to(u.erg/u.K).value #Boltzmann constant, #erg/K
        self.m_e = const.m_e.to(u.g).value  #electron mass, g
        
    
    
    ##########################################################################
    #---- Unit handling
    ##########################################################################
    
    def set_units(self, new_unit='cgs'):
        """
        Re-sets default units. ONLY affects attributes extracted through the
        getter methods
        
        Note, the only difference between setting cgs vs counts is in spectra

        Parameters
        ----------
        new_unit : {'cgs','cgs_wave', 'SI', 'counts'}, optional
            The default unit to use. The default is 'cgs'.
            NOTE, the main cgs_wave will give spectra in erg/s/Angstrom,
            while cgs gives in erg/s/Hz

        """
        #Checking valid units
        unit_lst = ['cgs', 'cgs_wave', 'SI', 'counts']
        if new_unit not in unit_lst:
            print('Invalid Unit!!!')
            print(f'Valid options are: {unit_lst}')
            print('Setting as default: cgs')
            new_unit = 'cgs'
            
        self.units = new_unit
    
    def set_flux(self):
        """
        Sets default output as a flux
        This ONLY affects spectra! Things like Eddington luminosity, or
        Bolometric luminosity remain as Luminosity!!
        
        Note: This will also take the redshift into account!!
        
        /cm^2 IF cgs or counts, /m^2 if SI

        """
        self.as_flux = True
        
        
    def set_lum(self):
        """
        Sets defualt output as luminosity (only necessary IF previously set
        as flux)

        """
        self.as_flux = False
    
    
    def _to_newUnit(self, L, as_spec=True):
        """
        Sets input luminosity/spectrum to new output units

        Parameters
        ----------
        L : float OR array
            Input lum/spectrum.

        Returns
        -------
        Lnew : float OR array
            In new units.
        unit : str
            new unit (in case the currently set unit is not desired)
        as_spec : bool
            If True, then input should be erg/s/Hz
            If false, then input should be erg/s

        """
        #If spectral density
        if as_spec:
            if self.units == 'cgs':
                Lnew = L
                
            elif self.units == 'counts':
                Lnew = (L*u.erg/u.s/u.Hz).to(u.keV/u.s/u.keV,
                                             equivalencies=u.spectral()).value
                
                if np.ndim(L) == 1:
                    Lnew /= self.Egrid
                else:
                    Lnew /= self.Egrid[:, np.newaxis]
            
            elif self.units == 'cgs_wave':
                Lnew = (L*u.erg/u.s/u.Hz).to(u.erg/u.s/u.AA,
                                equivalencies=u.spectral_density(self.nu_grid*u.Hz)).value
            
            else:
                Lnew = L*1e-7
        
        #If just a luminosity
        else:
            if self.units == 'cgs' or self.units == 'counts':
                Lnew = L
            else:
                Lnew = L*1e-7
        
        return Lnew
    
    
    def _to_flux(self, L):
        """
        Converts to a flux - takes redshift into accounts    

        Parameters
        ----------
        L : float OR array
            Luminosity to be converted.

        Returns
        -------
        f : float OR array
            Flux seen by observer

        """
        
        if self.units == 'cgs' or self.units == 'counts':
            d = self.d #distance in cm
        else:
            d = self.d/100 #distance in m
        
        f = L/(4*np.pi*d**2 * (1+self.z))
        return f
            
            
        
    
    
    ##########################################################################
    #---- Disc properties
    ##########################################################################
    
    def _calc_Ledd(self):
        """
        Caclulate eddington Luminosity

        """
        Ledd = 1.39e38 * self.M #erg/s
        self.L_edd = Ledd
    
    
    def _calc_risco(self):
        """
        Calculating innermost stable circular orbit for a spinning
        black hole. Follows Page and Thorne (1974). Note, can also be reffered
        to as r_ms, for marginally stable orbit
        
        return r_isco as property - so will be called in __init__

        """
        Z1 = 1 + (1 - self.a**2)**(1/3) * (
            (1 + self.a)**(1/3) + (1 - self.a)**(1/3))
        Z2 = np.sqrt(3 * self.a**2 + Z1**2)

        self.risco = 3 + Z2 - np.sign(self.a) * np.sqrt(
            (3 - Z1) * (3 + Z1 + 2*Z2))
    
    
    def _calc_r_selfGravity(self):
        """
        Calcultes the self gravity radius according to Laor & Netzer 1989
        
        NOTE: Assuming that \alpha=0.1 - in future should figure out how to
        constrain this properly!!!

        """
        alpha = 0.1 #assuming turbulence NOT comparable to sound speed
        #See Laor & Netzer 1989 for more details on constraining this parameter
        m9 = self.M/1e9
        self.r_sg = 2150 * m9**(-2/9) * self.mdot**(4/9) * alpha**(2/9)
    
    
    def _calc_efficiency(self):
        """
        Calculates the accretion efficiency eta, s.t L_bol = eta Mdot c^2
        Using the GR case, where eta = 1 - sqrt(1 - 2/(3 r_isco)) 
            Taken from: The Physcis and Evolution of Active Galactic Nuceli,
            H. Netzer, 2013, p.38
        
        Note to self!: When I derive this in Newtonian limit I get
        eta = 1/(2 r_isco). Not entirely sure how to derive the GR version.
        Should ask Chris at next meeting!!!!

        """
        
        self.eta = 1 - np.sqrt(1 - 2/(3*self.risco))
    
    
    def _calc_NTparams(self, r):
        """
        Calculates the Novikov-Thorne relativistic factors.
        see Active Galactic Nuclei, J. H. Krolik, p.151-154
        and Page & Thorne (1974)

        """
        y = np.sqrt(r)
        y_isc = np.sqrt(self.risco)
        y1 = 2 * np.cos((1/3) * np.arccos(self.a) - (np.pi/3))
        y2 = 2 * np.cos((1/3) * np.arccos(self.a) + (np.pi/3))
        y3 = -2 * np.cos((1/3) * np.arccos(self.a))

        
        B = 1 - (3/r) + ((2 * self.a)/(r**(3/2)))
        
        C1 = 1 - (y_isc/y) - ((3 * self.a)/(2 * y)) * np.log(y/y_isc)
        
        C2 = ((3 * (y1 - self.a)**2)/(y*y1 * (y1 - y2) * (y1 - y3))) * np.log(
            (y - y1)/(y_isc - y1))
        C2 += ((3 * (y2 - self.a)**2)/(y*y2 * (y2 - y1) * (y2 - y3))) * np.log(
            (y - y2)/(y_isc - y2))
        C2 += ((3 * (y3 - self.a)**2)/(y*y3 * (y3 - y1) * (y3 - y2))) * np.log(
            (y - y3)/(y_isc - y3))
        
        C = C1 - C2
        
        return C/B
        
        
    
    
    def calc_Tnt(self, r, fmd=1):
        """
        Calculates Novikov-Thorne disc temperature^4 at radius r. 
        
        Parameters
        ----------
        r : float OR array
            Radius at which to evaluate the temperature - Units : Rg
        fmd : float, optional
            Factor by which to modulate the mass accretion rate
            i.e fmd = mdot(t)/mdot_intrinsic
            If 1, then T4 will be the intrinisic NT temp
        
        Returns
        -------
        T4 : float OR array
            Novikov-Thorne temperature of the system, at radius r
            Note, this is at T^4 - Units : K^4
            
        """
        mdot_f = self.mdot * fmd
        
        Rt = self._calc_NTparams(r)
        const_fac = (3 * self.G * self.M * mdot_f * self.Mdot_edd)/(
            8 * np.pi * self.sigma_sb * (r * self.Rg)**3)
        
        T4 = const_fac * Rt
        return T4
    
    
    def calc_Trep(self, r, Lx):
        """
        Calculates the re-processed temperature at r for a given X-ray
        luminosity

        Parameters
        ----------
        r : float OR array
            Radial coordinate - units : Rg
        Lx : float OR array
            X-ray luminosity seen at point r, phi at time t - units : erg/s
        Returns
        -------
        T4rep : float OR array
            Reprocessed temperature at r, to the power 4

        """
        
        R = r * self.Rg
        H = self.hmax * self.Rg
        
        Frep = (Lx)/(4*np.pi * (R**2 + H**2))
        Frep *= H/np.sqrt(R**2 + H**2)
        Frep *= (1 - self.A) 
        
        T4rep = Frep/self.sigma_sb

        return T4rep * (1 - self.A)
    
    
    
    def _make_rbins(self, logr_in, logr_out):
        """
        Creates an array of radial bin edges, with spacing defined by dr_dex
        Calculates the bin edges from r_out and down to r_in. IF the bin
        between r_in and r_in+dr is less than dlog_r defined by dr_dex, then
        we simply create a slightly wider bin at this point to accomodate
        for the difference

        Parameters
        ----------
        logr_in : float
            Inner radius of model section - units : Rg.
        logr_out : float
            Outer radius of model section - units : Rg.

        Returns
        -------
        logr_bins : 1D-array
            Radial bin edges for section - units : Rg.

        """
        i = logr_out
        logr_bins = np.array([np.float64(logr_out)]) 
        while i > logr_in:
            r_next_edge = i - self.dlog_r
            logr_bins = np.insert(logr_bins, 0, r_next_edge)
            i = r_next_edge

        if np.around(logr_bins[0], 7) != logr_in:
            if logr_bins[0] < logr_in:
                if len(logr_bins) > 1:
                    logr_bins = np.delete(logr_bins, 0)
                    logr_bins[0] = logr_in
                else:
                    logr_bins[0] = logr_in
            else:
                logr_bins[0] = logr_in
        
        return logr_bins
    
    
    
    def _make_rbins_even(self, logr_in, logr_out):
        """
        Creates an array of bin edges, with initial spacing defined by dr_dex.
        Calculates the bin edges from log_rout down to log_rin.
        IF the bin between r_in and r_in+dr is less than dlog_r defined by dr_dex,
        then we re-scale dr_dex to ensure evenly-spaced bins!

        Parameters
        ----------
        logr_in : float
            log inner radius - Units : Rg.
        logr_out : float
            log outer radius - Units : Rg.

        Returns
        -------
        logr_bins : array
            log bin edges

        """
        num_bins = (logr_out - logr_in)*self.dr_dex
        num_bins = int(np.ceil(num_bins))
        
        dr_new = num_bins/(logr_out - logr_in)
        log_rbins = np.linspace(logr_out, logr_in, num_bins+1)
        return log_rbins, dr_new
    
    

    
    
    ###########################################################################
    #---- Handling the SED
    #     i.e keeps track of all SED components, and can generate total SED
    #     Note, SED components differ between models...
    ###########################################################################
    
    def _add_SEDcomponent(self, Lnu, name=None):
        """
        Adds a spectral component to the SED dictionary

        Parameters
        ----------
        Lnu : array
            Spectral component - Must have same shape as Egrid
            Units : erg/s/Hz.
        name : str, optional
            The component name - this will be used to look up and extract the 
            component later on.
            If None, then uses comp<number> where number is the current number
            of components = 1. The default is 'none'.

        """
        
        if name == None:
            num = len(self.SEDdict.keys())
            name = f'comp{num+1}'
        
        self.SEDdict[name] = Lnu
        
    
    def get_SEDcomponent(self, component):
        """
        Extracts spectral component - taking into account unit options
        First checks if the component exists
        If not, then also runs make_SED to calculate the spectral component.
        If the component still does not exit, then it's because it does not 
        feature in the model, in which case this will throw an error
        
        Parameters
        ----------
        component : str
            Which component to extract
        
        Returns
        -------
        Lout : array
            Spectral component in whatever units that have been set.

        """
        
        if component in self.SEDdict.keys():
            pass
        else:
            self.make_SED()
            if component not in self.SEDdict.keys():
                raise ValueError('Component does not feature in this model!')
        
        Lout = self.SEDdict[component]
        Lout = self._to_newUnit(Lout, as_spec=True)
        if self.as_flux == True:
            Lout = self._to_flux(Lout)
        
        return Lout
    
    
    def get_totalSED(self):
        """
        Adds up all SED components and returns the total SED
        
        Checks if any SED components exist first. If not, then runs make_SED

        Returns
        -------
        Lout : array
            Total SED for the model.

        """
        
        if len(self.SEDdict.keys()) == 0:
            self.make_SED()
        
        Ltot = np.zeros(len(self.Egrid))
        for k in self.SEDdict.keys():
            Ltot += self.SEDdict[k]
        
        Lout = self._to_newUnit(Ltot, as_spec=True)
        if self.as_flux == True:
            Lout = self._to_flux(Lout)
        
        return Lout
    
    
    
    ###########################################################################
    #---- Section for handling light-curve/time-dependent output
    ###########################################################################
    
    def _add_varComponent(self, Lvar, name=None):
        """
        Adds varying spectral component to the VARdict attribute

        Parameters
        ----------
        Lvar : array
            Time-dependent spectral component. Calculated when evolve_spec()
            is run.
        name : str, optional
            The component name - this will be used to look up and extract the 
            component later on.
            If None, then uses comp<number> where number is the current number
            of components = 1. The default is 'none'.

        """
        
        if name == None:
            num = len(self.SEDdict.keys())
            name = f'comp{num+1}'
        
        self.VARdict[name] = Lvar
        
        
        
    def get_Lcurve(self, band, band_width, as_frac=True, 
                   band_units='keV', component='all'):
        """
        Extracts a light-curve from the set of time-dependent SEDs. Uses a
        bandpass centererd on 'band' with width 'band_width'. For simplicity,
        assumes the bandpass is a top-hat (i.e extract everything within the
        bandpass, and nothing outside of it!)
        
        NOTE: If 'band_width' is smaller than the model bin width, then it
        will use the model bin width instead.
        
        NOTE2: The units in 'band' and 'band_width' need to be the same!!!

        Parameters
        ----------
        band : float
            Bandpass midpoint - Units : {'keV', 'Hz', 'AA'}.
        band_width : float
            Bandpass width - Units : {'keV', 'Hz', 'AA'}
        as_frac : bool, optional
            If True, then output is normalised by the mean SED, so units F/Fmean
            If False, then output has the currently set units (i.e ergs/s/Hz, etc)
            The default is True.
        band_units : {'keV', 'Hz', 'AA'}, optional
            The band-pass units. Note AA is Angstrom. The default is 'keV'.
        component : {'all', 'intrinsic', 'wind_ref', 'disc', 'warm', 'hot'}, optional
            Specify spectral component to extract. If 'all', then uses total SED. 
            The default is 'all'.

        Returns
        -------
        Lc_out : array
            Model light-curve extracted for given band-pass.

        """
        
        
        Ltot_all, Lmean = self._LC_extractionCHECKS(component)
        
        #Handling units
        #Explicitly doing band limits as dE doesnt necessarily transform nicely
        bnd_max = band + band_width/2 
        bnd_min = band - band_width/2
        
        unit_dict = {'SI':'Hz', 'cgs':'Hz', 'counts':'keV'} #translating unit setting
        bnd_max = (bnd_max * u.Unit(band_units)).to(
            u.Unit(unit_dict[self.units]), equivalencies=u.spectral()).value
        bnd_min = (bnd_min * u.Unit(band_units)).to(
            u.Unit(unit_dict[self.units]), equivalencies=u.spectral()).value
        band = (band * u.Unit(band_units)).to(
            u.Unit(unit_dict[self.units]), equivalencies=u.spectral()).value
        
        if band_units != 'AA':
            band_width = bnd_max - bnd_min
        else:
            band_width = bnd_min - bnd_max
        

        #Now extracting Lcurve
        Lc_out = self._extractLC(Ltot_all, Lmean, band, band_width, as_frac)
        
        return Lc_out
    
    
    
    def get_Lcurve_fromRSP(self, rspfile, arffile=None, as_frac=True, component='all',
                           Emin=None, Emax=None, abs_file=None):
        """
        Generates a light-curve, but using a telescope response file rather
        than assuming a top hat (e.g the swift uvot uvw2 band-pass)
        
        Note, this assumes that the response has energy units in keV, and
        effective area in cm^2.
        Also, only works on 1D response files (e.g Swift-UVOT or XMM-OM files)
        Non-diagonal matrices, as commonly used in Xrays, is not currently
        supported

        Parameters
        ----------
        rspfile : str
            Input response file
        arffile : str
            Input axuillary response file
            Necessary when using X-ray rsp files to get correct efficteve area
            (not necessary for UVOT, as there rsp file gives effective area)
        as_frac : TYPE, optional
            If True, then output is normalised by the mean SED, so units F/Fmean
            If False, then output has the currently set units (i.e ergs/s/Hz, etc)
            The default is True.
        component : {'all', 'intrinsic', 'wind_ref', 'disc', 'warm', 'hot'}, optional
            Specify spectral component to extract. If 'all', then uses total SED. 
            The default is 'all'.
        Emin : float
            Min energy to start integrating the response file over
            If None, uses the minimum energy in the response file
            Units : keV
        Emax : float
            Max energy to integrate the response to
            If None, uses the maximum energy in the response file
            Units : keV
        abs_file : str
            File containing absorption coefficients as a function of energy
            Must be in two column-format, corresponding to (Es, abs_coeffs)
            If included, then the absoprtion model is taken into account when
            calculating the light-curves
            The default is None

        Returns
        -------
        LC_out : array
            Output light-curve

        """
        
        Ltot_all, Lmean = self._LC_extractionCHECKS(component)
        
        print(Lmean[100], Ltot_all[100, 100])
        
        #converting to kev (photons/s/keV)
        Ltot_all = (Ltot_all*u.erg/u.s/u.Hz).to(u.keV/u.s/u.keV,
                                        equivalencies=u.spectral()).value
        Lmean = (Lmean*u.erg/u.s/u.Hz).to(u.keV/u.s/u.keV,
                                        equivalencies=u.spectral()).value
        
        #now kev (photon/cm^2/keV)
        Ltot_all /= (4*np.pi*self.d**2 * (1+self.z))
        Lmean /= (4*np.pi*self.d**2 * (1+self.z))
        
        #now photons/cm^2/keV
        Ltot_all /= self.Egrid[:, np.newaxis]
        Lmean /= self.Egrid
        
        #importing instrument response (effective area)
        rEm, rsp = self._load_rsp(rspfile, arffile)
        
        #getting integration range
        if Emin == None:
            eidx_l = 0
        else:
            eidx_l = np.abs(rEm - Emin).argmin()
        
        if Emax == None:
            eidx_r = None
        else:
            eidx_r = np.abs(rEm - Emax).argmin()
            eidx_r += 1 #becuase python...
        
        #loading abs-model if included
        rEm = rEm[eidx_l:eidx_r]
        if abs_file != None:
            E_abs, abs_coeff = np.loadtxt(abs_file, unpack=True)
            abs_int = interp1d(E_abs, abs_coeff, kind='linear')
            rsp = rsp[eidx_l:eidx_r] * abs_int(rEm)
        else:
            rsp = rsp[eidx_l:eidx_r]
            
        
        #for each time-step, interpolating SED and then multiplying with eff area
        LC_out = np.array([])
        for i in range(len(Ltot_all[0, :])):
            Lt_interp = interp1d(self.E_obs, Ltot_all[:, i], kind='linear')
            
            LC_t = np.trapz(Lt_interp(rEm)*rsp, rEm) #Multiplying with response
            LC_out = np.append(LC_out, LC_t)
        
        if as_frac == True:
            Lm_interp = interp1d(self.E_obs, Lmean, kind='linear')
            Lm_filt = np.trapz(Lm_interp(rEm)*rsp, rEm)
            
            LC_out /= Lm_filt
        
        return LC_out
    
    
    
    def _load_rsp(self, rsp_file, arf_file):
        """
        Loads the response (effective area) of the detector given an input
        rsp and arf file
        Assumes rsp and arf files follow standard format (as used in Xspec)

        Parameters
        ----------
        rsp_file : str
            File containing response matrix
        arf_file : str
            File containing auxillary response 

        Returns
        -------
        Em : array
            Midpoint energies within response curve
            Units : keV
        rsp : array
            Detector effective area
            Units : cm^2

        """
        
        
        #loading rsp file first
        with fits.open(rsp_file, mode='readonly') as hdul:
            dat = hdul['MATRIX'].data
            
            El = np.array(dat.field(0)) #left energy bin edges
            Er = np.array(dat.field(1)) #right bin edges
            matrix = np.array(dat.field(5)) #response matrix
            
        #to get response matrix we sum over channels in response matrix
        rsp = np.empty(len(El))
        for i, mat in enumerate(matrix):
            rsp[i] = np.sum(mat)
            
        
        #now multiplying by arf (if given) to get actual effective area
        if arf_file != None:
            with fits.open(arf_file, mode='readonly') as hdul:
                adat = hdul['SPECRESP'].data
                arf_rsp = np.array(adat.field(2))
            
            rsp *= arf_rsp
            
        
        #getting Ebin midtpoints
        Em = El + 0.5*(Er - El)
        return Em, rsp
        
    
    
    def _extractLC(self, Ltot_all, Lmean, band, band_width, as_frac):
        """
        Extracts a light-curve from time-dep SEDs
        This gets called by get_Lcurve(), which handles units and component
        choice. So this routine only cares about actually extracting the LC,
        and assumes all input units are correct!!

        Parameters
        ----------
        Ltot_all : 2D-array
            Time-dep SED for extraction
        Lmean : 1D-array
            Mean SED, for normalisation
        band : float
            Midtpoint energy to extract
            Units : same as current object settings
        band_width : float
            Width of bandpass
            Units : same as current object setting
        as_frac : bool
            Whether to normalise by the mean

        Returns
        -------
        Lc_out : 1D-array
            Lightcurve

        """
        
        if self.units == 'SI' or self.units == 'cgs':
            idx_mod_up = np.abs(band + band_width/2 - self.nu_obs).argmin()
            idx_mod_low = np.abs(band - band_width/2 - self.nu_obs).argmin()
            
            if idx_mod_up == idx_mod_low:
                Lcurve = Ltot_all[idx_mod_up, :] * band_width
                Lb_mean = Lmean[idx_mod_up] * band_width
                
            else:
                Lc_band = Ltot_all[idx_mod_low:idx_mod_up+1, :]
                Lcurve = np.trapz(Lc_band, self.nu_grid[idx_mod_low:idx_mod_up+1], axis=0)
                
                Lmean_band = Lmean[idx_mod_low:idx_mod_up+1]
                Lb_mean = np.trapz(Lmean_band, self.nu_grid[idx_mod_low:idx_mod_up+1])
            
        elif self.units == 'counts':
            idx_mod_up = np.abs(band + band_width/2 - self.E_obs).argmin()
            idx_mod_low = np.abs(band - band_width/2 - self.E_obs).argmin()
            
            if idx_mod_up == idx_mod_low:
                Lcurve = Ltot_all[idx_mod_up, :] * band_width
                Lb_mean = Lmean[idx_mod_up] * band_width
            
            else:
                Lc_band = Ltot_all[idx_mod_low:idx_mod_up+1, :]
                Lcurve = np.trapz(Lc_band, self.E_obs[idx_mod_low:idx_mod_up+1], axis=0)
                
                Lmean_band = Lmean[idx_mod_low:idx_mod_up+1]
                Lb_mean = np.trapz(Lmean_band, self.E_obs[idx_mod_low:idx_mod_up+1])
        
        if as_frac == True:
            Lc_out = Lcurve/Lb_mean
        else:
            Lc_out = Lcurve
        
        return Lc_out
    
    
    def _LC_extractionCHECKS(self, component):
        """
        Checks if all necessary components exist for extracting LCs

        Parameters
        ----------
        component : {'all', 'intrinsic', 'wind_ref', 'disc', 'warm', 'hot'}, optional
            Specify spectral component to extract. If 'all', then uses total SED. 
            The default is 'all'.

        Returns
        -------
        None.

        """
        
        #Checking if evolve spec has already been run
        if hasattr(self, 'Lintrinsic_var'):
            pass
        else:
            raise ValueError('NONE type light-curve not permitted!! \n'
                             'Either run evolve_spec() FIRST \n'
                             'OR pass a light-curve here!')

        
        
        #MChecking if mean spec exists
        if hasattr(self, 'Lnu_intrinsic'):
            if self._reverb == self._SED_rep:
                pass
            else:
                self.make_intrinsicSED(reprocess=self._reverb)
        else:
            self.make_intrinsicSED(reprocess=self._reverb)
        
        
        #Extracting desired component
        if component == 'all':
            Ltot_all = self.Lintrinsic_var
            Lmean = self.Lnu_intrinsic
            if hasattr(self, 'Lref_var'):
                Ltot_all = Ltot_all + self.Lref_var
                Lmean = Lmean + self.make_windSED()
            
            if hasattr(self, 'Lline_var'):
                Ltot_all = Ltot_all + self.Lline_var
                Lmean = Lmean + self.make_lineSED()
            
        elif component == 'intrinsic':
            Ltot_all = self.Lintrinsic_var
            Lmean = self.Lnu_intrinsic
            
        else:
            Ltot_all = self.VARdict[component]
            Lmean = self.SEDdict[component]
        
        return Ltot_all, Lmean
    
    
    
    def get_timeDepSED(self, component='all', fuv=np.array([1])):
        """
        Extracts the fully time-dependent SED in whatever units are currently set
        
        Parameters
        ----------
        component : str, optional
            What spectral component to extract. If 'all', then uses total SED
            Defualt is 'all'
        fuv : array
            Light-curve to be passed (in case evolve_spec() hasen't been called
            yet)
            Default is len 1 val 1 array
    
        Returns
        -------
        Lout : array
            Fully time-dependent SEDs
            Shape = (len(Egrid), len(fuv))

        """
        

        if hasattr(self, 'Ltot_var'):
            pass
        else:
            self.evolve_spec(fuv)
        
        if component == 'all':
            Lall = self.Ltot_var
        else:
            Lall = self.VARdict[component]
                    
        Lout = self._to_newUnit(Lall, as_spec=True)
        if self.as_flux == True:
            Lout = self._to_flux(Lout)
        
        return Lout
    
    
    def get_accretionTimeSeries(self, par):
        """
        Extracts the time-series of a given spectral parameter/accretion 
        propperty, generated when evolving the SED

        Parameters
        ----------
        par : str
            Parameter/property to extract
            Must be contained within parvar_dict.keys()
            If unsure what properties are generated for a given model, run 
            self.parvar_dict.keys() first!

        Returns
        -------
        par_timeSereis : array
            Time series of given parameter/array

        """
        
        if hasattr(self, 'parvar_dict'):
            pass
        else:
            raise AttributeError('parvar_dict does not exist!'
                                 'This is probably becuase you have not run'
                                 'evolve_spec() first. Run this, and try again!')
        
        if par not in self.parvar_dict.keys():
            raise ValueError(f'{par} not calculated for given model!'
                             f'For this model par must be {self.parvar_dict.keys()}')
        
        return self.parvar_dict[par]
    
    
    ###########################################################################
    #---- Other useful methods
    ########################################################################### 
    
    def save(self, fname):
        """
        Saves the class instance to an ASCII file using pickle

        Parameters
        ----------
        fname : str
            Filename to save to
            automatically gives file ending as .agn

        Returns
        -------
        None.

        """
        
        agn_file = open(f'{fname}.agn', 'wb')
        pickle.dump(self, agn_file)
        agn_file.close()
    
    
    def set_verbose(self):
        self.verbose = True
    
    def verboseprint(self, *args):
        if self.verbose:
            for arg in args:
                print(arg)
        
        else:
            return None
    
    
        
        
if __name__ == '__main__':
    M =1e8
    dist = 200
    log_mdot = -1
    astar = 0.
    redshift = 0
    
    agn = AGNobject(M, dist, log_mdot, astar, redshift)
    #print(agn.tau_freefall(np.array([20, 15, 10]), np.array([15, 10, 6])))
    print(agn.tau_freefall(10, 2))