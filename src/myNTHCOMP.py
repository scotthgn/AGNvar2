#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 17:46:27 2023

@author: Scott Hagen

Adapting the NTHCOMP subroutine (Zdziarski, Johnson & Magdziarz 1996; 
Zycki, Done & Smith 1999) from fortran to python

Extending the functionality such that if one or more of the input parameters
is an array, then it will return multiple spectra corresponding to each
parameter value
"""

import numpy as np



def mdonthcomp(ear, gamma, kTe, kTbb):
    """
    Adapted from the subroutine donthcomp.f, distributed with XSPEC.
    (Zdziarski, Johnson & Magdziarz 1996; Zycki, Done & Smith 1999)
    
    Note that this has been adapted such that the seed photon spectrum is
    always black-body, AND such that gamma, kTe, or kTbb can be arrays. If
    one of the parameters is an array input, then the code will return a
    spectrum corresponding to each parameter value (in a 2D array - 
    shape=(len(ear), len(par_ar)))
    
    If multiple parameters are passed as array (e.g both gamma and kTe), then
    these need to be the same size, and the code will assume that each index
    corresponds to a paramter pair (or tripplet)

    Parameters
    ----------
    ear : array
        Input energy grid - Units : keV.
    gamma : array
        Photon index - Units : dimensionless
    kTe : array
        Plasma electron temperature - Units : keV
    kTbb : array
        black-body temperature

    Returns
    -------
    photar : array
        Output spectrum - shape = (len(ear), len(par_i)) where par_i is one
        of the input paramters

    """

    #calc photarr
    xth, nth, spt = _thcompt(kTbb/511.0, kTe/511.0, gamma)
    
    #Calculating normalisation factor
    xx = 1/511.0
    ih = np.argwhere(xx > xth)
    ih = ih[ih < nth]
    ih = ih[-1] + 1
    
    spp = spt[ih-1] + (spt[ih] - spt[ih-1]) * (
        xx - xth[ih-1])/(xth[ih] - xth[ih-1])
    normfac = 1.0/spp
    
    #re-casting onto original energy grid
    photar = np.zeros((len(ear), len(gamma)))
    prim = np.zeros((len(ear), len(gamma)))
    
    j = 0
    for i in range(len(ear)):
        while j < nth and 511 * xth[j] < ear[i]:
            j = j+1
        
        if j <= nth:
            if j > 0:
                jl = j - 1
                prim[i] = spt[jl] + ((ear[i]/511 - xth[jl]) * (
                    spt[jl+1] - spt[jl])/(xth[jl+1] - xth[jl]))
            
            else:
                prim[i] = spt[0]
    
    ne = len(ear)
    photar[1:ne] = 0.5 * (prim[1:ne]/ear[1:ne, np.newaxis]**2 + prim[0:ne-1]/ear[0:ne-1, np.newaxis]**2)
    photar[1:] *= (ear[1:ne, np.newaxis] - ear[0:ne-1, np.newaxis]) * normfac
    
    return photar

def _thcompt(tempbb, theta, gamma):
    """
    Adapted from the subroutine thCompton in donthcomp.f, included in Xspec
    (Zdziarski, Johnson & Magdziarz 1996; Zycki, Done & Smith 1999)
    
    Throughout we use the same naming convention as in donthcomp.f
    We will aslo try to stick to a similar code structure as in the original,
    to make comparisons easily. The main difference is that this version is
    written to mostly using array operations rather than for loops, hence 
    optimising for python
    
    The comments describing the physica are also taken directly from donthcomp.f
    
    Parameters
    ----------
    tempbb : array
        Black-body temperature, normalised by the electron rest energy (511 keV)
    theta : array
        Plasam electron temperature, normalised by the electron rest energy
    gamma : array
        Photon index

    Returns
    -------
    x : array
        Energy grid used for calculations
    jmax : int
        Max index
    sptot : array
        Output spectrum

    """
    
    #Thompson optical depth
    tautom = np.sqrt(2.25 + 3/(theta * ((gamma + 0.5)**2 - 2.25))) - 1.5
    
    #initialising arrays
    #in donthcomp.f these have length 900 - hence here they have shape (900, len(par_i))
    dphesc = np.zeros((900, len(gamma)))
    dphdot = np.zeros((900, len(gamma)))
    rel = np.zeros(900) #only cares about energy array, which is same throughout
    bet = np.zeros((900, len(gamma)))
    c2 = np.zeros(900)
    sptot = np.zeros((900, len(gamma)))
    
    #jmax = num photon energies
    #Using same energy grid for each parameter entry to avoid unecessary looping
    delta = 0.02
    deltal = delta * np.log(10.0)
    xmin = 1e-4 * min(tempbb)
    xmax = 40.0 * max(theta)
    jmax = min(899, int(np.log10(xmax/xmin)/delta+1))
    
    #Energy array
    x = np.zeros(900)
    x[:jmax+1] = xmin * 10**(np.arange(jmax+1)*delta) #only populating up to jmax!!
    
    
    #Calculating c2 and rel arrays
    #As in donthcomp.f, c2 is the relativistic correction to Kompaneets eqn
    #and rel is the Klein-Nishina cross-section divided by the Thompson cross-section
    w = np.zeros(900)
    w[:jmax] = x[:jmax]
    w1 = np.sqrt(x[:jmax] * x[1:jmax+1]) #here w1 is x[j+1/2] up to jmax
  
    c2[:jmax] = (w1**4)/(1+4.6*w1 + 1.1*w1*w1)
    
    
    #for x < 0.05 use asymptotic limit for rel
    #Here simply calculating up to jmax, then overwriting later
    #This is to ensure values beyond jmax are still 0!!!
    rel[:jmax] = 1 - 2*w[:jmax] + 26*w[:jmax]*w[:jmax]/5
    
    #intiating    
    z1 = np.zeros(900)
    z2 = np.zeros(900)
    z3 = np.zeros(900)
    z4 = np.zeros(900)
    z5 = np.zeros(900)
    z6 = np.zeros(900)
    #only filling up to jmax - rest remain at 0!!!
    z1[:jmax] = (1+w[:jmax])/w[:jmax]**3
    z2[:jmax] = 1+2*w[:jmax]
    z3[:jmax] = np.log(z2[:jmax])
    z4[:jmax] = 2*w[:jmax]*(1+w[:jmax])/z2[:jmax]
    z5[:jmax] = z3[:jmax]/2/w[:jmax]
    z6[:jmax] = (1+3*w[:jmax])/z2[:jmax]/z2[:jmax]
    
    #Overwriting for values x >= 0.05
    rel[w >= 0.05] = 0.75*(z1[w>=0.05]*(z4[w>=0.05]-z3[w>=0.05])+z5[w>=0.05]-z6[w>=0.05])
    

    
    #Now calculating thermal emission spectra (i.e seed spectrum)
    #Does one per parameter entry - so outputs in 2D array with shape (len(x), len(par_i))
    jmaxth = min(900, int(np.log10(50*max(tempbb)/xmin)/delta))
    if jmaxth > jmax:
        jmaxth = jmax
    
    planck = 15/(np.pi*tempbb)**4
    dphdot[:jmaxth] = planck * x[:jmaxth, np.newaxis]**2
    dphdot[:jmaxth] /= (np.exp(x[:jmaxth, np.newaxis]/tempbb) - 1) #BB spec
    
    
    
    
    
    #Calculating beta array - probability of escape per Thompson time
    #This is evaluated for a spherical geometry and nearly uniform sources
    #Between x=0.1 and 1.0, a funciton flz modifies beta to allow the increasingly
    #large energy change per scattering to gradually eliminate spatial diffusion
    jnr = min(int(np.log10(0.1/xmin)/delta+1), jmax-1)
    jrel = min(int(np.log10(1/xmin)/delta+1), jmax)
    xnr = x[jnr-1]
    xr = x[jrel-1]
    
    taukn = tautom * rel[:, np.newaxis]
    #x < 0.1
    bet[:jnr-1] = 1/tautom/(1+taukn[:jnr-1]/3)
    
    #0.1 < x < 1
    flz = 1 - ((x - xnr)/(xr - xnr))
    bet[jnr-1:jrel] = 1/tautom/(1+taukn[jnr-1:jrel]/3*flz[jnr-1:jrel, np.newaxis])
    
    
    #x > 1
    bet[jrel:jmax] = 1/tautom
    
    dphesc = _thermlc(tautom, theta, deltal, x, jmax, dphesc, dphdot, bet, c2)
    
    sptot_tst = np.zeros((900, len(gamma)))
    for j in range(0, jmax-1):
        sptot_tst[j] = dphesc[j] * x[j]**2
    
    #spectrum in E F_E
    sptot[:jmax-1] = dphesc[:jmax-1] * x[:jmax-1, np.newaxis]**2

    return x, jmax, sptot_tst
    

def _thermlc(tautom, theta, deltal, x, jmax, dphesc, dphdot, bet, c2):
    """
    Adapted from the subroutine thermlc in donthcomp.f, included in Xspec
    (Zdziarski, Johnson & Magdziarz 1996; Zycki, Done & Smith 1999)

    Parameters
    ----------
    tuatom : array, shape(len(par_i))
        Thompson scattering cross-sections for each parameter combination.
    theta : array, shape(len(par_i))
        Plasma electron temperature, normalised to electron rest energy (511 keV).
    deltal : float
        10-log interval of photona array.
    x : array, shape(900,)
        Energy array, normalised to electron rest energy.
    jmax : int
        Max index in energy array that is considered.
    dphesc : array, shape(len(x), len(par_i))
        Escaping photon density.
    dphdot : array, shape(len(x), len(par_i))
        Photon production rate (i.e seed spectrum).
    bet : array, shape(len(x), len(par_i))
        Probability of escape per Thompson time.
    c2 : array, shape(len(x))
        Coefficients in Kompaneets equation.

    Returns
    -------
    dphesc : array
        Escaping photon density

    """

    c20 = tautom/deltal
    
    #Determining u, where u(x) is dimensionless photon occupation number
    #Defining the coefficients that go into the equation:
    #a(j)*u(j+1)+b(j)*u(j)+c(j)*u(j-1) = d(j)
    
    w1 = np.sqrt(x[1:jmax-1]*x[2:jmax]) #this is x(j+1/2)
    w2 = np.sqrt(x[0:jmax-2]*x[1:jmax-1]) #this is x(j-1/2)
    
    a = np.zeros((900, len(theta)))
    b = np.zeros((900, len(theta)))
    c = np.zeros((900, len(theta)))
    d = np.zeros((900, len(theta)))
    a[1:jmax-1] = -c20 * c2[1:jmax-1, np.newaxis] * (theta/deltal/w1[:, np.newaxis]+0.5)
    
    t1 = -c20 * c2[1:jmax-1, np.newaxis] * (0.5-theta/deltal/w1[:, np.newaxis])
    t2 = c20 * c2[0:jmax-2, np.newaxis] * (theta/deltal/w2[:, np.newaxis]+0.5)
    t3 = x[1:jmax-1, np.newaxis]**3 * (tautom*bet[1:jmax-1])
    b[1:jmax-1] = t1 + t2 + t3
    c[1:jmax-1] = c20 * c2[0:jmax-2, np.newaxis] * (0.5-theta/deltal/w2[:, np.newaxis])
    d[1:jmax-1] = x[1:jmax-1, np.newaxis] * dphdot[1:jmax-1]
    
    #Defining constants that go into boundary terms
    #u(0) = aa*u(1) (zero flux at lowest energy)
    #u(j x 2) given from region 2 above
    x32 = np.sqrt(x[0] * x[1])
    aa = (theta/deltal/x32 + 0.5)/(theta/deltal/x32 - 0.5)

    u = np.zeros((900, len(theta)))
    u[jmax-1] = 0.0
    
    #Invert tridiagonal matrix
    alp = np.zeros((900, len(theta)))
    gam = np.zeros((900, len(theta)))
    g = np.zeros((900, len(theta)))
    alp[1] = b[1] + c[1]*aa
    gam[1] = a[1]/alp[1]
    g[1] = d[1]/alp[1]
    
    
    #here need for loop as next value always depends on previous...
    for j in range(2, jmax-1): 
        alp[j] = b[j] - c[j]*gam[j-1]
        gam[j] = a[j]/alp[j]
        
        #avoiding the 2nd for loop in donthcomp.f ...
        if j != jmax-2:
            g[j] = (d[j] - c[j]*g[j-1])/alp[j]
    
    
    g[jmax-2] = (d[jmax-2]-a[jmax-2]*u[jmax]-c[jmax-2]*g[jmax-3])/alp[jmax-2]
    u[jmax-2] = g[jmax-2]
    for j in range(2, jmax):
        jj = jmax-j
        u[jj] = g[jj]-gam[jj]*u[jj+1]

    u[0] = aa*u[1]
    
    dphesc = x[:, np.newaxis]*x[:, np.newaxis]*u*bet*tautom
    dphesc[dphesc < 0] = 0   
    return dphesc
    
    
    
    
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time
    import sys
    sys.path.append('/Users/astro-guest/Documents/phd/model_codes/AGN_uvVar_dev/src')
    
    from pyNTHCOMP import donthcomp
    
    ear = np.geomspace(1e-2, 1e3, 1000)
    
    gammas = np.linspace(1.5, 2.5, 1000)
    kTes = np.full(len(gammas), 100)
    kTbbs = np.full(len(gammas), 0.1)
    
    tms = time.time()
    phm_all = mdonthcomp(ear, gammas, kTes, kTbbs)
    tmf = time.time()
    print('Mine done')
    
    tos = time.time()
    for i in range(len(gammas)):
        pho_i = donthcomp(ear, (gammas[i], kTes[i], kTbbs[i], 0, 0, 0))
        
        if i == 0:
            pho_all = pho_i
        else:
            pho_all = np.column_stack((pho_all, pho_i))
    
    tof = time.time()
    print('Old done')
    
    for j in range(len(gammas)):
        plt.loglog(ear, ear*phm_all[:, j])
        plt.loglog(ear, ear*pho_all[:, j], color='k', ls='dashed')
    
    plt.ylim(1e-4, 3e-1)
    plt.show()
    
    
    
    print(f'Runtime mine: {tmf - tms} s')
    print(f'Runtime old: {tof - tos} s')
    

