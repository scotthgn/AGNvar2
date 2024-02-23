#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:52:18 2023

@author: Scott Hagen

Object to deal with all things CLOUDY (for those models that require this).
Contains methods for generating input files, running simulations (in
parallal if desired), reading output files, and extracting relevant information
from output files
 
"""

import numpy as np
from scipy.interpolate import interp1d
import os

class CLOUDY_object:
    
    _cloudy_path = os.getenv('CLOUDY_CODE_PATH') #path to cloudy source code
    
    ###########################################################################
    #---- Section for generating CLOUDY input files
    ###########################################################################
    
    def gene_runfile(self):
        """
        Generates the run file used for executing Cloudy

        Returns
        -------
        None.

        """
        
        with open('run', 'w') as f:
            if self._cloudy_path is None:
                f.write('cloudy.exe -r $1') #useful when running on COSMA!!!
            else:
                f.write(f'{self._cloudy_path}/cloudy.exe -r $1')
        
        os.system('chmod +x run') #make executable
    
    
    def geneCL_infile(self, log_hden, log_Nh, fcov, log_r, Ltot, sedname,
                       fname=None, iterate=False):
        """
        Generates a CLOUDY .in file necessary for running a simulation
        
        Throughout this assumes a constant density cloud, in the open geometry
        format
        Output file name will be CL_run_<sedname>.in

        Parameters
        ----------
        log_hden : float
            log Hydrogen number density
            Units : cm^-3
        log_Nh : float
            log Hydrogen column density of outflow (sets where to stop
            CLOUDY run)
        fcov : float
            Geometric covering fraction
            Units : \Omega/4pi (i.e must be 0<fcov<1)
        log_r : float
            log radius of outflow
            Units : cm
        Ltot : float
            Total luminsotiy (ie. integrated over entire SED)
            Units : erg/s
        sedname : str
            Filename for relevant .sed file.
        fname : str
            Filename for output .in file
        iterate : Bool or int
            Whether to iterate the Cloudy calculation
            If False : No iterations
            If True : iterates until convergence
            If int : Iterates for given number of times

        Returns
        -------
        None.

        """
        
        if sedname.__contains__('.'):
            sedname, _ = sedname.split('.')
        else:
            pass
        
        if fname is None:
            fname = f'CL_run_{sedname}'
        else:
            if fname.__contains__('.'):
                fname, _ = fname.split('.')
            else:
                pass
        
        with open(f'{fname}.in', 'w') as f:
            f.write(f'table SED "{sedname}.sed" \n')
            f.write(f'luminosity total {np.log10(Ltot)} \n')
            f.write(f'radius {log_r} \n')
            f.write(f'hden {log_hden} \n')
            f.write(f'covering fraction {fcov} \n')
            f.write(f'stop column density {log_Nh} \n')
            f.write('constant density \n')
            
            
            if iterate==True:
                f.write('iterate to convergence \n')
            elif iterate == False:
                pass
            else:
                if isinstance(iterate, int):
                    f.write(f'iterate {iterate}\n')
                else:
                    raise ValueError('iterate must be bool or int!')
                    
            f.write('print last iteration \n')
            f.write(f'set save prefix "{fname}" \n')
            f.write('save overview ".ovr" \n')
            f.write('save continuum ".con" units Hz last \n')
            f.write('save physical conditions ".phc" last \n')
        
        return
    
    
    def geneCL_SEDfile(self, nus_in, Lnu_in, sedname):
        """
        Generates a CLOUDY readable SED file for use in calculations
        
        The SED is always saved in nu*Lnu

        Parameters
        ----------
        nus : array
            Frequency grid
            Units : Hz
        Lnu : array
            Input SED
            Units : erg/s/Hz   (i.e the defualt used throughout in the code)
        sedname : str
            Output sed filename
        
        Returns
        -------
        None.

        """
        
        if sedname.__contains__('.'):
            sedname, dscrd = sedname.split('.')
        else:
            pass
        
        idx0 = np.argwhere(Lnu_in <= 0) #Cloudy doesn't like 0's....
        Lnu = np.delete(Lnu_in, idx0)
        nus = np.delete(nus_in, idx0)
        
        np.savetxt(f'{sedname}.sed', np.column_stack((nus, nus*Lnu)))
        
        #Extracting and uppdating file line to tell CLOUDY input units
        with open(f'{sedname}.sed', 'r') as f:
            linelst = f.readlines()
            linelst[1] = linelst[1].replace('\n', '')
            linelst[1] = linelst[1] + ' nuFnu units Hz \n'
        
        #Writing updated line with units
        #Also writing **** to tell file end
        with open(f'{sedname}.sed', 'w') as f:
            f.writelines(linelst)
            f.write('**************')
        
        return
    
    
    
    ###########################################################################
    #---- Section for loading CLOUDY output files
    ###########################################################################
    
    
    
    def loadCL_allSEDs(self, cldir, cl_prefix=None, attrname=None):
        """
        Loads and extracts ALL CLOUDY SEDs. These are then saved as a class attribute
        (which you can manipulate at your whim)

        Parameters
        ----------
        cldir : str
             Path to directory containing CLOUDY files.
        cl_prefix : str, optional
            Cloudy file prefix (i.e <cl_prefix>.con)
            If this is set, then the code will ONLY load the file corresponding
            to the correct prefix.
            If this is None, then will load ALL .con files within directory
            The default is None.
        attrname : str, optional
            Attribute name for use when extractin later on. 
            This follows the convention:
                Ldiff_<attrname>, Lref_<attrname>, etc
            If None, then uses cl_prefix, and if that is None then defualts to
            Ldiff, Lref, Ltran
            (i.e within the class would be self.Ldiff, self.Lref, self.Ltran)
            The default is None.

        Returns
        -------
        None.

        """
        
        #Checking cldir exists and contains cloudy output SEDs
        try:
            cl_lst = os.listdir(cldir)
        except:
            raise FileNotFoundError(f'{cldir} not found')
        
        
        if cl_prefix is None:
            cl_prefix = ''
        else:
            pass
        
        file_lst = [] #so user can keep track of what files were loaded
        i = 0 #counter, as not all files in cl_lst are relevant!
        for cl in cl_lst:
            if cl.__contains__('.con') and cl.__contains__(cl_prefix):
                file_lst.append(cl)               
                #col idxs: nus, tran, diff, ref, ref_lines, diff_lines
                cl_dati = np.loadtxt(f'{cldir}/{cl}', usecols=(0, 2, 3, 5, 7, 8)) 
                
                #If more than one file creates 2D array containing all spectra
                if i == 0:
                    nus = cl_dati[:, 0]
                    Ltran = cl_dati[:, 1]
                    Ldiff = cl_dati[:, 2]
                    Lref = cl_dati[:, 3]
                    ref_lines = cl_dati[:, 4]
                    diff_lines = cl_dati[:, 5]
                
                else:
                    Ltran = np.column_stack((Ltran, cl_dati[:, 1]))
                    Ldiff = np.column_stack((Ldiff, cl_dati[:, 2]))
                    Lref = np.column_stack((Lref, cl_dati[:, 3]))
                    ref_lines = np.column_stack((ref_lines, cl_dati[:, 4]))
                    diff_lines = np.column_stack((diff_lines, cl_dati[:, 5]))
                
                i += 1
                
            else:
                pass
        
        #Saving as class atribute
        if attrname is None:
            if cl_prefix is None:
                attrname = ''
            else:
                attrname = f'_{cl_prefix}'
        
        else:
            attrname = f'_{attrname}'
        
        #Setting loaded SEDs as class attributes
        self.cl_nus = nus #Cloudy always uses same frequency grid
        setattr(self, f'Ltran{attrname}', Ltran)
        setattr(self, f'Ldiff{attrname}', Ldiff)
        setattr(self, f'Lref{attrname}', Lref)
        setattr(self, f'ref_lines{attrname}', ref_lines)
        setattr(self, f'diff_lines{attrname}', diff_lines)    
        setattr(self, f'file_list{attrname}', file_lst)
        setattr(self, f'_line_sub{attrname}', False) #keeping track if data are line-subtracted
    
    
    
    ###########################################################################
    #---- Section for manipulating CLOUDY files
    ###########################################################################
    
    def subtract_lines(self, attrname=None):
        """
        Removes emission lines from the diffuse and reflected spectra
        This is only applied to the spectra saved as the attributes:
            self.Ldiff_<attrname>
            self.Lref_<attrname>
        
        These spectra are then overwritten with the line-subtracted versions

        Parameters
        ----------
        attrname : str, optional
            Attribute name of spectral set to be subtracted 
            This follows the convention:
                Ldiff_<attrname>, Lref_<attrname>, etc
            If None then defualts to Ldiff, Lref, Ltran
            (i.e within the class would be self.Ldiff, self.Lref, self.Ltran)
            The default is None.

        Returns
        -------
        None.

        """
        
        if attrname is None:
            attrname = ''
        else:
            attrname = f'_{attrname}'
        
        
        #If SED attributes exist, then extract
        try:
            Ldiff = getattr(self, f'Ldiff{attrname}')
            Lref = getattr(self, f'Lref{attrname}')
            dlines = getattr(self, f'diff_lines{attrname}')
            rlines = getattr(self, f'ref_lines{attrname}')
        except:
            raise AttributeError(f'No loaded SEDs with attname={attrname}')
        
        #Removing line emission
        Ldiff -= dlines
        Lref -= rlines
        
        #Overwriting with cleaned SEDs
        setattr(self, f'Ldiff{attrname}', Ldiff)
        setattr(self, f'Lref{attrname}', Lref)
        setattr(self, f'_line_sub{attrname}', True)
    
    
    def rebin(self, nu_grid, attrname=None):
        """
        Re-bins Cloudy spectra onto a new energy gridding through simple
        linear interpolation
        
        Note, if the SEDs have not been line-subtracted first, then this will
        be done automatically, to avoid issues in the interpolation!

        Parameters
        ----------
        nu_bins : array
            Frequecny bin edges
            Units : Hz.
        attrname : str, optional
            Attribute name of spectral set to be subtracted 
            This follows the convention:
                Ldiff_<attrname>, Lref_<attrname>, etc
            If None then defualts to Ldiff, Lref, Ltran
            (i.e within the class would be self.Ldiff, self.Lref, self.Ltran)
            The default is None.

        Returns
        -------
        None.

        """
        
        if attrname is None:
            attrname = ''
        else:
            attrname = f'_{attrname}'
            
        #Checking if SEDs exist, and are line subtracted
        try:
            lsub = getattr(self, f'_line_sub{attrname}')
        except:
            raise AttributeError(f'No loaded SEDs with attname={attrname}')
            
        if lsub == False:
            self.subtract_lines(attrname)
        else:
            pass
        
        
        #Now extracting SEDs
        Ldiff = getattr(self, f'Ldiff{attrname}')
        Lref = getattr(self, f'Lref{attrname}')
        Ltran = getattr(self, f'Ltran{attrname}')
        
        Ldiff_int = interp1d(self.cl_nus, Ldiff, axis=0)
        Lref_int = interp1d(self.cl_nus, Lref, axis=0)
        Ltran_int = interp1d(self.cl_nus, Ltran, axis=0)
        
        setattr(self, f'Ldiff{attrname}', Ldiff_int(nu_grid))
        setattr(self, f'Lref{attrname}', Lref_int(nu_grid))
        setattr(self, f'Ltran{attrname}', Ltran_int(nu_grid))
        
        
        
            



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    cl_ob = CLOUDY_object()
    
    cdir = '/Users/astro-guest/Documents/phd/model_codes/AGN_uvVAR_dev/'
    cdir += 'local_tests/QSOwnd_var_Warm_and_Hot_varTST_hden14/CL_output'
    cl_ob._loadCL_SEDs(cdir, cl_prefix='CL_run2', attrname='mean')
        
    
    plt.loglog(cl_ob.cl_nus, cl_ob.Lref_mean[:, 0])
    
    cl_ob.subtract_lines('mean')
    
    plt.loglog(cl_ob.cl_nus, cl_ob.Lref_mean[:, 0])
    
    plt.ylim(1e43, 2e45)
    plt.xlim(1e14, 2e16)
    
    
    new_nu = np.geomspace(1e14, 1e16, 100)
    cl_ob.rebin(new_nu, 'mean')
    
    plt.loglog(new_nu, cl_ob.Lref_mean[:, 0])
    
    
    plt.show()
                
    
    
        
