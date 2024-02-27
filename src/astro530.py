from astropy import units as u
from astropy import constants as const
import numpy as np
import math
import pandas

from scipy.special import expn
from scipy.interpolate import UnivariateSpline

from src import N_integrator
simpson_wrapper = N_integrator.simpson_wrapper

def Planck(nu=1,T=1000):
    """Planck Function

    Input Parameters:

    nu [um^-1]: Wavenumber of the Planck function in units of microns^-1 as a single value or list like.
    T [K]:  Temperture in kelvin

    Output: B_nu (T)

    """
    
    # Get length of nu and tell if its a single value or list-like
    try:
        length = len(nu)
        single = False
    except:
        length = 1
        single = True
        
    try:
        len(T)
        raise TypeError("T must be a single value, not list-like")
    except:
        pass
    
    try:
        nu=(nu.to(u.um**-1)).value
        T=(T.to(u.K)).value
    except: pass
    
    nu = nu/u.um
    T = T*u.K
    
    kB = const.k_B
    c = const.c
    h = const.h
    
    num = 2*h*c*nu**3
    exponent = h*c/kB * nu/T
    #print(exponent)
    
    if single:
        den = math.exp(exponent)-1
        return (num/den/u.sr).to(u.erg/u.s/u.sr/u.cm**2/u.Hz)
    
    den = []
    
    for e in exponent:
        den.append(math.exp(e)-1)
       
    Bv = []
    try:
        unit = (num[0]/den[0]).unit
    except:
        print("Error causing value:",num/den)
        raise ValueError("Unknown Error")
    
    for i in range(length):
        temp = (num[i]/den[i]).value
        Bv.append(temp)     
    Bv = np.array(Bv)    
    Bv[np.isnan(Bv)] = 0
    
    #den = math.exp(h*c/kB * nu/T)-1
    return (Bv*unit/u.sr).to(u.erg/u.s/u.sr/u.cm**2/u.Hz)

def Planck_Int(Temp):
    """Planck Integral
    Given temperature, the function will calculate the analytic solution to the planck function's integral from wavenumber 0 to Infinity
    
    input parameter:
    
    Temp (float): Temperature in units of Kelvin
    """
    
    h = const.h
    kB = const.k_B
    pi = np.pi
    c = const.c
    
    Temp = Temp*u.K
    Temp = Temp.value
    
    analytic_sol = (2*pi**4/(15*h**3*c**3)*(kB*Temp*u.K)**4/u.sr).to(u.erg/u.s/u.sr/u.cm**2/u.Hz/u.um)
    
    return analytic_sol

def Planck_tau(t, nu=[10], Teff = 5000*u.K):
    """Planck Function in terms of tau
    Default units are um^-1 for wavenumber and K for effective temperature.
    
    Input Parameters:
        t [array-like]: Array or list of optical depth values or a singlular value
        nu [array-like]: Array or list of wavenumber 
        Teff [float or astropy Quantity]: Value for effective temperature. Default is 5000 K.
    
    Output: B_ν (τ) [erg/s/cm^2/sr/Hz

    """
    
    # Get length of nu and tell if its a single value or list-like
    try:
        len_nu = len(nu)
    except:
        len_nu = 1
        nu = [nu]
        
    try:
        len_t = len(t)
        t_arr = np.array(t)
    except:
        len_t = 1
        t_arr = np.array([t])
        
    if (t_arr[0] * u.g).unit != u.g: 
        raise TypeError("Optical depth, t, must be unitless")
        
    if (nu[0] * u.g).unit == u.g:
        nu_arr = np.array(nu)/u.um
    else:
        nu_arr = np.array(nu)/u.um
        
    if (Teff * u.g).unit == u.g:
        Teff = Teff * u.K
    
    kB = const.k_B
    c = const.c
    h = const.h
    
    T_arr = Teff*(3/4*(t_arr+2/3))**(1/4)
    
    Bv_arr = np.zeros((len_nu,len_t))
    
    for i in range(len_nu):
        nu = nu_arr[i]
        num = 2*h*c*nu**3
        
        for j in range(len_t):
            T=T_arr[j]
            exponent = h*c/kB * nu/T
            
            den = math.exp(exponent)-1  
            if np.isnan((num/den).value):
                Bv_arr[i,j] = 0
            else:
                Bv_arr[i,j] = (num/den).value
            
            if i == 0 and j == 0:
                Bv_unit = (num/den).unit/u.sr
    return (Bv_arr*Bv_unit).to(u.erg/u.s/u.sr/u.cm**2/u.Hz)

def source_func(t, a0 = 1, a1 = 1, a2 = 1):
    """ Radiative transfer source function
    Given τ_ν as a single value or an array, and key word values for a_n where 
    n < 3, the function will output the source function value. This is viable 
    up to quadratic form.
    
    input parameters:
    
    t [float or array]: values for τ_ν, the optical depth.
    a0 [float]: Named variable for the zeroth order source term
    a1 [float]: Named variable for the linear source term
    a2 [float]: Named variable for the quadratic source term. Set to zero for a 
                linear source function.
                
    output values:
    
    S_ν [float or array]: outputs the value(s) for the source function. 
    
    """
    return a0+a1*t+a2*t**2

def SvxEn(tval, tau = 0, n=2, src_func = source_func, **kwargs):
    S = src_func(tval, **kwargs)
    
    if n == 1 and abs(tval-tau) == 0: return S*229.7  # E1(1e-100)
    
    En = expn(n,abs(tval-tau))
    
    return S*En

def eddington_flux(t_arr, tmin = 0, tmax = 1e2, n_size = 1e-5, src_func = source_func, 
                       int_wrapper=simpson_wrapper, **kwargs):
    """ Eddington Flux H_ν(t)
    Given an array of τ_ν, an optional function variable and keywords for said 
    function, this function will output the eddington flux at zero optical 
    depth as calculated using numerical integration with scipy.integrate.simpson  
    
    input parameters:
    
    t_arr [array-like]: values for optical depth from 0 to infinity.
    src_dunc [function]: name of the source function's function which is given 
                         at least optical depth values in array form. 
    int_wrapper [function]: name of the integration wrapper function which will 
                            numerically solve the flux problem. Make sure the 
                            function's inputs follow the same format as 
                            N_integrator.simpson_wrapper. 
    kwargs: keyword arguments for the source function and integrator.
    
    output values:
    
    H_ν(0) [float]: Outputs the value for the eddington flux at a τ_ν = 0.
    """
    t_arr = np.array([t_arr])
    src0 = src_func(t_arr[0], **kwargs)
    src_shape = np.shape(src0)
    
    if (src0*u.g).unit == u.g:
        Hv_unit = 1
    else:
        Hv_unit = src0.unit * u.sr
        
    if src_shape[0] == 1:    
        Hv = np.zeros(len(t_arr))
        twoD = False
    else:
        Hv = np.zeros((src_shape[0],len(t_arr))).T
        twoD = True
    
    len_t = len(t_arr)
    
    for i in range(len_t):
        
        tv = t_arr[i]
        
        if tv < 0: raise ValueError("Optical depth cannot be negative. Fix index = "+str(i))
        
        outward = int_wrapper(tv,tmax,n_size = n_size, scale = "log", function=SvxEn, 
                                  tau = tv, n = 2, src_func = src_func, **kwargs)
        inward = -int_wrapper(tmin,tv,n_size = n_size, scale = "log", function=SvxEn, 
                                      tau = tv, n = 2, src_func=src_func, **kwargs)
        if twoD:
            Hv_temp = 1/2*(outward+inward)
            for j in range(src_shape[0]):
                Hv[i,j] = Hv_temp[j]
        else:
            Hv[i] = 1/2*(outward+inward)
        if len_t > 4:
            if i == 0: print(round(100/len_t,0),"% Done")    
            if i%int(len_t/4): print("25% Done")
            if i%int(len_t/2): print("50% Done")
            if i%int(3*len_t/4): print("75% Done")
            
    return Hv*Hv_unit

def species_name_correction(species):
    species_list = list(species)
    if len(species_list) > 3:
        if species_list[-1] == "I" and species_list[-2] == "I" and species_list[-3] == "I":
            species = "".join(species[:-3]+"++")
        elif species_list[-1] == "I" and species_list[-2] == "I":
            species = "".join(species[:-2]+"+")
        elif species_list[-1] == "I":
            species = "".join(species[:-1])
            
    elif len(species_list) > 2:
        if species_list[-1] == "I" and species_list[-2] == "I":
            species = "".join(species[:-2]+"+")
        elif species_list[-1] == "I":
            species = "".join(species[:-1])
            
    elif len(species_list) > 1:
        if species_list[-1] == "I":
            species = "".join(species[:-1])
            
    return species
def partition(species="H",temp=5000,s_val = 0,k_val = 2,func = None, **kwargs):
    """Partition function from Gray 3 ed.
    
    Inputs:
    
    species (string): Species name you want a partition function for. Can be 
                    inputted as XIII or X++. If species is not in Gray, function 
                    will tell you.
                    
    temp (float or array): Temperature of material to look at in Kelvin.
    
    s_val: for spline value
    
    k_val: for spline curve
    
    Outputs:
    
    Partition function value of a species given a temperature"""
    
    try:
        rpf_df = pandas.read_csv("data/RepairedPartitionFunctions.csv")
    except:
        rpf_df = pandas.read_csv("../data/RepairedPartitionFunctions.csv")
        
    species = species_name_correction(species)
    
    if species == "H+": 
        species = "He"
        
    elif species == "H-": 
        chi = 0.755
        g0 = 1
        theta = 5040/temp
        
        return g0*10**(-theta*chi)
    try:
        data = rpf_df.loc[rpf_df["Theta="]==species].to_numpy()[0][1:-1]
    except:
        species = "He"
        data = rpf_df.loc[rpf_df["Theta="]==species].to_numpy()[0][1:-1]
    
    log_g0 = float(rpf_df.loc[rpf_df["Theta="]==species].log_g0.values[0])
    
    th = 5040/temp
    
    theta = []
    temp_data = []
    
    for i,d in enumerate(data):
        if d != "-":
            temp_data.append(float(d))
            theta.append(0.2*(i+1))
            
    if log_g0 != "-":
        temp_data.append(log_g0)
        theta.append(10)
        temp_data.append(log_g0)
        theta.append(12)
        temp_data.append(log_g0)
        theta.append(15)
        temp_data.append(log_g0)
        theta.append(20)
        
    data = np.array(temp_data,dtype="float")
    theta = np.array(theta)
    
    if func == None:
        p_US = UnivariateSpline(theta,data,s=0,k=k_val)
        if k_val != 1:
            temp = np.linspace(0.2,20,50)
            p_US = UnivariateSpline(temp, p_US(temp),s=0,k=1)
        
        output_arr = p_US(th)
        try:
            for i,theta in enumerate(th):
                if theta > 20:
                    output_arr[i] = log_g0
        except:
            if th > 20:
                output_arr = log_g0
        return 10**output_arr
    else: 
        print("Function is not ready for this choice") 
        return 0
    
def saha_LTE(species = "H", temp = 5000,Pe = None):
    try:
        ion_df = pandas.read_csv("data/ioniz.csv").fillna("-")
        nion_df = pandas.read_csv("data/nist_ioniz.csv").fillna(0)
    except:
        ion_df = pandas.read_csv("../data/ioniz.csv").fillna("-")
        nion_df = pandas.read_csv("../data/nist_ioniz.csv").fillna(0)
        
    try:
        temp=temp.value
    except: pass
    
    species = species_name_correction(species)
    
    output = 0.665*temp**(5/2)
    
    if Pe != None:
        try:
            output /= Pe
        except:
            pass
    s_list = list(species)
    
    r = 0
    
    for c in s_list:
        if c == "+":
            r+=1
    Ur = partition(species=species,temp=temp)
    
    if species != "H-":
        Ur1 = partition(species=species+"+",temp=temp)

        #print(nion_df.loc[nion_df["Element"]==species])
        try:
            chi = (nion_df.loc[nion_df["Element"]==species]["1ion"]).values[0]
        except:
            raise ValueError("Try another species, this one cannot be found.")
            
    else:
        Ur1 = partition(species="H",temp=temp)
        chi = 0.755
        #print(Ur)
    return 10**(np.log10(output) + np.log10(Ur1/Ur)+(-5040/temp*chi))
    
    