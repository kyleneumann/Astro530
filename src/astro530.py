from astropy import units as u
from astropy import constants as const
import numpy as np
import math

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
    
    unit = (num[0]/den[0]).unit
    
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
    from scipy.special import expn
    
    S = src_func(tval, **kwargs)
    
    if n == 1 and abs(tval-tau) == 0: return S*229.7  # E1(1e-100)
    
    En = expn(n,abs(tval-tau))
    
    return S*En

def astrophysical_flux(t_arr, tmin = 0, tmax = 1e2, n_size = 1e-5, src_func = source_func, 
                       int_wrapper=simpson_wrapper, **kwargs):
    """ Astrophysical Flux H_ν(t)
    Given an array of τ_ν, an optional function variable and keywords for said 
    function, this function will output the astrophysical flux at zero optical 
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
    
    H_ν(0) [float]: Outputs the value for the astrophysical flux at a τ_ν = 0.
    """
    
    t_arr = np.array(t_arr)
#     S_arr = src_func(t_arr,**kwargs)
#     E2_arr = expn(2,t_arr)
    
#     y_arr = np.array(S_arr * E2_arr)
    
    Hv = np.zeros(len(t_arr))
    
    for i in range(len(t_arr)):
        tv = t_arr[i]
        
        if tv < 0: raise ValueError("Optical depth cannot be negative. Fix index = "+str(i))
        
        outward = int_wrapper(tv,tmax,n_size = n_size, scale = "log", function=SvxEn, 
                                  tau = tv, n = 2, src_func = src_func, **kwargs)
        inward = -int_wrapper(tmin,tv,n_size = n_size, scale = "log", function=SvxEn, 
                                      tau = tv, n = 2, src_func=src_func, **kwargs)
        Hv[i] = 1/2*(outward+inward)
    #return y_arr
    
    return Hv

