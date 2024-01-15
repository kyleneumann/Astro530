from astropy import units as u
from astropy import constants as const
import math

def Planck(nu,T):
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
        Bv.append((num[i]/den[i]).value)     
    
    #den = math.exp(h*c/kB * nu/T)-1
    return (Bv*unit/u.sr).to(u.erg/u.s/u.sr/u.cm**2/u.Hz)
  
    



