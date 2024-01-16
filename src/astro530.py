from astropy import units as u
from astropy import constants as const
import numpy as np
import math

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
        Bv.append((num[i]/den[i]).value)     
    
    #den = math.exp(h*c/kB * nu/T)-1
    return (Bv*unit/u.sr).to(u.erg/u.s/u.sr/u.cm**2/u.Hz)
  
def BoxInt(x_list,y_list,style = "left"):
    """Box Integrator
    
    input parameters:
    
    x_list (list-like): List of x values for each given y value.
    y_list (list-like): List of y values for each given x value.
    style (string): "right", "left", and "mp". Determines how the integral sum is set up. "left" is where the rectangle's height is the left y-value per step. "right" is where the rectangle's height is the right y-value per step. "mp" is where the rectangle's height is the midpoint (mean) between the two adjacent y-values per step.
    
    output parameters:
    
    area (float): area under the given curve
    """
    if len(x_list) != len(y_list):
        raise ValueError("x_list and y_list must be the same length.")
    
    dx_list = []
    for i in range(len(x_list)):
        if i == len(x_list)-1:
            pass
        else:
            dx = x_list[i+1]-x_list[i]
            if dx < 0:
                raise ValueError("x_list must be in ascending order.")
            dx_list.append(dx)
            
    if len(x_list) - len(dx_list) != 1:
        raise ValueError("Logic is lost.")
        
    h_list = []
    
    if style == "left":
        for i in range(len(dx_list)):
            h_list.append(y_list[i])
    elif style =="right":
        for i in range(len(dx_list)):
            h_list.append(y_list[i+1])
    elif style =="mp":
        for i in range(len(dx_list)):
            midpoint = (y_list[i]+y_list[i+1])/2
            h_list.append(midpoint)
    else: raise ValueError("The style variable must be either 'left', 'right', or 'mp'.")
    
    area = 0
    for i in range(len(dx_list)):
        area += dx_list[i]*h_list[i]
        
    return area

def funct_BoxInt(xmin, xmax, n = 100, style="mp", function=Planck, **kwargs):
    """ Functional Box Integrator
    Given some parameters and a function to model, with kwargs, this will use 
    the included BoxInt to integrate the function.
    
    input parameters:
    
    xmin (float): minimum x value of integration.
    
    xmax (float): maximum x value of integration.
    
    n (int): number of steps in integration.
    
    style (string): "right", "left", and "mp". Determines how the integral sum 
    is set up. "left" is where the rectangle's height is the left y-value per 
    step. "right" is where the rectangle's height is the right y-value per step.
    "mp" is where the rectangle's height is the midpoint (mean) between the two 
    adjacent y-values per step.
    
    function: mathematical function that outputs numerical values. Must be able 
        to accept keyword arguments. The "x" parameter must be the first 
        variable in the function.
        
    **kwargs: variables to input into mathematical function. Make sure the names 
        agree with the given function. This is only required for the unchanging 
        argument as the x-axis is being integrated over. 
    """
    
    n = int(n)
    
    x_list = np.linspace(xmin,xmax,n)
    y_list = function((x_list* u.dimensionless_unscaled).value,**kwargs)
#     y_list = function(**kwargs)
    
    return BoxInt(x_list,y_list,style)

    



