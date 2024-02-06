from astropy import units as u
from astropy import constants as const
import numpy as np
import math

from scipy.integrate import simpson

def linear_func(x,m=1,b=0):
    return m*x + b

def BoxInt(x_list,y_list,style = "left"):
    """Box Integrator
    
    input parameters:
    
    x_list (list-like): List of x values for each given y value.
    y_list (list-like): List of y values for each given x value.
    style (string): "right", "left", and "mp". Determines how the integral sum is set up. "left" is where the rectangle's height is the left y-value per step. "right" is where the rectangle's height is the right y-value per step. "mp" is where the rectangle's height is the midpoint (mean) between the two adjacent y-values per step.
    
    output parameters:
    
    area (float): area under the given curve
    """
    
    x_unit = (x_list*u.g/u.g).unit
    x_list = (x_list*u.g/u.g).value
    
    y_unit = (y_list*u.g/u.g).unit
    y_list = (y_list*u.g/u.g).value
    
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
    
    area = np.sum(np.array(dx_list)*np.array(h_list))
#     for i in range(len(dx_list)):
#         area += dx_list[i]*h_list[i]
        
    return area*x_unit*y_unit

def funct_BoxInt(xmin, xmax, n_size = 1e-2, n_den = 100, style="mp", scale = "linear", function=linear_func, **kwargs):
    """ Functional Box Integrator
    Given some parameters and a function to model, with kwargs, this will use 
    the included BoxInt to integrate the function.
    
    input parameters:
    
    xmin (float): minimum x value of integration.
    
    xmax (float): maximum x value of integration.
    
    n_den (float or int): number of steps in a single x unit for integration.
    
    style (string): "right", "left", and "mp". Determines how the integral sum 
    is set up. "left" is where the rectangle's height is the left y-value per 
    step. "right" is where the rectangle's height is the right y-value per step.
    "mp" is where the rectangle's height is the midpoint (mean) between the two 
    adjacent y-values per step.
    
    scale (string): "linear" or "log" depending on how the steps will be spread out within each x unit.
    
    function: mathematical function that outputs numerical values. Must be able 
        to accept keyword arguments. The "x" parameter must be the first 
        variable in the function.
        
    **kwargs: variables to input into mathematical function. Make sure the names 
        agree with the given function. This is only required for the unchanging 
        argument as the x-axis is being integrated over. 
    """
    if xmax < xmin:
        raise ValueError("xmax must be greater than xmin.")
    elif xmax == xmin: 
        return 0.
    
    if n_size != 1e-2:
        n = int((xmax - xmin)/n_size)
    else:
        n = int((xmax - xmin)*n_den)
    
    if scale == "linear":
        x_list = np.linspace(xmin,xmax,n)
    elif scale == "log":
        x_list = np.float_power(10,np.arange(np.log10(xmin),np.log10(xmax),n_size))
    else:
        raise ValueError("scale must equal either 'linear' or 'log'")
    y_list = function((x_list* u.dimensionless_unscaled).value,**kwargs)
#     y_list = function(**kwargs)
    
    return BoxInt(x_list,y_list,style)

def BoxInt_unitless(x_list,y_list,style = "left"):
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
        h_list = y_list[0:len(dx_list)]
#         for i in range(len(dx_list)):
#             h_list.append(y_list[i])
    elif style =="right":
        h_list = y_list[1:len(dx_list)+1]
#         for i in range(len(dx_list)):
#             h_list.append(y_list[i+1])
    elif style =="mp":
        for i in range(len(dx_list)):
            midpoint = (y_list[i]+y_list[i+1])/2
            h_list.append(midpoint)
    else: raise ValueError("The style variable must be either 'left', 'right', or 'mp'.")
    
    area = 0
    
    area = np.sum(np.multiply(np.array(dx_list),np.array(h_list)))
#     for i in range(len(dx_list)):
#         area += dx_list[i]*h_list[i]
        
    return area

def funct_BoxInt_unitless(xmin, xmax, n_size = 1e-2, n_den = 100, style="mp", scale = "linear", function=linear_func, **kwargs):
    """ Functional Box Integrator
    Given some parameters and a function to model, with kwargs, this will use 
    the included BoxInt to integrate the function.
    
    input parameters:
    
    xmin (float): minimum x value of integration.
    
    xmax (float): maximum x value of integration.
    
    n_den (float or int): number of steps in a single x unit for integration.
    
    style (string): "right", "left", and "mp". Determines how the integral sum 
    is set up. "left" is where the rectangle's height is the left y-value per 
    step. "right" is where the rectangle's height is the right y-value per step.
    "mp" is where the rectangle's height is the midpoint (mean) between the two 
    adjacent y-values per step.
    
    scale (string): "linear" or "log" depending on how the steps will be spread out within each x unit.
    
    function: mathematical function that outputs numerical values. Must be able 
        to accept keyword arguments. The "x" parameter must be the first 
        variable in the function.
        
    **kwargs: variables to input into mathematical function. Make sure the names 
        agree with the given function. This is only required for the unchanging 
        argument as the x-axis is being integrated over. 
    """
    if xmax < xmin:
        raise ValueError("xmax must be greater than xmin.")
    elif xmax == xmin: 
        return 0.
    
    if n_size != 1e-2:
        n = int((xmax - xmin)/n_size)
    else:
        n = int((xmax - xmin)*n_den)
    
    if scale == "linear":
        x_list = np.linspace(xmin,xmax,n)
    elif scale == "log":
        x_list = np.float_power(10,np.arange(np.log10(xmin),np.log10(xmax),n_size))
        
    else:
        raise ValueError("scale must equal either 'linear' or 'log'")
    y_list = function(x_list,**kwargs)
#     y_list = function(**kwargs)
    
    return BoxInt_unitless(x_list,y_list,style)

def converge(minlims=[1e-16,1], maxlims=[1.1,100], nlims=[1,1e5], iterations=30, 
             tol=1e-6, cycles = 1, function = linear_func, **kwargs):

    cycles = int(cycles)
    iterations = int(iterations)
    
    if minlims[1] >= maxlims[0]:
        raise ValueError("The upper minimum limit cannot be greater than the lower maximum limit.")
    
    unit = (minlims[0] * u.g/u.g).unit
    
    xmins = np.geomspace(minlims[1],minlims[0],iterations)*unit

    xmaxs = np.linspace(maxlims[0],maxlims[1],iterations)*unit
    xmax = 13*unit

    n_vals = np.geomspace(nlims[0],nlims[1],iterations)
    n = 10

    min_index = 0
    max_index = 0
    n_index = 0

    dy_min = []
    dy_max = []
    dy_n = []

    indexes = []
    for runs in range(cycles):
        min_index = 0
        max_index = 0
        n_index = 0

        dy_min = []
        dy_max = []
        dy_n = []

        dy = 1
        for i in range(len(xmins)):
            if dy > tol:
                xmin = xmins[i]

                min_index = i
                if i != 0:
                    current = funct_BoxInt(xmin,xmax,n=n,function=function,**kwargs)
                    past = funct_BoxInt(xmins[i-1],xmax,n=n,function=Planck,**kwargs)
                    dy = abs(current-past)/past
                    dy_min.append(dy.value)
                    indexes.append(i)

        dy = 1          
        for i in range(len(xmaxs)):
            if dy > tol:
                xmax = xmaxs[i]

                max_index = i
                if i != 0:
                    current = funct_BoxInt(xmin,xmax,n=n,function=Planck,**kwargs)
                    past = funct_BoxInt(xmins[i-1],xmax,n=n,function=Planck,**kwargs)
                    dy = abs(current-past)/past
                    dy_max.append(dy.value)

        dy = 1             
        for i in range(len(n_vals)):
            if dy > tol:
                n = int(n_vals[i])

                n_index = i
                if i != 0:
                    current = funct_BoxInt(xmin,xmax,n=n,function=Planck,**kwargs)
                    past = funct_BoxInt(xmins[i-1],xmax,n=n,function=Planck,**kwargs)
                    dy = abs(current-past)/past
                    dy_n.append(dy.value)
                    
    return (dy_min, dy_max, dy_n)

def simpson_wrapper(xmin, xmax, n_size = 1e-2, n_den = None, scale = "linear", function = linear_func, **kwargs):
    """ Functional Simpson Integrator
    Given some parameters and a function to model, with kwargs, this will use 
    Simpson rule via scipy to integrate the function.
    
    input parameters:
    
    xmin (float): minimum x value of integration.
    
    xmax (float): maximum x value of integration.
    
    n_size (float or int): step size in linear or log space.
    
    style (string): "right", "left", and "mp". Determines how the integral sum 
    is set up. "left" is where the rectangle's height is the left y-value per 
    step. "right" is where the rectangle's height is the right y-value per step.
    "mp" is where the rectangle's height is the midpoint (mean) between the two 
    adjacent y-values per step.
    
    scale (string): "linear" or "log" depending on how the steps will be spread out within each x unit.
    
    function: mathematical function that outputs numerical values. Must be able 
        to accept keyword arguments. The "x" parameter must be the first 
        variable in the function.
        
    **kwargs: variables to input into mathematical function. Make sure the names 
        agree with the given function. This is only required for the unchanging 
        argument as the x-axis is being integrated over. 
    """
    
    if xmax < xmin:
        raise ValueError("xmax must be greater than xmin.")
    elif xmax == xmin: 
        return 0.
    
    if n_den == None or n_den <= 0:
        n = int((xmax - xmin)/n_size)
    else:
        n = int((xmax - xmin)*n_den)
        
    if scale == "linear":
        x_arr = np.linspace(xmin, xmax, n)
    elif scale == "log":
        logxmax = np.log10(xmax)
        
        if xmin < 0: raise ValueError("Don't use negative bounds when using logspace.")
        
        elif xmin == 0:
            logxmin = np.log10(1e-16)
            if logxmax == -16: 
                return 0
            x_arr = np.logspace(logxmin,logxmax, int((logxmax-logxmin)/n_size-1))
            x_arr = np.append([0],x_arr)
        else:
            logxmin = np.log10(xmin)
            x_arr = np.logspace(logxmin,logxmax, int((logxmax-logxmin)/n_size))
        
        
    else:
        raise ValueError("scale must equal either 'linear' or 'log'")
        
    y_list = function(x_arr,**kwargs)
    
#     sim_arr = []
    
#     for i in range(np.shape(y_list)[0]):
#         sim_arr.append(simpson(y_list[i], x = x_arr))
    
    return simpson(y_list, x = x_arr)