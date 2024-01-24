from astropy import units as u
from astropy import constants as const
import numpy as np
import math

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

def funct_BoxInt(xmin, xmax, n_den = 100, style="mp", scale = "linear", function=linear_func, **kwargs):
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
    
    n = int((xmax - xmin)*n_den)
    
    if scale == "linear":
        x_list = np.linspace(xmin,xmax,n)
    elif scale == "log":
        x_list = np.geomspace(xmin,xmax,n)
    else:
        raise ValueError("scale must equal either 'linear' or 'log'")
    y_list = function((x_list* u.dimensionless_unscaled).value,**kwargs)
#     y_list = function(**kwargs)
    
    return BoxInt(x_list,y_list,style)