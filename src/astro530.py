from astropy import units as u
from astropy import constants as const
import numpy as np
import math
import pandas
pd = pandas

from scipy.special import expn
from scipy.interpolate import UnivariateSpline

from src import N_integrator
simpson_wrapper = N_integrator.simpson_wrapper

from scipy.special import wofz
m_Na = 22.9897*u.g/u.mol/const.N_A

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

def Planck_tau(t, nu=[10], Teff = 5000*u.K,**kwargs):
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
    print(t_arr)
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
        print(tv)
        
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
    try: rpf_df
    except:
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
    
    log_g0 = rpf_df.loc[rpf_df["Theta="]==species].log_g0.values[0]
    
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
    
def init_saha():
    global ion_df,nion_df,rpf_df
    try:
        ion_df = pandas.read_csv("data/ioniz.csv").fillna("-")
        nion_df = pandas.read_csv("data/nist_ioniz.csv").fillna(0)
        rpf_df = pandas.read_csv("data/RepairedPartitionFunctions.csv")
    except:
        ion_df = pandas.read_csv("../data/ioniz.csv").fillna("-")
        nion_df = pandas.read_csv("../data/nist_ioniz.csv").fillna(0)
        rpf_df = pandas.read_csv("../data/RepairedPartitionFunctions.csv")
            
    
def saha_LTE(species = "H", temp = 5000,Pe = None):
    try: 
        #print("it works")
        ion_df
    except:
        try:
            ion_df = pandas.read_csv("data/ioniz.csv").fillna("-")
            nion_df = pandas.read_csv("data/nist_ioniz.csv").fillna(0)
        except:
            ion_df = pandas.read_csv("../data/ioniz.csv").fillna("-")
            nion_df = pandas.read_csv("../data/nist_ioniz.csv").fillna(0)
    else: pass
        
    try:
        temp=temp.value
    except: pass
    
    species = species_name_correction(species)
    
    output = 0.665*temp**(5/2)
    
    try:
        Pe = (Pe.cgs).value
    except:
        pass
    
    try:
        if Pe != None:
            try:
                output /= Pe
            except:
                pass
    except:
        output /= np.array(Pe)
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
    #return 10**(np.log10(output) + np.log10(Ur1/Ur)+(-5040/temp*chi))
    return output*Ur1/Ur*10**(-5040/temp*chi)

def calc_Pe(species="H",T=4310*u.K,Pg=10**2.87*u.dyne/u.cm**2,Pe = None, Phi = None):
    try: ab_df
    except:
        try:
            ab_df = pandas.read_csv("data/SolarAbundance.csv").fillna("-")
        except:
            ab_df = pandas.read_csv("../data/SolarAbundance.csv").fillna("-")
    
    try:
        temp = species[:]
        temp[0] = "t"
    except:
        species = [species]
    
        
    # Units
    try:
        Pg = (Pg.to(u.dyne/u.cm**2)).value
        T = (T.to(u.K)).value
        if Pe != None: 
            Pe = (Pe.to(u.dyne/u.cm**2)).value
            
    except: pass
        
    A_list = []
    for element in species:
        try:
            A = 10**float((ab_df.loc[ab_df.element == element].logA).values[0])
        except:
            A = 0
        if element == "He":
            A_He = A
        A_list.append(A)
    A_arr = np.array(A_list)
    
    init_saha()
    
    try:
        if Pe == None:
            Pe = np.sqrt(saha_LTE(species="H",temp=T*u.K)*Pg)
#         Pe = Pg/(1+A_He)/A_He
#         for A in A_arr:
#             Pe *= A
    except: pass
    
    
    num = 0
    den = 0
    for i, element in enumerate(species):
        #print(element)
        try:
            phi = (Phi.loc[Phi.species==element].Phi).values[0]
        except:
            phi = saha_LTE(species=element,temp=T*u.K)
        num += A_arr[i]*(phi/Pe)/(1+phi/Pe)
        
        den += A_arr[i]*(1+(phi/Pe)/(1+phi/Pe))
    return Pg*u.dyne/u.cm**2*num/den

def true_Pe(species="H",T=5000*u.K,Pg=100*u.dyne/u.cm**2,tol=1e-8,single = False):
    dPe = 2*tol
    
    try: ab_df
    except:
        init_Abundance()
    
    if species == "all":
        species = []
        for element in ab_df.element:
            A = (ab_df.loc[ab_df.element == element].A).values[0]
            if A != "-" and A > tol:
                species.append(element)
                
    try:
        temp = species[:]
        temp[0] = "t"
    except:
        species = [species]
    
    Phi = init_Phi(species=species,T=T)
    #print(Phi)
    
    Pe = calc_Pe(species=species,T=T,Pg = Pg, Phi = Phi)

    Pe_list = [Pe.value]

    while dPe > tol:
        Pe_new = calc_Pe(species=species,T=T,Pg=Pg,Pe=Pe.value, Phi = Phi)
        dPe = abs(Pe-Pe_new)/Pe_new
        Pe = Pe_new
        Pe_list.append(Pe.value)
    if single:
        return Pe
    else:
        return np.array(Pe_list)*Pe.unit
    
def Pe_calc(species=None,T=5000*u.K,Pg=100*u.dyne/u.cm**2,tol=1e-8,single = True):
    """
    Best Version of calculating Pe
    """
    dPe = 2*tol
    
    try: ab_df
    except:
        init_Abundance()
    
    A_list = []
    single_element = False
    single_param = False
    
    try:
        if species == None:
            species = []
            for element in ab_df.element:
                A = (ab_df.loc[ab_df.element == element].A).values[0]
                if A != "-":# and A > tol:
                    species.append(element)
                    A_list.append(A)
    except:
        try:
            temp = species
            temp[0] = "K"
            for element in species:
                A = (ab_df.loc[ab_df.element == element].A).values[0]
                if A != "-":
                    A_list.append(A) 
                else:
                    A_list.append(0)
        except:
            element = species
            species = [species]
            A = (ab_df.loc[ab_df.element == element].A).values[0]
            if A != "-":
                A_list.append(A)
            else:
                A_list.append(0)
                
    try:
        Pg = Pg.to(u.dyne/u.cm**2)
        T = T.to(u.K)          
    except:
        Pg = Pg*u.dyne/u.cm**2
        T = T*u.K
                
    try: 
        Pg[0]
    except:
        single_param = True
        
    A_arr = np.array(A_list)
    
    Phi = init_Phi(species=species,T=T,dtype="array")
    #return Phi
    #A_arr = np.resize(A_arr,np.shape(Phi))
    #print(Phi)
    
    Pe = Pg.value*1/(1+A_arr[1])*np.sum(A_arr[2:])#calc_Pe(species=species,T=T,Pg = Pg, Phi = Phi)

    Pe_list = [Pe]
    
    
    if single_param:
        while dPe > tol:
            num = np.sum((A_arr*Phi/Pe)/(1+Phi/Pe))
            den = np.sum((A_arr*(1+(Phi/Pe)/(1+Phi/Pe))))

            Pe_new = Pg.value*num/den
            dPe = abs(Pe-Pe_new)/Pe_new
            Pe = Pe_new#.value
            Pe_list.append(Pe)
        if single:
            return Pe*Pg.unit
        else:
            return np.array(Pe_list)*Pg.unit
    else:
        n_param = len(Pg)
        Pe_list = Pe_list[0]
        Pe_list3 = []
        
        for i in range(n_param):
            Pe = Pe_list[i]
            Pe_list2 = []
            dPe = 2*tol
            
            while dPe > tol:
                num = np.sum((A_arr*Phi[:,i]/Pe)/(1+Phi[:,i]/Pe))
                den = np.sum((A_arr*(1+(Phi[:,i]/Pe)/(1+Phi[:,i]/Pe))))

                Pe_new = Pg[i].value*num/den
                dPe = abs(Pe-Pe_new)/Pe_new
                
                Pe = Pe_new#.value
                Pe_list2.append(Pe)
            if single:
                Pe_list3.append(Pe)
            else:
                Pe_list3.append(Pe_list2)
                
        return np.array(Pe_list3)*Pg.unit
    
def init_Abundance():
    global ab_df
    try:
        ab_df = pandas.read_csv("data/SolarAbundance.csv").fillna("-")
    except:
        ab_df = pandas.read_csv("../data/SolarAbundance.csv").fillna("-")
def find_Abundance(species = "H"):
    try: ab_df
    except: init_Abundance()
        
    try:
        A = (ab_df.loc[ab_df.element == species]["A"].to_numpy())[0]
        return A
    except:
        raise ValueError("Try different element")
        
def init_Phi(species=["H"],T=5000*u.K,dtype = "df"):
    if dtype != "array" or dtype != "df":
        dtype = "array"
    if dtype == "df":
        data = [["species","Phi"]]
        for element in species:
            data.append([element,saha_LTE(species=element,temp=T)])
        #print(data[0])
        df = pandas.DataFrame(data=data[1:],
                        columns=data[0])
        return df
    elif dtype == "array":
        data = []
        for element in species:
            data.append(saha_LTE(species=element,temp=T))
        return np.array(data)
    
    
def stim_em_coeff(wavelength=5000,T=5000):
    if (wavelength*u.s).unit == u.s:
        pass
    else:
        wavelength = (wavelength.to(u.angstrom)).value
        T = (T.to(u.K)).value
        #Pe = (Pe.to(u.dyne*u.cm**-2)).value
        
    chi = 1.2398e4/wavelength
    theta = 5040/T
    
    return (1-10**(-chi*theta)) 

def k_Hbf(T = 5000,wavelength = 1000,nmax = 100,stim_em_bool = False):    
    
    a0 = 1.0443e-26
    
    R=1.09678e-3
    
    if (wavelength*u.s).unit == u.s:
        pass
    else:
        wavelength = (wavelength.to(u.angstrom)).value
        T = (T.to(u.K)).value
    K=0
    for n in range(1,nmax+1):
        #print(n)
        lmax = n**2/R
        
        chi = 13.6*(1-1/n**2)
        gbf = 1.-0.3456/(wavelength*R)**(1/3) * (wavelength*R/n**2 - 1/2)
        
        try:
            for i,l in enumerate(wavelength):
                if l > lmax: gbf[i] = 0
        except:
            if lmax < wavelength: gbf = 0
        
        K+= (wavelength/n)**3*gbf*10**(-5040/T * chi)
    
    K *= a0
    
    if stim_em_bool:
        stim_coeff = stim_em_coeff(wavelength=wavelength,T=T)
    else:
        stim_coeff = 1
    
    return K*stim_coeff*u.cm**2

def k_Hff(T = 5000,wavelength = 1000,stim_em_bool=False):
    a0 = 1.0443e-26
    
    R=1.09678e-3
    
    if (wavelength*u.s).unit == u.s:
        pass
    else:
        wavelength = (wavelength.to(u.angstrom)).value
        T = (T.to(u.K)).value
        
    theta = 5040/T
    loge = 0.43429    
    I = ((const.h*const.c*const.Ryd).to(u.eV)).value    
    chi = 1.2398e4 / wavelength    
    gff = 1+ 0.3456/(wavelength*R)**(1/3) * (loge/(theta*chi)+1/2)
    
    K = a0*wavelength**3*gff*loge/(2*theta*I)*10**(-theta*I)
    
    if stim_em_bool:
        stim_coeff = stim_em_coeff(wavelength=wavelength,T=T)
    else:
        stim_coeff = 1
    
    return K*stim_coeff*u.cm**2

def k_Hnbf(Pe,T = 5000,wavelength = 1000, stim_em_bool=False):
    if (wavelength*u.s).unit == u.s:
        pass
    else:
        wavelength = (wavelength.to(u.angstrom)).value
        T = (T.to(u.K)).value
        (Pe.to(u.dyne*u.cm**-2)).value
    
    a0 = 0.1199654
    a1 = -1.18267e-6
    a2 = 2.64243e-7
    a3 = -4.40524e-11
    a4 = 3.23992e-15
    a5 = -1.39568e-19
    a6 = 2.78701e-24
    
    abf = (a0+a1*wavelength+a2*wavelength**2+a3*wavelength**3+a4*wavelength**4+
           a5*wavelength**5+a6*wavelength**6)*10**-17
    
    theta = 5040/T
    
    try:
        zeroed = False
        for i,a in enumerate(abf):
            if a <= 0 or zeroed:
                abf[i] = 0
                zeroed = True
    except:
        if abf <= 0 or zeroed:
            abf = 0
            zeroed = True
    
    K = 4.158e-10*abf*Pe*theta**(5/2) * 10**(0.754*theta)
    
    if stim_em_bool:
        stim_coeff = stim_em_coeff(wavelength=wavelength,T=T)
    else:
        stim_coeff = 1
    try:
        return K.value*stim_coeff*u.cm**2
    except:
        return K*stim_coeff*u.cm**2

def k_Hnff(Pe,T = 5000,wavelength = 9000):
    if (wavelength*u.s).unit == u.s:
        pass
    else:
        wavelength = (wavelength.to(u.angstrom)).value
        T = (T.to(u.K)).value
        Pe = (Pe.to(u.dyne*u.cm**-2)).value
    
        
    logl = np.log10(wavelength)
    logth = np.log10(5040/T)
        
    f0 = -2.2763 - 1.6850*logl + 0.76661*logl**2-0.053346*logl**3
    f1 = 15.2827-9.2846*logl+1.99381*logl**2-0.142631*logl**3
    f2 = -197.789+190.266*logl-67.9775*logl**2+10.6913*logl**3-0.625151*logl**4
    
    K = 1e-26*Pe*10**(f0+f1*logth+f2*logth**2)
    
    return K*u.cm**2

def k_e(Pe = 10*u.dyne/u.cm**2,Pg = 20*u.dyne/u.cm**2,species = "H"):
    
    if species == "all":
        try: ab_df
        except:
            init_Abundance()
        species = []
        for element in ab_df.element:
            A = (ab_df.loc[ab_df.element == element].A).values[0]
            if A != "-":
                species.append(element)
    #print(species)
    try:
        if Pe > Pg: raise ValueError("Electron pressure must be less than gas pressure")
    except:pass
    if (Pe*u.s).unit == u.s:
        if (Pg*u.s).unit != u.s:
            Pg = (Pg.to(u.dyne*u.cm**-2)).value
    else:
        Pe = (Pe.to(u.dyne*u.cm**-2)).value
        Pg = (Pg.to(u.dyne*u.cm**-2)).value
    
    #species = np.array(species)
    sumAj = 0
    try:
        for element in species:
            sumAj += find_Abundance(species = element)
    except:
        sumAj = find_Abundance(species = species)
    alpha = 0.6648*10**-24*u.cm**2
    return alpha*Pe/(Pg-Pe)*sumAj

def k_total(Pe=10*u.dyne/u.cm**2,T = 5000*u.K,wavelength = 9000*u.angstrom,Pg=None,species="H",norm_bool = False,**kwargs):
    if (wavelength*u.s).unit == u.s:
        pass
    else:
        wavelength = (wavelength.to(u.angstrom)).value
        T = (T.to(u.K)).value
        Pe = (Pe.to(u.dyne*u.cm**-2)).value
        
    KHbf = k_Hbf(T=T,wavelength=wavelength)
    KHff = k_Hff(T=T,wavelength=wavelength)
    KHnbf = k_Hnbf(Pe,T=T,wavelength=wavelength)
    KHnff = k_Hnff(Pe,T=T,wavelength=wavelength)
    
    chi = 1.2398e4/wavelength
    theta = 5040/T
    
    phi = saha_LTE("H",temp=T)
    
    try:
        if Pg != None:
            Ke = k_e(Pe=Pe,Pg=Pg,species=species)
        else: 
            Ke = 0*KHbf.unit
            norm_bool = True
    except:
        Ke = k_e(Pe=Pe,Pg=Pg,species=species)
        norm_bool = True
    
    if norm_bool:
        norm = 1/(1+phi/Pe)
    else:
        norm = 1
    
    Kt = ((KHbf+KHff+KHnbf)*(1-10**(-chi*theta))+KHnff)*norm
    
    return Kt+Ke

def opacity_500(tau=0.9,print_bool = False):
    """
        Uses t500 to calculate opacity
    """
    try: val_df
    except:
        init_VAL()
    t500 = val_df.tau_500.to_numpy()
    T_arr = val_df["T"].to_numpy()
    Pg_arr = val_df.Ptotal.to_numpy()*val_df["Pgas/Ptotal"].to_numpy()
    
    T = UnivariateSpline(t500,T_arr,s=0,k=2)(tau)*u.K
    Pg = UnivariateSpline(t500,Pg_arr,s=0,k=2)(tau)*u.Ba
    
    Pe = Pe_calc(T=T,Pg=Pg,single=True)
    if print_bool:
        print("T:",T)
        print("Pg:",Pg)
        print("Pe:",Pe)
    
    return k_alt(wavelength=5000*u.Angstrom,T=T,Pg=Pg,Pe=Pe)    
    
    
def k_cont(wavelength=5000*u.Angstrom,T=5000*u.K,Pe=50*u.dyn/u.cm**2,Pg=2e5*u.dyn/u.cm**2,print_bool = False,**kwargs):

    sAjmu = abundance_mass()
    species = []
    for element in ab_df.element:
        A = (ab_df.loc[ab_df.element == element].A).values[0]
        if A != "-":
            species.append(element)
   
    K_tot = k_total(Pe=Pe,T=T,wavelength=wavelength,Pg=Pg,species=species,**kwargs)
    
    if print_bool:
        print("T =",T)
        print("P_e =",Pe)
        print("P_g =",Pg)
        print("Wavelength =",wavelength)
        print("Mean particle mass =",sAjmu)
        print("Opacity =",K_tot/sAjmu)
        
    
    return K_tot/sAjmu

def abundance_mass():
    sAjmu = 0
    try:
        ab_df
    except:
        init_Abundance()
    
    amu = (const.N_A)**-1*1*u.g/u.mol        
    
    species = []
    for element in ab_df.element:
        A = (ab_df.loc[ab_df.element == element].A).values[0]
        if A != "-":
            species.append(element)
    
    for ele in species:
        df_e = ab_df.loc[ab_df["element"]==ele]
        A = df_e.A.to_numpy()[0]
        mu = df_e.weight.to_numpy()[0]
        sAjmu += A*mu
        
    return sAjmu * amu
    
def init_VAL():
    global val_df
    try:
        openfile = open("data/VALIIIC_sci_e.txt","r")
    except:
        openfile = open("../data/VALIIIC_sci_e.txt","r")
    openlines = openfile.readlines()
    openfile.close()

    table = []
    for i,line in enumerate(openlines):
        data = line.split()
        if len(data) == 0: continue
        if data[0] == "#":
            if i == 0:
                header = data[1:]
            #print(line)
            continue
        table.append(data)

    table = np.array(table,dtype=float)

    val_df = pd.DataFrame(table,columns = header)
    
    return val_df
    

def Voigt(a,u):
    return np.real(wofz(u+1j*a))
def Voigt_wrapper_Na(nu, T=5000*u.K, Pg=300*u.dyn/u.cm**2, uturb=0*u.km/u.s, n_e = 10**6/u.cm**3,ideal_gas = False):
    c = const.c
    try: 
        nu = nu.to(u.Hz)
        T = T.to(u.K)
        Pg = Pg.to(u.Ba)
        uturb = uturb.to(u.cm/u.s)
    except: 
        nu = nu*u.Hz
        T = T *u.K
        Pg = Pg*u.Ba
        uturb = uturb*u.cm/u.s
    nu0 = (np.array([5889.95,5895.924])*u.Angstrom).to(u.Hz,equivalencies=u.spectral())
    
    delta_nu = nu0/c*np.sqrt((2*const.k_B*T/m_Na)+uturb**2)
    
    if ideal_gas:
        Pe = n_e*const.k_B*T
    else:
        Pe = Pe_calc(Pg=Pg,T=T)
    try:
        nu[0]
        single_nu = False
    except:
        single_nu = True
        
    if single_nu:
        gamma = lorentz_Na(nu, Pe=Pe,Pg=Pg,T=T)
        #eta = (nu-nu0)*c/nu0
        
        u0 = (nu-nu0)/delta_nu
        a0 = gamma/(4*np.pi*delta_nu)
#         print("delta_nu =",delta_nu.cgs)
#         print("u =",u0.cgs)
#         print("a =",a0.cgs)
        return Voigt(a0,u0)/(np.sqrt(np.pi)*delta_nu)
    else: 
        V_list = []
        for v in nu:
            gamma = lorentz_Na(v, Pe=Pe, Pg=Pg,T=T)
            #eta = (v-nu0)*c/nu0
        
            u0 = (v-nu0)/delta_nu
            a0 = gamma/(4*np.pi*delta_nu)
            Vo = Voigt(a0,u0)
            
            V_list.append(Vo.value)
        V_arr = np.array(V_list)*Vo.unit
        #return V_arr.cgs
        V0 = V_arr[:,0]/(np.sqrt(np.pi)*delta_nu[0])
        V1 = V_arr[:,1]/(np.sqrt(np.pi)*delta_nu[1])
        return (np.array([V0,V1]).T*V0.unit).cgs
    
def lorentz_Na(nu, Pe = 100*u.Ba, Pg = 300*u.Ba, T = 5000*u.K):
    nu0 = (np.array([5889.95,5895.924])*u.Angstrom).to(u.Hz,equivalencies=u.spectral())
    
    # Radiation
    yrad = 4*np.pi *np.array([6.16,6.14])*10**7*u.Hz
    
    # Gamma4
    logC4 = np.array([-15.17,-15.33])
    logy4 = 19+2/3*logC4+np.log10(Pe/(1*u.Ba))-5/6 * np.log10(T/(1*u.K))
    
    # Gamma6
    I = 5.17*u.eV
    chi_n = 0*u.eV#2.1*u.eV
    chi_nu = const.h*nu0#nu
#     print(I-chi_n-chi_nu)
#     print(I-chi_n)
    
    logC6 = np.log10(0.3)-30+np.log10(((1/(I-chi_n-chi_nu)**2-1/(I-chi_n)**2).to(1/u.eV**2)).value)
    #print(logC6)
    #logy6 = 20+2/5*np.log10(C6.cgs.value)+np.log(Pg/u.Ba)-7/10*np.log10(T/u.K)
    logy6 = 20+2/5*logC6+np.log10(Pg/u.Ba)-7/10*np.log10(T/u.K)
    
    y4 = 10**logy4
    y6 = 10**logy6
    
#     print("logy4 =",logy4)
#     print("logy6 =",logy6)
    return yrad + (y4+y6)*u.Hz
def mono_line_ext_NA(nu, T=5000*u.K, Pg=300*u.dyn/u.cm**2, uturb=0*u.km/u.s,n_e = 10/u.cm**3,print_bool=False,**kwargs):
    c = const.c
    h = const.h
    try: 
        nu = nu.to(u.Hz)
        T = T.to(u.K)
        Pg = Pg.to(u.Ba)
        uturb = uturb.to(u.cm/u.s)
    except: 
        nu = nu*u.Hz
        T = T *u.K
        Pg = Pg*u.Ba
        uturb = uturb*u.cm/u.s
    nu0 = (np.array([5889.95,5895.924])*u.Angstrom).to(u.Hz,equivalencies=u.spectral())
    Aul = np.array([6.16,6.14])*10**7*u.Hz
    gu = np.array([4,2])
    gl = np.array([2,2])
    Bul = Aul*c**2/(2*h*nu0**3)
    Blu = gu/gl*Bul
    delta_nu = nu0/c*np.sqrt((2*const.k_B*T/m_Na)+uturb**2)
    #print(Blu.cgs)
    ϕ = Voigt_wrapper_Na(nu, T=T, Pg=Pg, uturb=uturb,n_e=n_e,**kwargs)
#     print(ϕ)
#     print(ϕ*np.sqrt(np.pi)*delta_nu)
    #print(np.sqrt(np.pi)*delta_nu*ϕ)
    if print_bool:
        print("Δλ_D:", (c*delta_nu/(nu0)**2).to(u.AA))
    try:
        sig_0 = h*nu/(4*np.pi)*Blu[0]*ϕ[:,0]
        sig_1 = h*nu/(4*np.pi)*Blu[1]*ϕ[:,1]
        return 4*np.pi*(np.array([sig_0,sig_1]).T*sig_0.unit).cgs
    except:
        return (h*nu*Blu*ϕ).cgs
    
def line_opacity_Na(nu,T=5000*u.K,Pg=100*u.Ba,uturb=0*u.km/u.s,rho = 10*u.g/u.cm**3,nH = 10**6/u.cm**3,ne = 10**6/u.cm**3,print_bool=False,**kwargs):
        
    σ = mono_line_ext_NA(nu,T=T,Pg=Pg,uturb=uturb,n_e=ne,print_bool=print_bool,**kwargs)
    A = find_Abundance("Na")
    Pe = ideal_gas(ne,T) #Pe_calc(T=T,Pg=Pg)
    fe = 2/partition(species="Na",temp=T.value)
    fi = 1/(1+saha_LTE(species="Na",temp=T,Pe = Pe))
    spectral_emission = (1-np.exp(-const.h*(np.array([5890,5896])*u.Angstrom).to(u.Hz,equivalencies=u.spectral())/(const.k_B*T)))
    if print_bool:
        print("A =",A)
        print("Pe =",Pe)
        print("fe =",fe)
        print("fi =",fi)
        print("SEF =", spectral_emission)
        print("σ =", σ)
    
    try:
        #print("work")
        return σ*A*nH/rho*fi*fe*spectral_emission

    except:
        print("Something broke, value may be off.")
        kappa_list = []
        for s in σ:
            kappa_list.append(s*A*nH/rho*fi*fe*spectral_emission)
        return np.array(kappa_list)*kappa_list[0].unit

def ideal_gas(n,T):
    return n*const.k_B*T
    
    