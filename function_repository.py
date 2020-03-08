import numpy as np
import random as rd                                                                                                                                                              
import matplotlib.pyplot as plt
import matplotlib.patches as patches                                                                                                                                                  
from scipy.signal import argrelextrema
from math import log10

class FunctionRepository():
     def __init__(self,x,omega_0):
          self.x=x
          self.omega_0=omega_0

     def cos_exp_function( self, period): # Compute the price for different amount of sold shares
          """
          Input: (np.array) x containing the different quantities the agent might sell
               (int) omega_0 : Initial quantity of shares the agent does own
               (int) period : Period of the cosine := size of the sets of high interest
          Output: (np.array) y containing the corresponding prices
          """
          log_omega_0 = int(log10(self.omega_0))
          return np.exp((self.omega_0-self.x)/10**(log_omega_0))+((self.omega_0 - self.x)/(period*10**(log_omega_0-1)))*np.cos(self.x/period*(2*np.pi)) - 1.0