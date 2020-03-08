import numpy as np
import random as rd                                                                                                                                                               
import matplotlib.pyplot as plt
import matplotlib.patches as patches                                                                                                                                                  
from scipy.signal import argrelextrema
from math import log10

class FunctionRepository():
	def __init__(self, env):
	     self.env = env
     def cos_exp_function( self, x, omega_0, period):
          
          # Compute the price for different amount of sold shares
          """
          Input: (np.array) x containing the different quantities the agent might sell
               (int) omega_0 : Initial quantity of shares the agent does own
               (int) period : Period of the cosine := size of the sets of high interest
          Output: (np.array) y containing the corresponding prices
          """
          log_omega_0 = int(log10(omega_0))
          return np.exp((omega_0-x)/10**(log_omega_0))+((omega_0 - x)/(period*10**(log_omega_0-1)))*np.cos(x/period*(2*np.pi)) - 1.0