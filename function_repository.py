import numpy as np
import random as rd                                                                                                                                                              
import matplotlib.pyplot as plt
import matplotlib.patches as patches                                                                                                                                                  
from scipy.signal import argrelextrema
from math import log10
import math

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

     def brownian_decr_function(self):
          N=1220
          t=np.arange(N)
          b=13
          a=100
          X=[10]
          for i in range(N):
               X+=[X[-1]+b*np.cos(i)+a*np.random.randn(1)]

          #Δt_sqrt = math.sqrt(1 / N)
          #Z = np.random.randn(N)
          #Z[0] = 0
          #B = np.cumsum(Δt_sqrt * Z)
          return X