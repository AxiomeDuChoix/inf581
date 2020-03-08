##########################################################
# This file was used to plot the figures from the report #
##########################################################

import numpy as np
import random as rd                                                                                                                                                               
import matplotlib.pyplot as plt
import matplotlib.patches as patches                                                                                                                                                  
from scipy.signal import argrelextrema
from math import log10
import function_repository

class PlotFunction():
 # Hyperparamters, that can be modified
     omega_0 = 122 # Initial quantity of shares the agent A does own
     period = 25 # Period of the cosine := size of the sets of interest for the buyer

     step = 0.1 # Resolution of the plot 
     plt.rcParams["figure.figsize"] = (8,8) # Size for the plot  

     def __init__(self, env):
	     self.env = env

    
                    


     def plot_prices(self,quantities_sold, prices, x_high_price, y_high_price): # Plot the prices with an highlight on the high interest sets in the 'coffee' example
          """
          Input: (np.array) quantities_sold containing the different quantities the agent might sell
               (np.array) prices containing the corresponding sell prices
               (np.array) x_high_price containing the sets of high interest for the buyer
               (np.array) y_high_price containing the corresponding sell prices
          """ 
          plt.ylim(np.min(prices), np.max(prices)*1.01) 
          
          # Plot the price values
          plt.plot(quantities_sold, prices) 
          plt.title('Selling price of coffee according to the quantity of coffee sold')                    
          plt.xlabel('Quantity of coffee sold (arbitrary units)') 
          plt.ylabel('Selling price (arbitrary units)')
          
          # Visualize the sets of high interest 
          plt.vlines(x_high_price, -0.1, y_high_price,color='r')
          plt.scatter(x_high_price, np.array([0 for i in range(len(x_high_price))]), marker='x', color='red')
          for i in range(len(x_high_price)):
               plt.text(x_high_price[i]-1.5, y_high_price[i]+0.1, str(x_high_price[i]), fontsize=10, color='red')  
          
          plt.text(quantities_sold[-1] * 0.55, prices[0] * 0.9, 'Volumes of shares of interest' + '\n' + 'to buyers are shown in red', fontsize= 12, color='red')
          plt.show() 

     def selling_strategy(self,quantities_sold, prices, step, period, omega_0, decision='random'):
          """
          Output : (np.array) sells containing the successive cumulative quantities of shares sold
          """
          max_sold_shares = 1.5 * period # The maximum amount of shares the agent is allowed to sell at each iteration
          if decision == 'random': # Random decisions from the agent 
               sells = [0]
               sold_shares = 0
               while sold_shares < quantities_sold[-1]:
                    possible_sold_quantity = quantities_sold[int(sold_shares/step):int(sold_shares/step)+int(max_sold_shares/step)]
                    new_sell = int(rd.choice(possible_sold_quantity))
                    sells += [new_sell]
                    sold_shares = new_sell
          else: # Greedy strategy : he targets the local maxima
               sells = [int(argrelextrema(prices, np.greater)[0][i]*step) for i in range(len(argrelextrema(prices, np.greater)[0]))]
               sells = [0] + sells + [omega_0]
               
          return np.array(sells)
     
     def plot_price_strategy(self,quantities_sold, prices, sells):
          # Allow the visualization in the coffee example
          profit = 0
          plt.figure(figsize=(8,8))
          plt.plot(quantities_sold, prices)
          plt.title('Selling price of coffee according to the quantity of coffee sold')                    
          plt.xlabel('Quantity of coffee sold (arbitrary units)') 
          plt.ylabel('Selling price (arbitrary units)')
          for i in range(len(sells)-1):
               # Plot the rectangle
               profit += (quantities_sold[int(sells[i+1]/step)]-quantities_sold[int(sells[i]/step)])*prices[int(sells[i+1]/step)]  
               rectangle = plt.Rectangle((quantities_sold[int(sells[i]/step)],0), quantities_sold[int(sells[i+1]/step)]-quantities_sold[int(sells[i]/step)], prices[int(sells[i+1]/step)], fc='None',ec="green",hatch='/',fill='False')
               plt.gca().add_patch(rectangle)
          plt.text(quantities_sold[-1] * 0.55, prices[0] * 0.9, 'Profit =' + str(profit), fontsize= 12, color='green')
          plt.autoscale()
          plt.show()

     def main(self):
          # Run main to plot all the figures from the report
          
          quantities_sold = np.arange(0,omega_0+step,step) # Different quantities that the agent does sell                                                                                                                                                                                                                                    

          x_high_price = np.array([period * i for i in range(1,int(omega_0 / period)+1)]) # The packs of shares of interest for the buyer
          y_high_price = price_function(x_high_price, omega_0, period)
          
          # Plot the price function
          prices = function_repository.cos(quantities_sold, omega_0, period)     
          plot_prices(quantities_sold, prices, x_high_price, y_high_price)
          
          # Plot the reward of a random selling strategy
          sells_rd = selling_strategy(quantities_sold, prices, step, period, omega_0, 'random')
          plot_price_strategy(quantities_sold, prices, sells_rd)
          
          # Plot the reward of an optimal selling strategy
          sells_opt = selling_strategy(quantities_sold, prices, step, period, omega_0, 'optimal')
          plot_price_strategy(quantities_sold, prices, sells_opt)
          


