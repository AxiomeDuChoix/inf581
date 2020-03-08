import plot_price_function
import function_repository
import matplotlib.pyplot as plt
import numpy as np

#plot_agent=plot_price_function.PlotFunction(122,25,0.1)
#plot_agent.main()

browny=function_repository.FunctionRepository(100,27)
depp=browny.brownian_decr_function()
johnny=np.arange(1221)
plt.plot(johnny,depp, color='green')

gabillon=browny.brownian_decr_function()
victor=np.arange(1221)
plt.plot(victor,gabillon, color='red')

horseman=browny.brownian_decr_function()
bojack=np.arange(1221)
plt.plot(bojack,horseman, color='blue')
plt.show()

