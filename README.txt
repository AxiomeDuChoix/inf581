Before running our code, you'll have to add our new environment to the 'gym' package, that should thus be installed (for example with 'pip install gym' or 'conda install -c akode gym').

HOW TO ADD THE OPTIMAL LIQUIDATION ENVIRONMENT TO THE GYM PACKAGE ?

Since 'Anaconda' is the most widely used Python package manager, we detail the procedure for an 'Anaconda' user, but it can of course be easily adapted. 

Our code is based on a 'gym' environment that should be added to the 'gym' package. To do this, you must go into the directory containing 'Anaconda'.
Depending from your operating system, this directory might be at different locations :

- Windows 10: C:/Users/<your-username>/Anaconda3/
- macOS: /Users/<your-username>/anaconda3
- Linux: /home/<your-username>/anaconda3

(Please note that this location might differ according to where 'Anaconda' was installed on your computer).

Then you have to get into the 'gym' directory. 

Example for Linux : 
~/anaconda3/lib/python3.7/site-packages/gym/

As you want to add a new environment, you then move to the subdirectory 'envs'.

1 - Edit the file '__init__.py' in this directory : .../anaconda3/.../gym/envs/__init__.py
by inserting the OptimalLiquidation environment in the 'Toy Text' environment index. 
To do this, copy and paste the following code:

register(
    id='OptimalLiquidation-v0',
    entry_point='gym.envs.toy_text:OptimalLiquidationEnv',
    max_episode_steps=1000,
)

in the '# Toy Text' section from the '__init__.py' file.

2 - Then go into the subdirectory 'toy_text' (i.e. '.../anaconda3/.../gym/envs/toy_text') containing the 'Toy Text' environments, and edit the '__init__.py' file from this directory (i.e. '.../anaconda3/.../gym/envs/toy_text/__init__.py'.)
Add the 'Optimal Liquidation' environment to 'Toy Text' by inserting the following line of code at the end of the '__init__.py' file :

from gym.envs.toy_text.optimal_liquidation import OptimalLiquidationEnv

3 - Now you just have to add the Python file 'optimal_liquidation.py' provided in our code, which encodes the new environment, to the 'toy_text' directory where you are located (i.e. '.../anaconda3/.../gym/envs/toy_text/optimal_liquidation.py').

