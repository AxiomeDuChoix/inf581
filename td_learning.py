import numpy as np

class SarsaSolver():

	def __init__(self, env):
		self.env = env
		self.states = list(range(self.env.action_space.n))
		self.actions = list(range(self.env.action_space.n))

    def sarsa():

        # Initialize value functions with zeros
		q_array = np.zeros(shape=(len(self.states),len(self.actions)))