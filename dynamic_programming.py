import numpy as np
import copy

class DynamicProgrammingSolver():

	def __init__(self, env):
		self.env = env
		self.states = list(range(self.env.observation_space.n))
		self.actions = list(range(self.env.action_space.n))

		is_final_array = np.full(shape=len(self.states), fill_value=False, dtype=np.bool)
		transition_array = np.zeros(shape=(len(self.states), len(self.actions), len(self.states)))
		reward_array = np.full(shape=(len(self.states),len(self.actions),len(self.states)), fill_value=0)
		for state in self.states:
			for action in self.possible_actions(state):
				for next_state_tuple in self.env.P[state][action]:
					transition_probability, next_state, next_state_reward, next_state_is_final = next_state_tuple
					is_final_array[next_state] = next_state_is_final
					transition_array[state, action, next_state] = transition_probability
					reward_array[state, action, next_state] = next_state_reward
		
		self.transition_array = transition_array
		self.reward_array = reward_array
		self.is_final_array = is_final_array

	def possible_actions(self, state):
		omega,_=np.unravel_index(state,self.env.shape)
		return range(omega+1)

	def greedy_policy(self, state, v_array, gamma):
		return np.array([(self.transition_array[state, action]*(self.reward_array[state, action] + gamma * v_array)).sum() for action in self.possible_actions(state)]).argmax()

	def value_iteration(self, gamma=1.0, epsilon=0.001):
		# Initialize value functions with zeros
		v_array = np.zeros(len(self.states))   
		stop = False
		delta_history = []

		while not stop:

			delta = 0.

			new_v_array = copy.deepcopy(v_array)
	   
			for state in self.states:
				if self.is_final_array[state]:
					new_v_array[state] = 0
				else:
					new_v_array[state] = max([(self.transition_array[state, action]*(self.reward_array[state, action] + gamma*v_array)).sum() for action in self.possible_actions(state)])
				
				delta = max(abs(new_v_array[state] - v_array[state]), delta)
			
			delta_history.append(delta)
			v_array = new_v_array
			
			if delta < epsilon:
				stop = True

		policy = [self.greedy_policy(state, v_array, gamma) for state in self.states]       

		return v_array, policy, delta_history


	def policy_evaluation(self, policy, gamma, epsilon):
		# Initialize value functions with zeros
		v_array = np.zeros(len(self.states))   
		stop = False


		while not stop:

			delta = 0.

			new_v_array = copy.deepcopy(v_array)
			
			for state in self.states:
				if self.is_final_array[state]:
					new_v_array[state] = 0
				else:
					action = policy[state]
					new_v_array[state] = (self.transition_array[state, action]*(self.reward_array[state, action] + gamma * v_array)).sum()
				
				delta = max(abs(new_v_array[state] - v_array[state]), delta)
			
			v_array = new_v_array
			
			if delta < epsilon:
				stop = True

		return v_array



	def policy_iteration(self, gamma=1, epsilon=0.001):

		# Random initial policy
		policy = np.random.randint(low=min(self.actions), high=max(self.actions), size=len(self.states), dtype='int')  

		stop = False

		while not stop:

			v_array = self.policy_evaluation(policy, gamma, epsilon)

			new_policy = np.copy(policy)

			for state in self.states:

				new_policy[state] = self.greedy_policy(state, v_array, gamma)

			if np.array_equal(new_policy, policy):
				stop = True
			else:
				policy = new_policy

		return v_array, policy

