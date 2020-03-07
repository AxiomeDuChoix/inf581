import numpy as np

class TDLearningSolver():

	def __init__(self, env):
		self.env = env
		self.states = list(range(self.env.action_space.n))
		self.actions = list(range(self.env.action_space.n))
 
    def epsilon_greedy_policy(self, state, epsilon, q_array): 
        if np.random.uniform(0, 1) < epsilon: 
            action = self.env.action_space.sample() 
        else: 
            action = np.argmax(q_array[state, :]) 
        return action 

    def greedy_policy(self, state, q_array):
        return np.argmax(q_array[state, :])

    #Sarsa solver:
    def sarsa(self, alpha = 0.9, gamma = 0.95, epsilon = 0.1, total_episodes = 1000, t = 100):
        
        #Initialize value functions with zeros
		q_array = np.zeros(shape=(len(self.states),len(self.actions)))
        q_array_memory = np.zeros(shape=(total_episodes//t,len(self.states),len(self.actions)))

        #SARSA learning 
        for episode in range(total_episodes): 
            t = 0
            state1 = self.env.reset() 
            action1 = self.epsilon_greedy_policy(state1, epsilon, q_array) 

            while (not done):

                #Keeping Q-values for training visualization
                if episode%t == 0:
                    q_array_memory[episode//t] = q_array
                
                #Making an action in the environment 
                state2, reward, done, info = self.env.step(action1) 
        
                #Choosing the next action 
                action2 = self.epsilon_greedy_policy(state2, epsilon, q_array) 
                
                #Learning the Q-value 
                td = reward + gamma * q_array[state2,action2] - q_array[state1,action1]
                q_array[state1,action1] = q_array[state1,action1] + alpha * td 
        
                state1 = state2 
                action1 = action2 
            
        return q_array, q_array_memory
           

    #Q_learning solver:
    def q_learning(self, alpha = 0.9, gamma = 0.95, epsilon = 0.1, total_episodes = 1000, t = 100)

        #Initialize value functions with zeros
        q_array = np.zeros(shape=(len(self.states),len(self.actions)))
        q_array_memory = np.zeros(shape=(total_episodes//t,len(self.states),len(self.actions)))

        #Qlearning 
        for episode in range(total_episodes): 
            t = 0
            state1 = self.env.reset() 

            while (not done): 
        
                #Keeping Q-values for training visualization
                if episode%t == 0:
                    q_array_memory[episode//t] = q_array

                #Making an action in the environment 
                action1 = self.epsilon_greedy_policy(state1, epsilon, q_array)
                state2, reward, done, info = self.env.step(action1) 
        
                #Choosing the next action, this time following the greedy policy
                action2 = self.greedy_policy(state2, epsilon, q_array) 
                
                #Learning the Q-value 
                td = reward + gamma * q_array[state2,action2] - q_array[state1,action1]
                q_array[state1,action1] = q_array[state1,action1] + alpha * td 
        
                state1 = state2 

        return q_array, q_array_memory