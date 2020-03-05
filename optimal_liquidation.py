import gym
import numpy as np
import sys
from gym.envs.toy_text import discrete

################ PARAMETRES ################
T = 3
deltaOmega = 7
################ CLASSE PRINCIPALE ################
class OptimalLiquidationEnv(discrete.DiscreteEnv):

    def _offre_demande_lineaire(self,x): ### Dans la représentation intuitive. 
        return x+1
    def _actions_posibles(self, omega):
        return range(omega+1)
    def _next(self,omega,t,action):
        assert(omega-action>=0)
        if (t>0):
            return (omega-action,t-1)
        else:
            assert (t==0)
            return (omega,t)
    def _calculate_transition_prob(self, s, action):
        # delta_abscisse = np.array([0, -action])
        # vertical_winds = self._offre_demande_lineaire(Omega)-self._offre_demande_lineaire(Omega-action)
        # reward = action*(prix-vertical_winds)

        # new_position = np.array(current) + delta_abscisse + np.array([1, 0]) * vertical_winds
        # new_position = self._limit_coordinates(new_position).astype(int)
        # new_state = np.ravel_multi_index(tuple(new_position), self.shape)

        #État actuel:
        (omega,t)=np.unravel_index(s,self.shape)
        #Nouvel état (tuple):
        new_state_tuple = self._next(omega,t,action)
        new_state_int = np.ravel_multi_index(new_state_tuple,self.shape)
        #Reward:
        nouveau_prix = self._offre_demande_lineaire(new_state_int)
        reward = action*(nouveau_prix)
        # État final ou pas:
        is_done = ((omega-action == 0) or (t-1==0))
        # if (is_done):
        #     print("({},{})->{}->({},{})".format(omega,t,action, new_state_tuple[0],new_state_tuple[1]))
        # Si on n'est pas arrivé en 0 après T étapes, on ne gagne rien. 
        if (t-1==0 and omega-action>0):
            reward = 0
        return [(1.0, new_state_int, reward, is_done)]

    def __init__(self):
        self.start = (deltaOmega,T)
        self.shape = (deltaOmega+1,T+1)
        self.nS = np.prod(self.shape)
        self.nA = deltaOmega+1
        # Calculate transition probabilities
        P = {}
        for s in range(self.nS):
            omega,t = np.unravel_index(s,self.shape)
            P[s] = { a : [] for a in  self._actions_posibles(omega)}

            for action in self._actions_posibles(omega):
                P[s][action] = self._calculate_transition_prob(s, action)
        # We always start in state (deltaOmega, T)
        isd = np.zeros(self.nS)
        isd[np.ravel_multi_index(self.start, self.shape)] = 1.
        # We create an instance of the "discrete" environment
        super(OptimalLiquidationEnv, self).__init__(self.nS, self.nA, P, isd)

    def render(self, mode='human', close=False):
        self._render(mode, close)

    # def _render(self, mode='human', close=False):
    #     if close:
    #         return
    #     outfile = StringIO() if mode == 'ansi' else sys.stdout
    #     for s in range(self.nS):
    #         position = np.unravel_index(s, self.shape)
    #         # print(self.s)
    #         if self.s == s:
    #             output = " x "
    #         elif position == self.goal:
    #             output = " T "
    #         else:
    #             output = " o "

    #         if position[1] == 0:
    #             output = output.lstrip()
    #         if position[1] == self.shape[1] - 1:
    #             output = output.rstrip()
    #             output += "\n"
    #         outfile.write(output)
    #     outfile.write("\n")