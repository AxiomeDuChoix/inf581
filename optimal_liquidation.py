import gym
import numpy as np
import sys
from gym.envs.toy_text import discrete

################ PARAMETRES ################
# Actions détenues
Omega0 = 7
OmegaT = 0
deltaOmega = np.abs(Omega0-OmegaT)
# Quantité d'actions sur le marché
q0 = 0
qT = 7
# Prix de marché:
p0 = 8
pT = 1
deltaPrix = np.abs(p0-pT)
# Actions
ACTIONS = range(deltaOmega+1)

################ CLASSE PRINCIPALE ################

class OptimalLiquidationEnv(discrete.DiscreteEnv):

    metadata = {'render.modes': ['human', 'ansi']}
    
    def _offre_demande_lineaire(self,x): ### Dans la représentation intuitive. 
        resu = p0+(x-Omega0)*(pT-p0)/(OmegaT-Omega0)
        return int(resu)
    
    def _reflexion_p(self,p):
        return self.shape[0]-p

    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, prix, Omega, action):
        delta_abscisse = np.array([0, -action])
        vertical_winds = self._offre_demande_lineaire(Omega)-self._offre_demande_lineaire(Omega-action)
        reward = action*(prix-vertical_winds)

        new_position = np.array(current) + delta_abscisse + np.array([1, 0]) * vertical_winds
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)

        p = 1.0
        if (new_position[0]>=self.shape[0]) or (new_position[1]<0):
            p = 0.0
        is_done = (new_position[0]==self.shape[0]-1) or (new_position[1]==0)


        return [(1.0, new_state, reward, is_done)]

    def __init__(self):
        s1 =deltaPrix+1
        s2 = deltaOmega+1
        self.shape = (s1, s2)
        self.n_actions = len(ACTIONS)
        self.start = (self._reflexion_p(p0),Omega0-1)
        nS = np.prod(self.shape)
        nA = len(ACTIONS)
        # Calculate transition probabilities
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            (old_p,Omega)=position
            p = self._reflexion_p(old_p)
            P[s] = { a : [] for a in range(nA) }
            for action in ACTIONS:
                P[s][action] = self._calculate_transition_prob(position, p, Omega, action)
        # We always start in state (p0, Omega0)
        isd = np.zeros(nS)
        print("START = {}".format(self.start))
        print("SHAPE = {}".format(self.shape))
        isd[np.ravel_multi_index(self.start, self.shape)] = 1.0

        # We create an instance of the "discrete" environment
        super(OptimalLiquidationEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human', close=False):
        self._render(mode, close)

    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            # print(self.s)
            if self.s == s:
                output = " x "
            elif position == self.goal:
                output = " T "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"
            outfile.write(output)
        outfile.write("\n")