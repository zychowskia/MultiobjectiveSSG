import json
import numpy as np
import networkx as nx
from itertools import product

class FlipItGame:
    def __init__(self, json_data, n_objectives=2):
        self.n_objectives = n_objectives
        self.graph_data = json_data['graph']
        self.rounds = json_data['rounds']
        self.entry_nodes = set(json_data['entryNodes'])
        
        self.num_nodes = self.graph_data['vertexCount']
        self.nodes = list(range(self.num_nodes))
        
        self.G = nx.DiGraph()
        self.G.add_nodes_from(self.nodes)
        for edge in self.graph_data['edges']:
            self.G.add_edge(edge['from'], edge['to'])
            
        self.def_rewards = self._extend_payoffs(json_data['defenderRewards'])
        self.def_costs = self._extend_payoffs(json_data['defenderCosts'])
        self.att_rewards = self._extend_payoffs(json_data['attackerRewards'])
        self.att_costs = self._extend_payoffs(json_data['attackerCosts'])

        self.actions = list(range(self.num_nodes + 1))
        self.pure_strategies = list(product(self.actions, repeat=self.rounds))

    def _extend_payoffs(self, scalar_list):
        extended = []
        for val in scalar_list:
            vec = np.random.uniform(1.0, 10.0, self.n_objectives - 1).tolist()
            vec.append(val)
            extended.append(np.array(vec))
        return extended

    def play(self, def_strategy, att_strategy):
        control_state = {n: 0 for n in self.nodes} 
        
        def_payoff = np.zeros(self.n_objectives)
        att_payoff = np.zeros(self.n_objectives)
        
        for t in range(self.rounds):
            d_action = def_strategy[t]
            a_action = att_strategy[t]
            
            if d_action < self.num_nodes:
                def_payoff -= self.def_costs[d_action]
            if a_action < self.num_nodes:
                att_payoff -= self.att_costs[a_action]
                
            if d_action < self.num_nodes:
                control_state[d_action] = 0
            
            if a_action < self.num_nodes:
                if d_action != a_action:
                    can_flip = False
                    if a_action in self.entry_nodes:
                        can_flip = True
                    else:
                        preds = list(self.G.predecessors(a_action))
                        if any(control_state[p] == 1 for p in preds):
                            can_flip = True
                    
                    if can_flip:
                        control_state[a_action] = 1
            
            for n in self.nodes:
                if control_state[n] == 0:
                    def_payoff += self.def_rewards[n]
                else:
                    att_payoff += self.att_rewards[n]
                    
        return def_payoff, att_payoff