from copy import deepcopy

import numpy as np

from gat.environment.state import State


class Env:
    def __init__(self):
        pass

    def __create_state(self, graph_size):
        graph = np.random.uniform(size=(graph_size, 2))
        return State(graph)

    def step(self, action):
        '''
        return state, totalcost, done
        '''

        done = self.state.update(action)
        if done:
            return self.state, -self.state.total_cost(), done
        return self.state, 0, done

    def reset(self, graph_size):
        '''
        return state
        '''
        self.state = self.__create_state(graph_size)
        return self.state

    def render(self):
        self.state.render()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
