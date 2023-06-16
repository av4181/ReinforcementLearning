from collections import deque

from be.kdg.rl.agent.percept import Percept
from be.kdg.rl.environment.environment import Environment
import numpy as np
from numpy.random import choice


class Episode:
    """
    A collection of Percepts forms an Episode. A Percept is added per step/time t.
    The Percept contains the state, action, reward and next_state.
    This class is INCOMPLETE
    """

    def __init__(self, env: Environment) -> None:
        self._env = env
        self._percepts: [Percept] = deque()
        self.Gt = 0  # return Gt (discounted sum of rewards)

    def add(self, percept: Percept):
        self._percepts.append(percept)

    def percepts(self, n: int):
        """ Return n final percepts from this Episode """
        return list(self._percepts)[-n:]

    # TE VERVOLLEDIGEN
    def compute_returns(self,t,λ) -> None:
        """ For EACH Percept in the Episode, calculate its discounted return Gt"""
        if t < (self.size - 1):
            p = self._percepts[t+1]
            self.Gt += np.exp(λ, t) * p.reward
            t += 1
            self.compute_returns()
        else:
            return self.Gt
            print(f'The discounted sum of rewards at timestamp {t}: {self.Gt}')

    # TE VERVOLLEDIGEN
    def sample(self, batch_size: int):
        """ Sample and return a random batch of Percepts from this Episode """
        sample = choice(self._percepts, batch_size)
        return sample

    @property
    def size(self):
        return len(self._percepts)
