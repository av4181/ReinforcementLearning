from abc import ABC, abstractmethod
import numpy as np

from be.kdg.rl.agent.episode import Episode
from be.kdg.rl.environment.environment import Environment


class LearningStrategy(ABC):
    """
    Implementations of this class represent a Learning Method
    This class is INCOMPLETE
    """
    env: Environment

    def __init__(self, environment: Environment, λ, γ, t_max) -> None:
        self.env = environment
        self.λ = λ  # exponential decay rate used for exploration/exploitation (given)
        self.γ = γ  # discount rate for exploration (given)
        self.ε_max = 1.0  # Exploration probability at start (given)
        self.ε_min = 0.0005  # Minimum exploration probability (given)

        self.ε = self.ε_max  # (decaying) probability of selecting random action according to ε-soft policy
        self.t_max = t_max  # upper limit voor episode
        self.t = 0  # episode time step
        self.τ = 0  # overall time step

    @abstractmethod
    def next_action(self, state):
        pass
    # uit de policy pi een actie kiezen
    # de policy pi is een tabel met voor elke state een kansverdeling over de acties
    # Bij start :
    # Richtingen           L        D          R        U
    #               s1     0.25     0.25        0.25    0.25
    #               s2      0.25    0.25        0.25    0.25
    #               s3      0.25    0.25        0.25    0.25

    @abstractmethod
    def learn(self, episode: Episode):
        # ALGO 2
        # at this point subclasses insert their implementation
        # see for example be\kdg\rl\learning\tabular\tabular_learning.py
        # teller t telt per episode
        # teller tau update bij einde episode
        self.t += 1
        # self.τ += 1

    @abstractmethod
    def on_learning_start(self):
        """
        Implements all necessary initialization that needs to be done at the start of new Episode
        Each subclasse learning algorithm should decide what to do here
        """
        pass

    def done(self):
        return self.t > self.t_max

# TE VERVOLLEDIGEN
    def decay(self):
        # Reduce epsilon ε, because we need less and less exploration as time progresses
        # Zie het epsilon greedy algorithme
        # dus epsilon wordt kleiner en kleiner

        self.ε = self.ε_min + (self.ε_max - self.ε_min) * np.exp(-self.λ * self.τ)
        print("Huidige epsilon waarde" + str(self.ε))
        pass

    def on_episode_end(self):
        self.τ += 1