import matplotlib
import pygame as pygame

from be.kdg.rl.agent.agent import TabularAgent, Agent, DQNAgent
from be.kdg.rl.environment.openai import FrozenLakeEnvironment, CartPoleEnvironment
from be.kdg.rl.learning.learningstrategy import LearningStrategy
from be.kdg.rl.learning.tabular.qlearning import Qlearning, NStepQlearning
from be.kdg.rl.learning.approximate.deep_qlearning import DeepQLearning
import gymnasium as gym

import matplotlib.pyplot as plt

from be.kdg.rl.learning.tabular.tabular_learning import TabularLearner
from be.kdg.rl.utils import parameters

plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 17})

if __name__ == '__main__':

    # https://towardsdatascience.com/the-values-of-actions-in-reinforcement-learning-using-q-learning-cb4b03be5c81
    # Algemeen Q learning algo :

    # Hyperparameters
    # ---------------
    # Bepaal de stepsize alfa tussen 0 en 1
    # Bepaal epsilon tussen 0 en 1 voor eploitation/exploration tradeoff
    # Bepaal de discount factor gamma voor future rewards en Gt bepaling
    # Initialiseer de Q tabel waar alle waardes random zijn behalve voor de terminal points zoals cadeau en gat
    # Bepaal het aantal episodes N
    #
    # Itereer van 1 tot N (aantal episodes)
    # -------------------------------------
    # neem een start state s
    # itereer tot de agent een terminal state bereikt
    # in een gegeven state s, kies een actie a door gebruik te maken van epsilon greedy
    # neem deze actie a en observeer (percept) wat de overgangsstate s' en zijn reward r zijn
    # update de Q tabel Q(S,A) + alfa*(r + gamma*maxQ(s',a) - Q(S,A)
    # Q(S,A) state waar onze agent zich in bevindt
    # maxQ(s',a) deel is de maximaal beschikbare Q-waarde in de staat waarin onze agent terechtkomt, over alle acties heen
    # en r is reward voor naar s' te gaan
    # set state s naar s'


    experiment = parameters.current_experiment  # zie parameters.py
    parameters.init()  # initialize local environment

    learning = parameters.params.get("experiment").get(experiment).get("learning")
    env_name = parameters.params.get("experiment").get(experiment).get("environment") + "()"
    agent = parameters.params.get("experiment").get(experiment).get("agent")
    n = parameters.params.get("experiment").get(experiment).get("n")
    ddqn = parameters.params.get("experiment").get(experiment).get("ddqn")

    environment = eval(env_name)
    agent: Agent = eval(agent + "(environment," + learning + "(environment" +
                        ("))" if n is None and ddqn is None else "," + n +
                                                                 ("))" if ddqn is None else "," + ddqn + "))"))
                        )
    agent.train()






