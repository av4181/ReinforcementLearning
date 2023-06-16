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

    # matplotlib.use("TkAgg")  # activate tkinter
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






