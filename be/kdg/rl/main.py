import matplotlib
import pygame as pygame

from be.kdg.rl.agent.agent import TabularAgent, Agent
from be.kdg.rl.environment.openai import FrozenLakeEnvironment
from be.kdg.rl.learning.learningstrategy import LearningStrategy
from be.kdg.rl.learning.tabular.qlearning import Qlearning, NStepQlearning
import gymnasium as gym

import matplotlib.pyplot as plt

from be.kdg.rl.learning.tabular.tabular_learning import TabularLearner
from be.kdg.rl.utils import parameters

plt.rcParams['figure.dpi'] = 300
plt.rcParams.update({'font.size': 17})

if __name__ == '__main__':

    matplotlib.use("TkAgg")  # activate tkinter
    experiment = parameters.current_experiment  # choose experiment to run
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


    # matplotlib.use("TkAgg")  # activate tkinter
    #
    # environment = FrozenLakeEnvironment()
    # environment.reset()
    # strategie = Qlearning(environment)
    #
    #
    # agent = TabularAgent(environment,strategie,n_episodes=50)
    # agent.train()

    # # create an Agent that uses Qlearning Strategy
    # agent: Agent = TabularAgent(environment, Qlearning(environment))
    # agent.train()

    # # create an Agent that uses NStepQlearning Strategy
    # agent: Agent = TabularAgent(environment, NStepQlearning(environment, 5))
    # agent.train()





