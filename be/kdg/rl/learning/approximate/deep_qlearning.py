from keras import Model

import numpy as np
import random
from collections import deque

from be.kdg.rl.agent.episode import Episode
from be.kdg.rl.environment.environment import Environment
from be.kdg.rl.learning.learningstrategy import LearningStrategy
from be.kdg.rl.learning.model import model1

# DEEP Q LEARNING = RL ALGORITME + ANN

class DeepQLearning(LearningStrategy):
    """
    Two neural nets q1 en q2 are trained together and used to predict the best action.
    These nets are denoted as Q1 and Q2 in the pseudocode.
    This class is INCOMPLETE.
    """
    q1: Model  # deze wordt getraind om de beste actie te voorspellen
    q2: Model  # wordt gebruikt om een training set te maken voor q1

    # The initial state is fed as an input to the neural network and returns the Q-value of all possible actions as an
    # output.
    # The policies can be represented using a lookup table, linear functions, or neural networks depending on the
    # complexity of the action space and state space for the environment.  An optimal policy is derived by selecting the
    # highest valued action in each state.

    def __init__(self, environment: Environment, batch_size: int, ddqn=False, λ=0.0005, γ=0.99, t_max=200) -> None:
        super().__init__(environment, λ, γ, t_max)
        self.batch_size = batch_size
        self.ddqn = ddqn  # al of niet gebruiken double deep q learning network
        # TODO: COMPLETE THE CODE
        # ALGORITME 7
        self.c = 10
        self.q1 = model1.create_model("model1", self.env.state_size, self.env.n_actions)
        self.q2 = model1.create_model("model1", self.env.state_size, self.env.n_actions)
        self.max_timesteps = 0  # save longest balancing

    def on_learning_start(self):
        # TODO: COMPLETE THE CODE
        # BEGIN TIMESTEP 0. STOPPEN BIJ MAX 500
        # INDIEN 195 TIMESTEPS WERDE GEHAALD IS GOED
        self.t = 0

    def next_action(self, state):
        """ Neural net decides on the next action to take """
        # zie learningstrategy epsilon decay
        # TODO: COMPLETE THE CODE
        exploitation_tradeoff = random.uniform(0, 1)
        if exploitation_tradeoff > self.ε:
            action = np.argmax(self.q1.predict(np.reshape(state, [1, self.env.state_size])))
        else:
            action = self.env.action_space.sample()  # a random next action
        return action

    def learn(self, episode: Episode):
        """ Sample batch from Episode and train NN on sample"""
        # TODO: COMPLETE THE CODE
        if episode.size >= self.batch_size:
            # vanaf een bepaalde batch_size ga je dus een random sample nemen uit de episode
            percepts = episode.sample(self.batch_size)
            self.learn_from_batch(percepts)
        super().learn(episode)

    def build_training_set(self, episode: Episode):
        """ Build training set from episode """
        # TODO: COMPLETE THE CODE
        # ReplayMemory voor toevoegen recepts en bufferen
        training_data = deque()
        for p in episode:  # random sample of percepts
            s = p.state
            a = p.action
            r = p.reward
            s2 = p.next_state
            done = p.done

            # dit een vektor met een waarde voor actie 1 en een waarde voor actie 2.  Voor CartPole dus 2 waardes
            q_values = self.q1.predict(np.reshape(s, [1, self.env.state_size]))
            if self.ddqn:
                # vraag q1 wat zijn voorspelling zou zijn voor de next state, geef dus het argument dat hoort bij de
                # maximale q-waarde
                optimal_a = np.argmax(self.q1.predict(np.reshape(s2, [1, self.env.state_size])))
                optimal_q = self.q2.predict(np.reshape(s2, [1, self.env.state_size]))[optimal_a]
            else:
                # idem opnieuw een vektor met 2 waardes voor de cartpole
                optimal_q = np.max(self.q2.predict(np.reshape(s2, [1,
                                                                   self.env.state_size])))  # Q2 wordt gebruikt om een training set te bouwen voor Q1

            if done:
                q_values[0][a] = r
            else:
                q_values[0][a] = r + self.γ * optimal_q
            training_data.append(
                (s, q_values[0][a]))  # koppeling van huidige state aan toekomstige informatie (predictie 2de netwerk)
        return training_data

    def train_network(self, training_set):
        """ Train neural net q1 on training set  """
        # TODO: COMPLETE THE CODE
        state = np.asarray([np.asarray(a) for a in np.transpose(list(training_set))[0][:]]).astype('float32')
        qval = np.asarray(np.transpose(list(training_set))[1]).astype('float32')
        self.q1.fit(state, qval, batch_size=self.batch_size, verbose=0)

