from abc import abstractmethod

from be.kdg.rl.agent.episode import Episode
from be.kdg.rl.agent.percept import Percept
from be.kdg.rl.environment.environment import Environment
from be.kdg.rl.learning.learningstrategy import LearningStrategy
from be.kdg.rl.learning.tabular.tabular_learning import TabularLearner


class Agent:

    def __init__(self, environment: Environment, learning_strategy: LearningStrategy, n_episodes=10000):
        super().__init__()
        self.env = environment
        self.learning_strategy = learning_strategy
        self.episodes: [Episode] = []
        self.n_episodes = n_episodes  # total episodes
        self.episode_count = 0

    @abstractmethod
    def train(self) -> None:
        pass

    @property
    def done(self):
        return self.episode_count > self.n_episodes


class TabularAgent(Agent):

    def __init__(self, environment: Environment, learning_strategy: TabularLearner, n_episodes=10000) -> None:
        super().__init__(environment, learning_strategy, n_episodes)

    def train(self) -> None:
        super(TabularAgent, self).train()

        # as longs as the agents hasn't reached the maximum number of episodes
        while not self.done:

            # start a new episode
            episode = Episode(self.env)
            self.episodes.append(episode)
            # initialize the start state
            state = self.env.reset()
            # reset the learning strategy
            self.learning_strategy.on_learning_start()

            # while the episode isn't finished by length
            while not self.learning_strategy.done():

                # learning strategy (policy) determines next action to take
                action = self.learning_strategy.next_action(state)
                # agent observes the results of his action: the next_state and the corresponding reward
                observation = self.env.step(action)[:-1]

                # render environment (don't render every step, only every X-th, or at the end of the learning process)
                # self.env.render()

                # create Percept from s,a,r,s' and add to Episode
                percept = Percept((state, action) + observation)
                episode.add(percept)

                # learn from one or more Percepts in the Episode
                self.learning_strategy.learn(episode)

                # update state
                state = percept.next_state

                # break if episode is over
                if percept.done:
                    break

            # end episode
            self.episode_count += 1

        self.env.close()
